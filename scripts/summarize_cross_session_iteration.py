"""Verify and summarize a saved cross-session iteration matrix on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ssm_decode.data import load_recording, sha256_file


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def r2(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    sse = np.square(truth - prediction).sum(axis=0)
    sst = np.square(truth - truth.mean(axis=0)).sum(axis=0)
    return float(1.0 - sse.sum() / max(float(sst.sum()), 1e-300))


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expected_fits(root, matrix):
    for task in matrix["tasks"]:
        for method, lrs in matrix["methods"].items():
            for seed in ([0] if method == "none" else matrix["seeds"]):
                for lr in lrs:
                    token = format(float(lr), ".8g").replace(".", "p").replace("-", "m")
                    yield task, method, seed, lr, root / task / method / f"seed{seed}" / f"lr{token}"


def fit_path(root, task, method, seed, lr):
    token = format(float(lr), ".8g").replace(".", "p").replace("-", "m")
    return root / task / method / f"seed{seed}" / f"lr{token}"


def check_fit(path, task, method, seed, lr, provenance, live_target):
    required = [path / name for name in ("fit_result.json", "manifest.json", "train_log.json", "best.pt", "last.pt")]
    if any(not item.is_file() for item in required):
        return None, [f"missing fit artifact: {path}"]
    result, manifest, log = read(path / "fit_result.json"), read(path / "manifest.json"), read(path / "train_log.json")
    errors = []
    if result.get("status") != "completed": errors.append("fit status is not completed")
    if (result.get("task"), result.get("method"), result.get("seed"), result.get("lr")) != (task, method, seed, lr): errors.append("fit identity differs")
    frozen = provenance["source_inputs"][task]
    for item in (result, manifest):
        if item.get("source_checkpoint_sha256") != frozen["checkpoint_sha256"]: errors.append("source checkpoint hash differs")
        if item.get("source_normalizer_sha256") != frozen["normalizer_sha256"]: errors.append("source normalizer hash differs")
        if item.get("code_hashes") != {Path(name).name: value for name, value in provenance["code_sha256"].items()}: errors.append("code hashes differ")
        if item.get("target_data_sha256") != live_target["sha256"]: errors.append("target data hash differs")
    if result.get("best_checkpoint_sha256") != sha(path / "best.pt"): errors.append("best checkpoint hash differs")
    if result.get("last_checkpoint_sha256") != sha(path / "last.pt"): errors.append("last checkpoint hash differs")
    try:
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=False)
        if best.get("args") != last.get("args"): errors.append("checkpoint args differ")
        if best.get("receipt") != manifest.get("receipt"): errors.append("best receipt differs")
        if best.get("best_step") != result.get("best_step"): errors.append("best step differs")
        initial, final = manifest.get("initial_parameter_hashes", {}), manifest.get("best_parameter_hashes", {})
        allowed = {name for stage in manifest.get("optimizer_stages", []) for group in stage.get("groups", []) for name in group.get("names", [])}
        changed = {name for name, value in initial.items() if final.get(name) != value}
        if not changed.issubset(allowed): errors.append("a frozen parameter changed")
        if method == "full" and not manifest.get("optimizer_stages"): errors.append("full has no optimizer stages")
    except Exception as exc:
        errors.append(f"checkpoint read failed: {type(exc).__name__}")
    values = [(item["step"], item["prefix_val_r2"]) for item in log if "prefix_val_r2" in item]
    if values:
        best_value = max(value for _, value in values)
        earliest = min(step for step, value in values if value == best_value)
        if earliest != result.get("best_step"): errors.append("best step is not earliest validation argmax")
    else:
        errors.append("train log has no prefix validation")
    row = {"fit": str(path), "task": task, "method": method, "seed": seed, "lr": lr,
           "completed": result.get("status") == "completed", "best_prefix_val_r2": result.get("best_prefix_val_r2"),
           "best_step": result.get("best_step"), "completed_steps": result.get("completed_steps"),
           "selection_admissible": result.get("selection_admissible"), "convergence_satisfied": result.get("convergence_satisfied"),
           "extensions": json.dumps(result.get("extensions", [])), "fit_seconds": result.get("fit_seconds"),
           "peak_cuda_allocated_mb": result.get("peak_cuda_allocated_mb"), "target_data_sha256": result.get("target_data_sha256"),
           "source_checkpoint_sha256": result.get("source_checkpoint_sha256"), "source_normalizer_sha256": result.get("source_normalizer_sha256"),
           "verified": not errors, "errors": " | ".join(errors)}
    return (row, result, manifest), errors


def check_evaluation(path, selected, fit, expected_code, target_hash, fit_result, live_target):
    required = [path / "metrics.json", path / "predictions.npz"]
    if any(not item.is_file() for item in required): return None, [f"missing selected evaluation: {path}"]
    metrics = read(path / "metrics.json"); errors = []
    if metrics.get("status") != "completed": errors.append("evaluation status is not completed")
    if metrics.get("checkpoint_sha256") != selected["best_pt_sha256"]: errors.append("evaluation checkpoint hash differs")
    if metrics.get("code_hashes") != expected_code: errors.append("evaluation code hashes differ")
    if metrics.get("target_data_sha256") != target_hash: errors.append("evaluation target hash differs")
    for key in ("task", "method", "seed", "lr"):
        if metrics.get(key) != fit_result.get(key): errors.append(f"evaluation {key} differs from selected fit")
    data = np.load(path / "predictions.npz")
    row = {"fit": str(fit), "task": metrics.get("task"), "method": metrics.get("method"), "seed": metrics.get("seed"),
           "lr": metrics.get("lr"), "verified": not errors, "errors": " | ".join(errors),
           "all_valid_count": int(len(data["allvalid_indices"])), "legacy_count": int(len(data["legacy_indices"])),
           "r2_oracle_tolerance": 2e-6, "r2_oracle_abs_error_max": 0.0}
    for scope, suffix in (("allvalid", "all_valid"), ("legacy", "legacy")):
        truth, indices = data[f"truth_{scope}"], data[f"{scope}_indices"]
        row[f"{suffix}_index_hash"] = array_sha(indices); row[f"{suffix}_truth_hash"] = array_sha(truth)
        for name in ("zero", "ridge"):
            row[f"{name}_{suffix}_r2"] = r2(truth, data[f"{name}_{scope}"])
            reported = metrics.get("final", {}).get(name, {}).get(suffix, {}).get("r2_variance_weighted")
            if reported is None:
                errors.append(f"saved {name} {suffix} R2 is missing")
            else:
                delta = abs(row[f"{name}_{suffix}_r2"] - reported)
                row["r2_oracle_abs_error_max"] = max(row["r2_oracle_abs_error_max"], delta)
                if delta > 2e-6: errors.append(f"saved {name} {suffix} R2 differs")
        live_truth = live_target["behavior"][indices]
        if not np.array_equal(truth, live_truth): errors.append(f"{suffix} truth differs from live physical target labels")
    support = data["support_indices"]
    if np.intersect1d(support, data["allvalid_indices"]).size: errors.append("support and query overlap")
    row["verified"] = not errors; row["errors"] = " | ".join(errors)
    return row, errors


def live_targets(matrix, manifests):
    values = {}
    for task, manifest in manifests.items():
        session = manifest["target_session"]
        record = load_recording(task, "held_in", session, root=Path(matrix["data_root"]))
        values[task] = {"sha256": sha256_file(record.path), "behavior": record.behavior}
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv); root, output = args.root.resolve(), args.output.resolve()
    matrix, provenance = read(root / "matrix.json"), read(root / "provenance.json")
    selections = read(root / "selections.json") if (root / "selections.json").is_file() else {"tasks": {}}
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for task, method, seed, lr, path in expected_fits(root, matrix):
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            manifests.setdefault(task, read(manifest_path))
    live_by_task = live_targets(matrix, manifests) if manifests else {}
    fit_rows, errors, fit_index = [], [], {}
    for task, method, seed, lr, path in expected_fits(root, matrix):
        if task not in live_by_task and (path / "fit_result.json").is_file():
            errors.append(f"cannot identify live target for {task}")
            continue
        item, found = check_fit(path, task, method, seed, lr, provenance, live_by_task.get(task, {"sha256": None, "behavior": None})); errors += found
        if item is not None:
            row, result, manifest = item; fit_rows.append(row); fit_index[str(path)] = (row, result, manifest)
    expected_code = {Path(name).name: digest for name, digest in provenance["code_sha256"].items()}
    selected_rows, diagnostics = [], {"query_metrics_read_for_selection": False, "methods": []}
    for task, task_selection in selections.get("tasks", {}).items():
        for method, info in task_selection.get("methods", {}).items():
            candidates, chosen = info["all_lr_validation_scores"], info["selected"]
            by_lr = {float(candidate["lr"]): candidate for candidate in candidates}
            rebuilt = []
            expected_seeds = [0] if method == "none" else matrix["seeds"]
            for lr in matrix["methods"][method]:
                candidate = by_lr.get(float(lr))
                if candidate is None:
                    errors.append(f"selection candidate is missing: {task}/{method}/{lr}")
                    continue
                if len(candidate.get("fits", [])) != len(expected_seeds):
                    errors.append(f"selection candidate seed count differs: {task}/{method}/{lr}")
                rows = []
                for seed in expected_seeds:
                    expected_path = str(fit_path(root, task, method, seed, lr))
                    saved = next((row for row in candidate["fits"] if row.get("fit") == expected_path and row.get("seed") == seed), None)
                    if saved is None:
                        errors.append(f"selection candidate path differs: {task}/{method}/{lr}/seed{seed}")
                        continue
                    entry = fit_index.get(expected_path)
                    if entry is None:
                        errors.append(f"candidate fit missing: {expected_path}"); continue
                    result = entry[1]
                    rows.append(result)
                admissible = len(rows) == len(candidate["fits"]) and all(row.get("selection_admissible") is True and row.get("convergence_satisfied") is True for row in rows)
                mean = None if not rows else float(np.mean([row["best_prefix_val_r2"] for row in rows], dtype=np.float64))
                if mean is not None and not np.isclose(mean, candidate["mean_best_prefix_val_r2"], atol=1e-12, rtol=0): errors.append(f"saved candidate mean differs: {task}/{method}/{candidate['lr']}")
                if admissible != candidate.get("admissible"): errors.append(f"saved candidate admissibility differs: {task}/{method}/{candidate['lr']}")
                rebuilt.append({"lr": candidate["lr"], "mean": mean, "admissible": admissible})
            valid = [row for row in rebuilt if row["admissible"]]
            recomputed = max(valid, key=lambda row: (row["mean"], -row["lr"])) if valid else None
            if recomputed is None or recomputed["lr"] != chosen["lr"]: errors.append(f"selection argmax differs: {task}/{method}")
            if set(by_lr) != {float(lr) for lr in matrix["methods"][method]}:
                errors.append(f"selection candidate LR set differs: {task}/{method}")
            diagnostics["methods"].append({"task": task, "method": method, "chosen_lr": chosen["lr"], "candidates": candidates, "rebuilt": rebuilt})
            for selected in chosen["fits"]:
                fit = Path(selected["fit"]); entry = fit_index.get(str(fit))
                if entry is None: errors.append(f"selected fit missing: {fit}"); continue
                _, result, _ = entry
                evaluation, found = check_evaluation(fit / "query_evaluation", selected, fit, expected_code, result.get("target_data_sha256"), result, live_by_task[task]); errors += found
                if evaluation: selected_rows.append(evaluation)
    for task, method, seed, lr, fit in expected_fits(root, matrix):
        selected_lr = selections.get("tasks", {}).get(task, {}).get("methods", {}).get(method, {}).get("selected", {}).get("lr")
        if selected_lr is not None and lr != selected_lr and (fit / "query_evaluation").exists(): errors.append(f"nonselected LR has query evaluation: {fit}")
    for task in matrix["tasks"]:
        rows = [row for row in selected_rows if row["task"] == task]
        for key in ("all_valid_index_hash", "all_valid_truth_hash", "legacy_index_hash", "legacy_truth_hash"):
            if rows and len({row[key] for row in rows}) != 1:
                errors.append(f"selected evaluations have inconsistent {key} for task {task}")
    groups = defaultdict(list)
    for row in selected_rows: groups[(row["task"], row["method"])].append(row)
    grouped = []
    for (task, method), rows in sorted(groups.items()):
        out = {"task": task, "method": method, "n": len(rows), "normalizer": matrix["normalization"], "policy": matrix["policy"], "validation_split": matrix["validation_split"]}
        for name in ("zero_all_valid_r2", "ridge_all_valid_r2", "zero_legacy_r2", "ridge_legacy_r2"):
            values = np.array([row[name] for row in rows], dtype=np.float64); out[name + "_mean"] = float(values.mean()); out[name + "_population_sd"] = float(values.std())
        grouped.append(out)
    write_csv(output / "fits.csv", fit_rows); write_csv(output / "selected_metrics.csv", selected_rows); write_csv(output / "groupedsummary.csv", grouped)
    (output / "selection_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    expected_train = sum(len(lrs) * (1 if method == "none" else len(matrix["seeds"])) * len(matrix["tasks"]) for method, lrs in matrix["methods"].items())
    expected_evaluations = sum((1 if method == "none" else len(matrix["seeds"])) * len(matrix["tasks"]) for method in matrix["methods"])
    complete = len(fit_rows) == expected_train and len(selected_rows) == expected_evaluations and not errors
    verification = {"schema": "cross_session_iteration_summary_v1", "status": "verified_complete" if complete else "partial_or_unverified", "expected_train_fits": expected_train, "found_train_fits": len(fit_rows), "expected_selected_evaluations": expected_evaluations, "found_selected_evaluations": len(selected_rows), "errors": errors, "raw_physical_truth": True, "r2": "float64 variance-weighted direct=zero primary; ridge also reported; legacy is diagnostic only", "r2_oracle_tolerance": 2e-6, "training_vram_is_resource_separate": True}
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if args.require_complete and not complete: raise SystemExit("matrix is incomplete or failed verification")


if __name__ == "__main__": main()
