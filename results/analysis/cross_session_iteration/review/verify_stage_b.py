"""Independently verify formal Stage B source and adaptation artifacts on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.data import load_recording
from ssm_decode.session_pretraining import source_statistics


ROOT = Path(__file__).resolve().parents[4]
REVIEW = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "session_pretraining_iteration_b.json"
NORMALIZER_KEYS = ("x_mean", "x_std", "y_mean", "y_std")


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    return json.loads(json.dumps(value))


def add_error(errors, artifact, message):
    errors.append({"artifact": str(artifact), "message": message})


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def current_pinned_runtime(errors, artifact):
    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONPATH=f"{ROOT / '.tools' / 'mamba_deps'}{os.pathsep}{ROOT}",
        CUDA_VISIBLE_DEVICES="",
    )
    code = (
        "import json; "
        "from ssm_decode.cross_session_iteration import _runtime_identity; "
        "print('PINNED_RUNTIME=' + json.dumps(_runtime_identity(), sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    prefix = "PINNED_RUNTIME="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if completed.returncode or len(lines) != 1:
        add_error(errors, artifact, "pinned runtime identity could not be read")
        return None
    try:
        return json.loads(lines[0][len(prefix):])
    except json.JSONDecodeError:
        add_error(errors, artifact, "pinned runtime identity is invalid JSON")
        return None


def validate_root(root, stage_a, cfg, provenance, runtime, errors):
    if read_json(root / "matrix.json") != cfg:
        add_error(errors, root, "frozen Stage B matrix differs")
    if CONFIG.is_file() and file_sha(CONFIG) != provenance.get("config_sha256"):
        add_error(errors, root, "live Stage B config differs from frozen provenance")
    launcher_snapshot = root / "code_snapshot" / "run_session_pretraining_iteration.py"
    if not launcher_snapshot.is_file() or file_sha(launcher_snapshot) != provenance.get("launcher_sha256"):
        add_error(errors, root, "Stage B launcher snapshot differs")
    adapt_launcher = ROOT / "scripts" / "run_cross_session_iteration.py"
    if not adapt_launcher.is_file() or file_sha(adapt_launcher) != provenance.get("adapt_launcher_sha256"):
        add_error(errors, root, "adaptation launcher differs")
    for relative, digest in provenance.get("code_sha256", {}).items():
        live = ROOT / relative
        snapshot = root / "code_snapshot" / relative
        if not live.is_file() or file_sha(live) != digest:
            add_error(errors, root, f"live Stage B code differs: {relative}")
        if not snapshot.is_file() or file_sha(snapshot) != digest:
            add_error(errors, root, f"Stage B code snapshot differs: {relative}")
    if runtime is not None and provenance.get("runtime_identity") != runtime:
        add_error(errors, root, "Stage B frozen runtime differs from current pinned runtime")
    if (file_sha(stage_a / "matrix.json") != provenance.get("stage_a_matrix_sha256") or
            read_json(stage_a / "provenance.json") != provenance.get("stage_a_provenance")):
        add_error(errors, root, "Stage A parent provenance differs")
    for task, rows in provenance.get("source_dataset_inputs", {}).items():
        if set(rows) == set():
            add_error(errors, root, f"source dataset receipt is empty: {task}")
        for session, item in rows.items():
            path = Path(item.get("path", ""))
            if not path.is_file() or file_sha(path) != item.get("sha256"):
                add_error(errors, root, f"live source dataset differs: {task}/{session}")


def expected_source_args(path, cfg, task, capacity):
    result = {key: value for key, value in cfg["shared_args"].items()
              if key not in ("mode", "source_only", "window_policy")}
    result.update(
        task=task,
        output=str(path),
        device="cuda:0",
        kind="mamba3_official",
        width=capacity["width"],
        layers=capacity["layers"],
    )
    return result


def expected_source_records(task, cfg, provenance):
    records = []
    root = Path(cfg["shared_args"]["data_root"])
    for session in provenance["source_dataset_inputs"][task]:
        records.append(load_recording(task, "held_in", session, root=root))
    splits = [debug._trial_split(record) for record in records]
    return records, splits, source_statistics(records, splits)


def validate_source(path, task, capacity, cfg, provenance, stage_a, runtime, source_info, errors):
    required = (
        "manifest.json", "best.pt", "normalizer.npz", "session_bank_best.pt",
        "session_bank_last.pt", "train_log.jsonl",
    )
    if not (path / "manifest.json").is_file():
        return None
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        add_error(errors, path, f"completed source artifacts are missing: {missing}")
        return None
    manifest = read_json(path / "manifest.json")
    expected_args = expected_source_args(path, cfg, task, capacity)
    if manifest.get("args") != expected_args:
        add_error(errors, path, "source arguments differ")
    if manifest.get("status") != "completed":
        add_error(errors, path, "source status differs")
    expected_code = {Path(name).name: digest for name, digest in provenance["code_sha256"].items()}
    if manifest.get("code_hashes") != expected_code:
        add_error(errors, path, "source code hashes differ")
    for name, digest in expected_code.items():
        snapshot = path / "code_snapshot" / name
        if not snapshot.is_file() or file_sha(snapshot) != digest:
            add_error(errors, path, f"source code snapshot differs: {name}")
    hashes = {
        "checkpoint_sha256": "best.pt",
        "normalizer_sha256": "normalizer.npz",
        "session_bank_best_sha256": "session_bank_best.pt",
        "train_log_sha256": "train_log.jsonl",
    }
    for key, name in hashes.items():
        if manifest.get(key) != file_sha(path / name):
            add_error(errors, path, f"source artifact hash differs: {name}")
    if runtime is not None and manifest.get("runtime_identity") != runtime:
        add_error(errors, path, "source runtime differs")

    try:
        plain = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        bank = torch.load(path / "session_bank_best.pt", map_location="cpu", weights_only=False)
        bank_last = torch.load(path / "session_bank_last.pt", map_location="cpu", weights_only=False)
    except Exception as exc:
        add_error(errors, path, f"source checkpoint load failed: {type(exc).__name__}")
        return None
    expected_receipt = {key: value for key, value in manifest.items() if key != "checkpoint_sha256"}
    if json_value(plain.get("session_pretraining_receipt", {})) != expected_receipt:
        add_error(errors, path, "embedded source receipt differs")
    if (plain.get("args") != expected_args or plain.get("best_step") != manifest.get("best_step") or
            plain.get("best_source_val_r2") != manifest.get("best_source_val_r2")):
        add_error(errors, path, "plain source checkpoint metadata differs")
    for key in ("args", "code_hashes", "source_data_sha256", "runtime_identity", "source_sessions"):
        if json_value(bank.get(key)) != manifest.get(key):
            add_error(errors, path, f"source bank metadata differs: {key}")
    if bank_last.get("args") != expected_args:
        add_error(errors, path, "last source bank arguments differ")

    events = []
    try:
        events = [json.loads(line) for line in (path / "train_log.jsonl").read_text().splitlines()]
    except (json.JSONDecodeError, OSError):
        add_error(errors, path, "source training log is invalid")
    expected_steps = sorted(set(
        [1, cfg["shared_args"]["steps"]] +
        list(range(cfg["shared_args"]["val_interval"], cfg["shared_args"]["steps"] + 1,
                   cfg["shared_args"]["val_interval"]))
    ))
    if [event.get("step") for event in events] != expected_steps:
        add_error(errors, path, "source validation log steps differ")
    for event in events:
        if any(not isinstance(event.get(key), (int, float)) or not math.isfinite(event[key])
               for key in ("loss", "gradient_norm", "source_val_r2", "elapsed_seconds")):
            add_error(errors, path, "source training log has a nonfinite value")
            break
    if events:
        chosen = max(events, key=lambda event: (event["source_val_r2"], -event["step"]))
        if any(value != chosen[key] for value, key in (
                (manifest.get("best_step"), "step"),
                (manifest.get("best_source_val_r2"), "source_val_r2"),
                (plain.get("best_step"), "step"),
                (plain.get("best_source_val_r2"), "source_val_r2"),
                (bank.get("best_step"), "step"),
                (bank.get("source_val_r2"), "source_val_r2"))):
            add_error(errors, path, "source validation argmax differs")

    records, splits, statistics = source_info
    source_sessions = [record.session for record in records]
    expected_data = {session: provenance["source_dataset_inputs"][task][session]["sha256"]
                     for session in source_sessions}
    parent_plan = read_json(stage_a / task / "none" / "seed0" / "lr0" / "manifest.json")["plan"]
    if manifest.get("plan") != parent_plan:
        add_error(errors, path, "source data plan differs from frozen Stage A plan")
    if (manifest.get("source_sessions") != source_sessions or
            manifest.get("source_train_bounds") != json_value([split[0] for split in splits]) or
            manifest.get("source_val_bounds") != json_value([split[1] for split in splits]) or
            manifest.get("source_data_sha256") != expected_data):
        add_error(errors, path, "source sessions, splits, or data hashes differ")
    saved_statistics = bank.get("source_statistics", [])
    if len(saved_statistics) != len(statistics):
        add_error(errors, path, "source statistic count differs")
    else:
        for index, (saved, expected) in enumerate(zip(saved_statistics, statistics)):
            if len(saved) != len(expected) or any(not np.array_equal(np.asarray(a), b)
                                                   for a, b in zip(saved, expected)):
                add_error(errors, path, f"source statistics differ: session {index}")
    fallback = {
        "x_mean": np.mean([values[0] for values in statistics], axis=0),
        "x_std": np.mean([values[1] for values in statistics], axis=0),
        "y_mean": statistics[0][2],
        "y_std": statistics[0][3],
    }
    try:
        with np.load(path / "normalizer.npz") as saved:
            if set(saved.files) != set(NORMALIZER_KEYS):
                add_error(errors, path, "source normalizer keys differ")
            for key, expected in fallback.items():
                if key not in saved or not np.array_equal(saved[key], expected):
                    add_error(errors, path, f"source fallback normalizer differs: {key}")
    except Exception as exc:
        add_error(errors, path, f"source normalizer load failed: {type(exc).__name__}")

    bank_state = bank.get("state_dict", {})
    plain_state = plain.get("state_dict", {})
    required_bank = ("bank.gain", "bank.bias", "bank.embedding", "base.in_proj.weight", "base.in_proj.bias")
    if any(name not in bank_state for name in required_bank):
        add_error(errors, path, "source bank state is incomplete")
    else:
        gain = bank_state["bank.gain"].mean(0)
        bias = bank_state["bank.bias"].mean(0)
        embedding = bank_state["bank.embedding"].mean(0)
        weight = bank_state["base.in_proj.weight"]
        expected_weight = weight * gain.unsqueeze(0)
        expected_bias = weight @ bias + bank_state["base.in_proj.bias"] + embedding
        if ("in_proj.weight" not in plain_state or "in_proj.bias" not in plain_state or
                not torch.allclose(plain_state["in_proj.weight"], expected_weight, atol=1e-6, rtol=1e-6) or
                not torch.allclose(plain_state["in_proj.bias"], expected_bias, atol=1e-6, rtol=1e-6)):
            add_error(errors, path, "mean source frontend fold differs")
        for name, value in plain_state.items():
            if name not in ("in_proj.weight", "in_proj.bias"):
                source_name = "base." + name
                if source_name not in bank_state or not torch.equal(value, bank_state[source_name]):
                    add_error(errors, path, f"folded source base tensor differs: {name}")
                    break
    for label, state in (("plain", plain_state), ("bank", bank_state)):
        if not state or any(not torch.isfinite(value).all() for value in state.values()):
            add_error(errors, path, f"{label} source checkpoint has a nonfinite tensor")
    base_parameters = sum(value.numel() for name, value in bank_state.items()
                          if name.startswith("base.") and not name.endswith("._extra_state"))
    frontend_parameters = sum(value.numel() for name, value in bank_state.items() if name.startswith("bank."))
    if manifest.get("base_parameters") != base_parameters:
        add_error(errors, path, "source base parameter count differs")
    if manifest.get("session_frontend_parameters") != frontend_parameters:
        add_error(errors, path, "source frontend parameter count differs")
    expected_flags = {
        "source_statistics_scope": "source training neural; eval-valid training labels",
        "policy": "recording_causal_fixed_window",
        "continuous_context": True,
        "label_partitions_masked": True,
        "session_balanced_sampling": True,
        "target_frontend_initialization": "mean of source gain, bias, and latent embedding folded into in_proj",
        "fold_input_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "fold_bf16_output_tolerance": {"atol": 1e-2, "rtol": 1e-2},
        "target_recording_loaded": False,
        "query_used_for_selection": False,
    }
    for key, value in expected_flags.items():
        if manifest.get(key) != value:
            add_error(errors, path, f"source contract differs: {key}")
    for key in ("source_fold_input_max_abs", "source_fold_max_abs"):
        values = manifest.get(key, {})
        if set(values) != set(source_sessions) or any(not isinstance(value, (int, float)) or
                                                       not math.isfinite(value) or value < 0
                                                       for value in values.values()):
            add_error(errors, path, f"source fold diagnostics differ: {key}")
    return {
        "task": task,
        "capacity": capacity["name"],
        "path": str(path),
        "best_step": manifest.get("best_step"),
        "best_source_val_r2": manifest.get("best_source_val_r2"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "normalizer_sha256": manifest.get("normalizer_sha256"),
        "session_bank_best_sha256": manifest.get("session_bank_best_sha256"),
        "train_log_sha256": manifest.get("train_log_sha256"),
        "base_parameters": manifest.get("base_parameters"),
        "session_frontend_parameters": manifest.get("session_frontend_parameters"),
        "fit_seconds": manifest.get("fit_seconds"),
        "peak_cuda_allocated_mb": manifest.get("peak_cuda_allocated_mb"),
        "fold_input_max_abs": max(manifest.get("source_fold_input_max_abs", {}).values(), default=None),
        "fold_output_max_abs": max(manifest.get("source_fold_max_abs", {}).values(), default=None),
    }


def expected_adaptation_matrix(cfg, stage_a, root, task, profile):
    matrix = dict(read_json(stage_a / "matrix.json"))
    selections = read_json(stage_a / "selections.json")
    matrix.update(
        name=f"session_iteration_b_{task}_{profile}",
        tasks=[task],
        physical_gpu_by_task={task: str(cfg["tasks"][task]["physical_gpu"])},
        policy=cfg["adapt_selection"]["policy"],
        seeds=cfg["adapt_selection"]["seeds"],
    )
    matrix["methods"] = {
        method: [selections["tasks"][task]["methods"][method]["selected"]["lr"]]
        for method in cfg["adapt_methods"]
    }
    source = read_json(stage_a / "matrix.json")["pretrained_by_task"][task]
    if profile != "old":
        source = str(root / "source" / task / profile / "best.pt")
    matrix["pretrained_by_task"] = {task: source}
    matrix["lr_transfer_basis"] = "Stage A prefix validation; query scores are never read"
    matrix["parent_stage_a_provenance"] = selections["provenance"]
    if profile != "old":
        source_dir = Path(source).parent
        matrix["source_pretraining_provenance"] = {
            "manifest_sha256": file_sha(source_dir / "manifest.json"),
            "bank_best_sha256": file_sha(source_dir / "session_bank_best.pt"),
            "receipt": read_json(source_dir / "manifest.json"),
        }
    return matrix


def validate_adaptation_matrices(root, stage_a, cfg, source_rows, errors):
    rows = []
    if not (stage_a / "selections.json").is_file():
        return rows
    source_index = {(row["task"], row["capacity"]): row for row in source_rows}
    for task in cfg["tasks"]:
        for profile in cfg["adapt_profiles"]:
            path = root / "adapt_matrices" / f"{task}_{profile}.json"
            adaptation_root = root / "adapt" / profile / task
            if not path.is_file():
                continue
            if profile != "old" and (task, profile) not in source_index:
                add_error(errors, path, "adaptation matrix exists before its verified source fit")
                continue
            expected = expected_adaptation_matrix(cfg, stage_a, root, task, profile)
            if read_json(path) != expected:
                add_error(errors, path, "derived adaptation matrix differs")
            if (adaptation_root / "matrix.json").is_file() and read_json(adaptation_root / "matrix.json") != expected:
                add_error(errors, adaptation_root, "adaptation root matrix differs")
            rows.append({
                "task": task,
                "profile": profile,
                "matrix": str(path),
                "matrix_sha256": file_sha(path),
                "adaptation_root": str(adaptation_root),
            })
    return rows


def run_adaptation_verifiers(rows, output, require_complete, errors):
    reports = []
    for row in rows:
        root = Path(row["adaptation_root"])
        if not (root / "provenance.json").is_file():
            continue
        report_path = output.parent / f"stage_b_{row['profile']}_{row['task']}.json"
        command = [
            sys.executable,
            str(REVIEW / "verify_formal.py"),
            "--root", str(root),
            "--output", str(report_path),
        ]
        if require_complete:
            command.append("--require-complete")
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode and require_complete:
            add_error(errors, root, "nested adaptation verification failed")
        if not report_path.is_file():
            add_error(errors, root, "nested adaptation report is missing")
            continue
        report = read_json(report_path)
        if report.get("errors"):
            for error in report["errors"]:
                add_error(errors, error.get("fit", root), f"nested adaptation: {error.get('message')}")
        reports.append({"task": row["task"], "profile": row["profile"],
                        "report_path": str(report_path), "report": report})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage-a-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-adaptations", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    stage_a = args.stage_a_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = read_json(root / "matrix.json")
    provenance = read_json(root / "provenance.json")
    errors = []
    runtime = current_pinned_runtime(errors, root)
    validate_root(root, stage_a, cfg, provenance, runtime, errors)

    source_info = {
        task: expected_source_records(task, cfg, provenance)
        for task in cfg["tasks"]
    }
    source_rows = []
    source_manifests = {}
    for task in cfg["tasks"]:
        for capacity in cfg["capacities"]:
            path = root / "source" / task / capacity["name"]
            row = validate_source(path, task, capacity, cfg, provenance, stage_a, runtime,
                                  source_info[task], errors)
            if row is not None:
                source_rows.append(row)
                source_manifests[(task, capacity["name"])] = read_json(path / "manifest.json")
    for task in cfg["tasks"]:
        manifests = [source_manifests.get((task, capacity["name"])) for capacity in cfg["capacities"]]
        if all(manifest is not None for manifest in manifests):
            for key in ("source_sessions", "source_train_bounds", "source_val_bounds",
                        "source_data_sha256", "runtime_identity"):
                if manifests[0].get(key) != manifests[1].get(key):
                    add_error(errors, root, f"source capacity controls differ: {task}/{key}")
            banks = [torch.load(root / "source" / task / capacity["name"] / "session_bank_best.pt",
                                map_location="cpu", weights_only=False) for capacity in cfg["capacities"]]
            for first, second in zip(banks[0].get("source_statistics", []),
                                     banks[1].get("source_statistics", [])):
                if any(not np.array_equal(np.asarray(left), np.asarray(right))
                       for left, right in zip(first, second)):
                    add_error(errors, root, f"source capacity statistics differ: {task}")
                    break

    matrix_rows = validate_adaptation_matrices(root, stage_a, cfg, source_rows, errors)
    nested = []
    if args.verify_adaptations or args.require_complete:
        nested = run_adaptation_verifiers(matrix_rows, output, args.require_complete, errors)
    completed_adapt_fits = sum(item["report"].get("completed_fit_count", 0) for item in nested)
    completed_adapt_evaluations = sum(item["report"].get("completed_evaluation_count", 0) for item in nested)
    grouped_scores = []
    for item in nested:
        for group in item["report"].get("grouped_scores", []):
            grouped_scores.append({"profile": item["profile"], **group})
    expected_sources = cfg["counts"]["source_fits"]
    expected_adaptations = cfg["counts"]["adapt_fits"]
    expected_matrices = len(cfg["tasks"]) * len(cfg["adapt_profiles"])
    complete = (
        len(source_rows) == expected_sources and
        len(matrix_rows) == expected_matrices and
        completed_adapt_fits == expected_adaptations and
        completed_adapt_evaluations == expected_adaptations and
        not errors
    )
    report = {
        "schema": "independent_session_pretraining_iteration_b_review_v1",
        "status": "verified_complete" if complete else "in_progress_or_failed",
        "root": str(root),
        "stage_a_root": str(stage_a),
        "config_sha256": provenance.get("config_sha256"),
        "launcher_sha256": provenance.get("launcher_sha256"),
        "current_pinned_runtime": runtime,
        "expected_source_fit_count": expected_sources,
        "completed_source_fit_count": len(source_rows),
        "expected_adaptation_matrix_count": expected_matrices,
        "completed_adaptation_matrix_count": len(matrix_rows),
        "expected_adaptation_fit_count": expected_adaptations,
        "completed_adaptation_fit_count": completed_adapt_fits,
        "expected_adaptation_evaluation_count": expected_adaptations,
        "completed_adaptation_evaluation_count": completed_adapt_evaluations,
        "source_fits": source_rows,
        "adaptation_matrices": matrix_rows,
        "nested_reports": [{key: item[key] for key in ("task", "profile", "report_path")}
                           for item in nested],
        "grouped_scores": grouped_scores,
        "errors": errors,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(output.with_suffix(".sources.csv"), source_rows)
    write_csv(output.with_suffix(".grouped.csv"), grouped_scores)
    print(json.dumps({key: report[key] for key in (
        "status", "completed_source_fit_count", "completed_adaptation_fit_count",
        "completed_adaptation_evaluation_count", "errors")}, indent=2))
    if args.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
