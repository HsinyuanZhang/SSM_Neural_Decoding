"""Run the frozen Stage A cross-session adaptation matrix on two GPU queues."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "ssm_decode.cross_session_iteration"
RESULT_NAME = "fit_result.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lr_token(lr):
    return format(float(lr), ".8g").replace(".", "p").replace("-", "m")


def read_json(path):
    return json.loads(Path(path).read_text())


def _code_hashes(cfg):
    result = {}
    for relative in cfg["code_files"]:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        result[relative] = sha256(source)
    return result


def _source_inputs(cfg):
    inputs = {}
    for task in cfg["tasks"]:
        checkpoint = Path(cfg["pretrained_by_task"][task])
        normalizer = checkpoint.parent / "normalizer.npz"
        if not checkpoint.is_file() or not normalizer.is_file():
            raise FileNotFoundError(f"source checkpoint or normalizer is missing for {task}")
        inputs[task] = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "normalizer_path": str(normalizer), "normalizer_sha256": sha256(normalizer)}
    return inputs


def _provenance(matrix_bytes, cfg):
    launcher = Path(__file__)
    return {
        "schema": "cross_session_iteration_launcher_v1",
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "launcher_sha256": sha256(launcher),
        "code_sha256": _code_hashes(cfg),
        "source_inputs": _source_inputs(cfg),
    }


def _verify_live(provenance, cfg, matrix_bytes):
    current = _provenance(matrix_bytes, cfg)
    if current != provenance:
        raise RuntimeError("matrix, launcher, or runner code changed after matrix freeze")


def _verify_source_inputs(provenance, cfg):
    if _source_inputs(cfg) != provenance.get("source_inputs"):
        raise RuntimeError("source checkpoint or source normalizer bytes changed after matrix freeze")


def _verify_run_live(provenance, cfg, output):
    matrix = output / "matrix.json"
    if not matrix.is_file():
        raise RuntimeError("frozen matrix is missing from output root")
    _verify_live(provenance, cfg, matrix.read_bytes())


def _prepare_root(output, matrix_path):
    matrix_bytes = matrix_path.read_bytes()
    cfg = json.loads(matrix_bytes)
    if output.exists():
        copied = output / "matrix.json"
        provenance_file = output / "provenance.json"
        if not copied.is_file() or not provenance_file.is_file():
            raise RuntimeError("existing output root has no frozen matrix provenance")
        copied_bytes = copied.read_bytes()
        if copied_bytes != matrix_bytes:
            raise RuntimeError("provided matrix differs from the frozen matrix")
        cfg = json.loads(copied_bytes)
        provenance = read_json(provenance_file)
        _verify_live(provenance, cfg, copied_bytes)
        return cfg, provenance
    output.mkdir(parents=True)
    provenance = _provenance(matrix_bytes, cfg)
    (output / "matrix.json").write_bytes(matrix_bytes)
    snapshot = output / "code_snapshot"
    for relative in cfg["code_files"]:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copyfile(Path(__file__), snapshot / "run_cross_session_iteration.py")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return cfg, provenance


def _fit_path(output, task, method, seed, lr):
    return output / task / method / f"seed{seed}" / f"lr{lr_token(lr)}"


def _result_payload(fit):
    candidate = fit / RESULT_NAME
    if candidate.is_file():
        payload = read_json(candidate)
        if payload.get("status") == "completed" and isinstance(payload.get("best_prefix_val_r2"), (int, float)):
            return payload
    return None


def _complete(fit):
    return _result_payload(fit) is not None


def _expected_runner_hashes(cfg):
    return {Path(relative).name: digest for relative, digest in _code_hashes(cfg).items()}


def _fit_manifest(fit):
    path = fit / "manifest.json"
    return read_json(path) if path.is_file() else None


def _current_target_hash(fit):
    manifest = _fit_manifest(fit)
    if not isinstance(manifest, dict):
        return None
    target_session = manifest.get("target_session")
    rows = manifest.get("plan", {}).get("held_in", [])
    row = next((item for item in rows if item.get("session") == target_session), None)
    if row is None or not Path(row.get("path", "")).is_file():
        return None
    return sha256(row["path"])


def _completed_fit_is_current(fit, cfg, provenance, task):
    payload = _result_payload(fit)
    if payload is None or not (fit / "best.pt").is_file() or not (fit / "last.pt").is_file():
        return False
    if payload.get("code_hashes") != _expected_runner_hashes(cfg):
        return False
    frozen = provenance["source_inputs"][task]
    target_hash = _current_target_hash(fit)
    manifest = _fit_manifest(fit)
    if not isinstance(manifest, dict):
        return False
    method = fit.parent.parent.name
    lrs = [lr for lr in cfg["methods"].get(method, []) if fit.name == f"lr{lr_token(lr)}"]
    seeds = [0] if method == "none" else cfg["seeds"]
    if fit.parent.parent.parent.name != task or len(lrs) != 1:
        return False
    seed_candidates = [seed for seed in seeds if fit.parent.name == f"seed{seed}"]
    if len(seed_candidates) != 1:
        return False
    identity = {"task": task, "method": method, "seed": seed_candidates[0], "lr": lrs[0]}
    fit_args = manifest.get("args", {})
    if any(payload.get(key) != value or fit_args.get(key) != value for key, value in identity.items()):
        return False
    fixed = ("steps", "max_steps", "support_trials", "query_cutoff_trials", "normalization",
             "validation_split", "policy", "context", "batch_size", "eval_batch_size",
             "val_interval", "rank", "alpha", "lora_scope", "weight_decay", "train_input_bias", "data_root")
    if any(fit_args.get(key) != cfg[key] for key in fixed):
        return False
    if (fit_args.get("pretrained") != cfg["pretrained_by_task"][task] or
            fit_args.get("device") != "cuda:0" or fit_args.get("defer_query") is not True or
            manifest.get("code_hashes") != _expected_runner_hashes(cfg)):
        return False
    if cfg["full_initialization"] == "source" and fit_args.get("ui_checkpoint") is not None:
        return False
    return (payload.get("best_checkpoint_sha256") == sha256(fit / "best.pt") and
            payload.get("last_checkpoint_sha256") == sha256(fit / "last.pt") and
            payload.get("source_checkpoint_sha256") == frozen["checkpoint_sha256"] and
            payload.get("source_normalizer_sha256") == frozen["normalizer_sha256"] and
            payload.get("target_data_sha256") == target_hash and
            isinstance(manifest, dict) and manifest.get("source_checkpoint_sha256") == frozen["checkpoint_sha256"] and
            manifest.get("source_normalizer_sha256") == frozen["normalizer_sha256"] and
            manifest.get("target_data_sha256") == target_hash)


def _train_command(cfg, task, method, seed, lr, fit, ui_checkpoint=None):
    command = [sys.executable, "-m", RUNNER, "--defer-query", "--task", task,
               "--pretrained", cfg["pretrained_by_task"][task], "--output", str(fit),
               "--method", method, "--device", "cuda:0", "--seed", str(seed), "--lr", str(lr),
               "--steps", str(cfg["steps"]), "--max-steps", str(cfg["max_steps"]),
               "--support-trials", str(cfg["support_trials"]), "--normalization", cfg["normalization"],
               "--query-cutoff-trials", str(cfg["query_cutoff_trials"]),
               "--data-root", cfg["data_root"],
               "--validation-split", cfg["validation_split"], "--policy", cfg["policy"],
               "--context", str(cfg["context"]), "--batch-size", str(cfg["batch_size"]),
               "--eval-batch-size", str(cfg["eval_batch_size"]), "--val-interval", str(cfg["val_interval"]),
               "--rank", str(cfg["rank"]), "--alpha", str(cfg["alpha"]),
               "--lora-scope", cfg["lora_scope"], "--weight-decay", str(cfg["weight_decay"])]
    if cfg["train_input_bias"]:
        command.append("--train-input-bias")
    else:
        command.append("--no-train-input-bias")
    if ui_checkpoint is not None:
        command += ["--ui-checkpoint", str(ui_checkpoint)]
    return command


def _child_env(output, task, physical_gpu):
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = physical_gpu
    env["PYTHONPATH"] = f"{ROOT / '.tools' / 'mamba_deps'}:{ROOT}"
    env["TRITON_CACHE_DIR"] = str(output / ".triton_cache" / f"gpu{physical_gpu}")
    return env


def _run_fit(cfg, provenance, output, task, physical_gpu, method, seed, lr, ui_checkpoint=None):
    _verify_run_live(provenance, cfg, output)
    fit = _fit_path(output, task, method, seed, lr)
    if _completed_fit_is_current(fit, cfg, provenance, task):
        return fit
    if _complete(fit):
        raise RuntimeError(f"completed fit does not match frozen code or checkpoint hashes: {fit}")
    if fit.exists() and any(fit.iterdir()):
        raise RuntimeError(f"incomplete fit exists: {fit}")
    fit.parent.mkdir(parents=True, exist_ok=True)
    command = _train_command(cfg, task, method, seed, lr, fit, ui_checkpoint)
    log = fit.parent / f"{fit.name}.log"
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env=_child_env(output, task, physical_gpu))
    if result.returncode:
        raise RuntimeError(f"fit failed: {fit}; see {log}")
    if not _completed_fit_is_current(fit, cfg, provenance, task):
        raise RuntimeError(f"fit did not create a current completed result, best.pt, and last.pt: {fit}")
    return fit


def _io_warmstarts(cfg, provenance, output, task):
    selected = {}
    for seed in cfg["seeds"]:
        options = []
        for lr in cfg["methods"]["io"]:
            fit = _fit_path(output, task, "io", seed, lr)
            payload = _result_payload(fit)
            if not _completed_fit_is_current(fit, cfg, provenance, task):
                raise RuntimeError(f"missing completed IO fit for UI selection: {fit}")
            options.append({"lr": lr, "fit": str(fit), "best_prefix_val_r2": payload["best_prefix_val_r2"],
                            "best_pt_sha256": sha256(fit / "best.pt"),
                            "selection_admissible": payload.get("selection_admissible") is True,
                            "convergence_satisfied": payload.get("convergence_satisfied") is True})
        eligible = [row for row in options if row["selection_admissible"] and row["convergence_satisfied"]]
        best = None if not eligible else max(eligible, key=lambda row: (row["best_prefix_val_r2"], -row["lr"]))
        if best is None and cfg["full_initialization"] == "selected_io":
            raise RuntimeError(f"no converged IO warmstart for task {task}, seed {seed}")
        selected[str(seed)] = {"candidates": options, "selected": best,
                               "selection_basis": "converged prefix validation only"}
    return selected


def _queue_train(cfg, provenance, output, task, physical_gpu):
    if cfg["full_initialization"] not in {"source", "selected_io"}:
        raise ValueError("full_initialization must be source or selected_io")
    # Run all IO jobs first. The selected-IO full control is explicit in the matrix.
    for seed in cfg["seeds"]:
        for lr in cfg["methods"]["io"]:
            _run_fit(cfg, provenance, output, task, physical_gpu, "io", seed, lr)
    warmstarts = _io_warmstarts(cfg, provenance, output, task)
    (output / task / "ui_warmstarts.json").write_text(json.dumps(warmstarts, indent=2) + "\n")
    for method, lrs in cfg["methods"].items():
        if method == "io":
            continue
        seeds = [0] if method == "none" else cfg["seeds"]
        for seed in seeds:
            for lr in lrs:
                warm = None
                if method == "full" and cfg["full_initialization"] == "selected_io":
                    warm = Path(warmstarts[str(seed)]["selected"]["fit"]) / "best.pt"
                _run_fit(cfg, provenance, output, task, physical_gpu, method, seed, lr, warm)


def _select(cfg, output, provenance, persist=True):
    _verify_run_live(provenance, cfg, output)
    selections = {"schema": "cross_session_iteration_selection_v1", "provenance": provenance,
                  "query_metrics_read": False, "tasks": {}}
    for task in cfg["tasks"]:
        task_data = {"ui_warmstarts": read_json(output / task / "ui_warmstarts.json"), "methods": {}}
        task_target_hashes = set()
        for method, lrs in cfg["methods"].items():
            seeds = [0] if method == "none" else cfg["seeds"]
            candidates = []
            for lr in lrs:
                fits = []
                for seed in seeds:
                    fit = _fit_path(output, task, method, seed, lr)
                    payload = _result_payload(fit)
                    if not _completed_fit_is_current(fit, cfg, provenance, task):
                        raise RuntimeError(f"missing completed fit for selection: {fit}")
                    fits.append({"seed": seed, "fit": str(fit), "best_pt_sha256": sha256(fit / "best.pt"),
                                 "best_prefix_val_r2": payload["best_prefix_val_r2"],
                                 "selection_admissible": payload.get("selection_admissible") is True,
                                 "convergence_satisfied": payload.get("convergence_satisfied") is True,
                                 "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
                                 "source_normalizer_sha256": payload["source_normalizer_sha256"],
                                 "target_data_sha256": payload["target_data_sha256"]})
                target_hashes = {row["target_data_sha256"] for row in fits}
                if len(target_hashes) != 1:
                    raise RuntimeError(f"candidate fits have inconsistent target data hashes for task {task}, method {method}, lr {lr}")
                task_target_hashes.update(target_hashes)
                excluded = [f"seed {row['seed']}: selection_admissible={row['selection_admissible']}, convergence_satisfied={row['convergence_satisfied']}"
                            for row in fits if not (row["selection_admissible"] and row["convergence_satisfied"])]
                candidates.append({"lr": lr, "mean_best_prefix_val_r2": sum(row["best_prefix_val_r2"] for row in fits) / len(fits),
                                   "fits": fits, "admissible": not excluded, "excluded_reasons": excluded})
            eligible = [row for row in candidates if row["admissible"]]
            if not eligible:
                raise RuntimeError(f"no admissible learning rate for task {task}, method {method}")
            chosen = max(eligible, key=lambda row: (row["mean_best_prefix_val_r2"], -row["lr"]))
            task_data["methods"][method] = {"all_lr_validation_scores": candidates, "selected": chosen}
        if len(task_target_hashes) != 1:
            raise RuntimeError(f"fits have inconsistent target data hashes for task {task}")
        task_data["target_data_sha256"] = next(iter(task_target_hashes))
        selections["tasks"][task] = task_data
    if persist:
        (output / "selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    return selections


def _evaluation_is_current(fit, destination, cfg, provenance, task, selected):
    if not _completed_fit_is_current(fit, cfg, provenance, task):
        return False
    metrics_path = destination / "metrics.json"
    predictions = destination / "predictions.npz"
    if not metrics_path.is_file() or not predictions.is_file():
        return False
    metrics = read_json(metrics_path)
    payload = _result_payload(fit)
    fit_args = _fit_manifest(fit)["args"]
    identity = ("task", "method", "seed", "lr", "rank", "lora_scope", "normalization",
                "validation_split", "policy", "support_trials", "query_cutoff_trials")
    if any(metrics.get(key) != fit_args[key] for key in identity):
        return False
    return (metrics.get("status") == "completed" and
            metrics.get("checkpoint_sha256") == selected["best_pt_sha256"] == sha256(fit / "best.pt") and
            metrics.get("code_hashes") == _expected_runner_hashes(cfg) and
            metrics.get("target_data_sha256") == payload.get("target_data_sha256") == _current_target_hash(fit))


def _evaluate(cfg, output, selections, provenance):
    _verify_run_live(provenance, cfg, output)
    errors = []
    def queue(task):
        physical_gpu = cfg["physical_gpu_by_task"][task]
        try:
            for method in cfg["methods"]:
                selected = selections["tasks"][task]["methods"][method]["selected"]
                for row in selected["fits"]:
                    _verify_run_live(provenance, cfg, output)
                    fit = Path(row["fit"])
                    destination = fit / "query_evaluation"
                    if _evaluation_is_current(fit, destination, cfg, provenance, task, row):
                        continue
                    if destination.exists():
                        raise RuntimeError(f"existing query evaluation does not match frozen provenance: {destination}")
                    command = [sys.executable, "-m", RUNNER, "--evaluate-checkpoint", str(fit / "best.pt"),
                               "--output", str(destination), "--device", "cuda:0"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with (fit / "query_evaluation.log").open("w") as stream:
                        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                env=_child_env(output, task, physical_gpu))
                    if result.returncode or not _evaluation_is_current(fit, destination, cfg, provenance, task, row):
                        raise RuntimeError(f"query evaluation failed: {fit}")
        except Exception as exc:
            errors.append((task, str(exc)))
    threads = [threading.Thread(target=queue, args=(task,)) for task in cfg["tasks"]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(str(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("train", "select", "evaluate", "all"), default="all")
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    cfg, provenance = _prepare_root(output, args.matrix.resolve())
    errors = []
    if args.phase in {"train", "all"}:
        def queue(task):
            try:
                _queue_train(cfg, provenance, output, task, cfg["physical_gpu_by_task"][task])
            except Exception as exc:
                errors.append((task, str(exc)))
        threads = [threading.Thread(target=queue, args=(task,)) for task in cfg["tasks"]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise SystemExit(json.dumps(errors))
    if args.phase in {"select", "all"}:
        selections = _select(cfg, output, provenance)
    else:
        selections = read_json(output / "selections.json") if (output / "selections.json").is_file() else None
    if args.phase in {"evaluate", "all"}:
        if selections is None:
            raise RuntimeError("select phase must create selections.json before evaluation")
        if selections != _select(cfg, output, provenance, persist=False):
            raise RuntimeError("saved selections differ from prefix-only recomputation")
        _evaluate(cfg, output, selections, provenance)


if __name__ == "__main__":
    main()
