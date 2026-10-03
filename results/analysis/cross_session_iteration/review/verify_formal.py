"""Independently verify formal cross-session iteration artifacts on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ssm_decode.data import load_recording


ROOT = Path(__file__).resolve().parents[4]
NORMALIZER_KEYS = ("x_mean", "x_std", "y_mean", "y_std")
IDENTITY_KEYS = ("task", "method", "seed", "lr")
FIXED_KEYS = (
    "steps", "max_steps", "support_trials", "query_cutoff_trials",
    "normalization", "validation_split", "policy", "context", "batch_size",
    "eval_batch_size", "val_interval", "rank", "alpha", "lora_scope",
    "weight_decay", "train_input_bias", "data_root",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_sha(value):
    return array_sha(value.detach().cpu().contiguous().numpy())


def lr_token(lr):
    return format(float(lr), ".8g").replace(".", "p").replace("-", "m")


def expected_fits(root, matrix):
    for task in matrix["tasks"]:
        for method, lrs in matrix["methods"].items():
            seeds = [0] if method == "none" else matrix["seeds"]
            for seed in seeds:
                for lr in lrs:
                    path = root / task / method / f"seed{seed}" / f"lr{lr_token(lr)}"
                    yield task, method, seed, lr, path


def physical_r2(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    sse_channels = np.square(truth - prediction).sum(0)
    sst_channels = np.square(truth - truth.mean(0, keepdims=True)).sum(0)
    sse = float(sse_channels.sum())
    sst = float(sst_channels.sum())
    return {"r2": float(1.0 - sse / sst), "sse": sse, "sst": sst,
            "n_bins": int(len(truth))}


def expected_code(provenance):
    return {Path(name).name: digest for name, digest in provenance["code_sha256"].items()}


def task_data(matrix, task):
    from ssm_decode.data import source_target_plan

    plan = source_target_plan(task, root=Path(matrix["data_root"]))
    session = plan["cross_session_local_dev"]["target_session"]
    record = load_recording(task, "held_in", session, root=Path(matrix["data_root"]))
    usable = [tuple(bound) for bound in record.trial_bounds if bound[1] - bound[0] >= 50]
    support = usable[:matrix["support_trials"]]
    query = usable[matrix["query_cutoff_trials"]:]
    val_indices = set(range(4, matrix["support_trials"], 5))
    if len(val_indices) < math.ceil(matrix["support_trials"] / 5):
        val_indices.add(matrix["support_trials"] - 1)
    train = [bound for index, bound in enumerate(support) if index not in val_indices]
    valid = [bound for index, bound in enumerate(support) if index in val_indices]

    def selected_indices(bounds, offset=0):
        values = np.concatenate([np.arange(start + offset, end) for start, end in bounds])
        return values[record.eval_mask[values]]

    support_indices = selected_indices(support)
    allvalid_indices = selected_indices(query)
    legacy_indices = selected_indices(query, 49)
    prefix_rows = np.concatenate([record.neural[start:end] for start, end in support], axis=0)
    x_mean = prefix_rows.astype(np.float64).mean(0).astype(np.float32)
    x_std = prefix_rows.astype(np.float64).std(0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    return {
        "plan": plan, "record": record, "support": support, "train": train,
        "valid": valid, "query": query, "support_indices": support_indices,
        "allvalid_indices": allvalid_indices, "legacy_indices": legacy_indices,
        "x_mean": x_mean, "x_std": x_std, "target_sha256": file_sha(record.path),
    }


def add_error(errors, fit, message):
    errors.append({"fit": str(fit), "message": message})


def pinned_runtime(errors, root):
    """Read the runtime through the same isolated dependency path as formal jobs."""
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
        add_error(errors, root, "pinned runtime identity could not be read")
        return None
    try:
        return json.loads(lines[0][len(prefix):])
    except json.JSONDecodeError:
        add_error(errors, root, "pinned runtime identity is invalid JSON")
        return None


def validate_root(root, matrix_path, matrix, provenance, errors):
    if file_sha(matrix_path) != provenance.get("matrix_sha256"):
        add_error(errors, root, "frozen matrix hash differs")
    launcher = root / "code_snapshot" / "run_cross_session_iteration.py"
    if not launcher.is_file() or file_sha(launcher) != provenance.get("launcher_sha256"):
        add_error(errors, root, "launcher snapshot hash differs")
    for relative, digest in provenance.get("code_sha256", {}).items():
        snapshot = root / "code_snapshot" / relative
        if not snapshot.is_file() or file_sha(snapshot) != digest:
            add_error(errors, root, f"code snapshot hash differs: {relative}")
    for task, values in provenance.get("source_inputs", {}).items():
        for kind in ("checkpoint", "normalizer"):
            path = Path(values[f"{kind}_path"])
            if not path.is_file() or file_sha(path) != values[f"{kind}_sha256"]:
                add_error(errors, root, f"live source {kind} differs: {task}")
    if matrix.get("schema") != "cross_session_iteration_a_v1":
        add_error(errors, root, "matrix schema differs")


def validate_normalizer(path, task_info, source_normalizer, manifest, checkpoints, errors):
    normalizer_path = path / "normalizer.npz"
    if not normalizer_path.is_file() or file_sha(normalizer_path) != manifest.get("normalizer_sha256"):
        add_error(errors, path, "normalizer hash differs")
        return None
    with np.load(normalizer_path) as saved, np.load(source_normalizer) as source:
        values = {key: saved[key].copy() for key in NORMALIZER_KEYS}
        if not np.array_equal(values["x_mean"], task_info["x_mean"]):
            add_error(errors, path, "support x_mean differs")
        if not np.array_equal(values["x_std"], task_info["x_std"]):
            add_error(errors, path, "support x_std differs")
        for key in ("y_mean", "y_std"):
            if not np.array_equal(values[key], source[key]):
                add_error(errors, path, f"source {key} changed")
    for name, checkpoint in checkpoints.items():
        for key in NORMALIZER_KEYS:
            if not np.array_equal(np.asarray(checkpoint["normalizer"][key]), values[key]):
                add_error(errors, path, f"{name} checkpoint normalizer differs: {key}")
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": array_sha(value)} for key, value in values.items()}


def validate_fit(task, method, seed, lr, path, matrix, provenance, info, runtime, errors):
    required = ("fit_result.json", "manifest.json", "train_log.json", "best.pt", "last.pt")
    if not (path / "fit_result.json").is_file():
        return None
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        add_error(errors, path, f"completed fit artifacts are missing: {missing}")
        return None
    result = read_json(path / "fit_result.json")
    manifest = read_json(path / "manifest.json")
    log = read_json(path / "train_log.json")
    identity = {"task": task, "method": method, "seed": seed, "lr": lr}
    if result.get("status") != "completed":
        add_error(errors, path, "fit status is not completed")
    for key, value in identity.items():
        if result.get(key) != value or manifest.get("args", {}).get(key) != value:
            add_error(errors, path, f"fit identity differs: {key}")
    for key in FIXED_KEYS:
        if manifest.get("args", {}).get(key) != matrix[key]:
            add_error(errors, path, f"matrix argument differs: {key}")
    args = manifest.get("args", {})
    if args.get("pretrained") != matrix["pretrained_by_task"][task]:
        add_error(errors, path, "source checkpoint path differs")
    if args.get("defer_query") is not True or args.get("device") != "cuda:0":
        add_error(errors, path, "fit execution arguments differ")
    if args.get("ui_checkpoint") is not None or manifest.get("ui_checkpoint_sha256") is not None:
        add_error(errors, path, "source-initialized full contract differs")
    code = expected_code(provenance)
    if result.get("code_hashes") != code or manifest.get("code_hashes") != code:
        add_error(errors, path, "fit code hashes differ")
    source = provenance["source_inputs"][task]
    for item_name, item in (("result", result), ("manifest", manifest)):
        if item.get("source_checkpoint_sha256") != source["checkpoint_sha256"]:
            add_error(errors, path, f"{item_name} source checkpoint hash differs")
        if item.get("source_normalizer_sha256") != source["normalizer_sha256"]:
            add_error(errors, path, f"{item_name} source normalizer hash differs")
        if item.get("target_data_sha256") != info["target_sha256"]:
            add_error(errors, path, f"{item_name} target data hash differs")
    if manifest.get("plan") != info["plan"] or manifest.get("target_session") != info["record"].session:
        add_error(errors, path, "target data plan differs")
    for key, expected in (("support_bounds", info["support"]), ("train_bounds", info["train"]),
                          ("val_bounds", info["valid"]), ("query_bounds", info["query"])):
        if manifest.get(key) != [list(value) for value in expected]:
            add_error(errors, path, f"{key} differs")
    for key in ("prefix_before_query_verified", "query_labels_masked"):
        if manifest.get(key) is not True:
            add_error(errors, path, f"{key} is not true")
    if manifest.get("query_used_for_selection") is not False or result.get("query_used_for_selection") is not False:
        add_error(errors, path, "query selection flag differs")
    if result.get("query_evaluated") is not False or manifest.get("held_out_accessed") is not False:
        add_error(errors, path, "fit access flag differs")
    if result.get("receipt") != manifest.get("receipt"):
        add_error(errors, path, "fit receipt differs")
    manifest_runtime = manifest.get("runtime_identity")
    if runtime is not None and manifest_runtime != runtime:
        add_error(errors, path, "fit runtime differs from current pinned runtime")
    if (not isinstance(manifest_runtime, dict) or
            manifest.get("torch_version") != manifest_runtime.get("torch") or
            manifest.get("cuda_version") != manifest_runtime.get("cuda")):
        add_error(errors, path, "fit runtime aliases differ")

    best_path, last_path = path / "best.pt", path / "last.pt"
    if result.get("best_checkpoint_sha256") != file_sha(best_path):
        add_error(errors, path, "best checkpoint hash differs")
    if result.get("last_checkpoint_sha256") != file_sha(last_path):
        add_error(errors, path, "last checkpoint hash differs")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    checkpoints = {"best": best, "last": last}
    for name, checkpoint in checkpoints.items():
        if checkpoint.get("args") != args or checkpoint.get("receipt") != manifest.get("receipt"):
            add_error(errors, path, f"{name} checkpoint contract differs")
        if checkpoint.get("code_hashes") != code:
            add_error(errors, path, f"{name} checkpoint code hashes differ")
        if checkpoint.get("target_data_sha256") != info["target_sha256"]:
            add_error(errors, path, f"{name} checkpoint target hash differs")
        if checkpoint.get("runtime_identity") != manifest.get("runtime_identity"):
            add_error(errors, path, f"{name} checkpoint runtime differs")
    if best.get("best_step") != result.get("best_step"):
        add_error(errors, path, "best checkpoint step differs")
    if last.get("best_step") != result.get("completed_steps"):
        add_error(errors, path, "last checkpoint step differs")

    initial_hashes = manifest.get("initial_parameter_hashes", {})
    names = set(initial_hashes)
    for name, checkpoint in checkpoints.items():
        saved_hashes = manifest.get(f"{name}_parameter_hashes", {})
        state = checkpoint.get("state_dict", {})
        if set(saved_hashes) != names or not names.issubset(state):
            add_error(errors, path, f"{name} parameter names differ")
            continue
        actual_hashes = {key: tensor_sha(state[key]) for key in names}
        if actual_hashes != saved_hashes:
            add_error(errors, path, f"{name} parameter hashes differ")
        changed = sorted(key for key in names if initial_hashes[key] != saved_hashes[key])
        if changed != sorted(manifest.get("changed_parameter_names", {}).get(name, [])):
            add_error(errors, path, f"{name} changed parameter list differs")
    best_buffers = {key: value for key, value in best.get("state_dict", {}).items() if key not in names}
    last_buffers = {key: value for key, value in last.get("state_dict", {}).items() if key not in names}
    if set(best_buffers) != set(last_buffers) or any(not torch.equal(value, last_buffers[key])
                                                     for key, value in best_buffers.items()):
        add_error(errors, path, "checkpoint buffers differ")

    stages = manifest.get("optimizer_stages", [])
    allowed = set()
    stage_steps = []
    state = best.get("state_dict", {})
    initial_stage_names = (set() if not stages else {
        name for group in stages[0].get("groups", []) for name in group.get("names", [])
    })
    for stage_index, stage in enumerate(stages):
        stage_steps.append(stage.get("step"))
        stage_names = []
        for group in stage.get("groups", []):
            allowed.update(group.get("names", []))
            stage_names.extend(group.get("names", []))
            if (not set(group.get("names", [])).issubset(state) or
                    group.get("count") != sum(state[name].numel() for name in group.get("names", []) if name in state)):
                add_error(errors, path, f"optimizer stage {stage_index} parameter count differs")
            if group.get("weight_decay") not in (0.0, matrix["weight_decay"]):
                add_error(errors, path, f"optimizer stage {stage_index} weight decay differs")
            expected_lr = lr if set(group.get("names", [])).issubset(initial_stage_names) else lr * 0.5
            if group.get("lr") != expected_lr:
                add_error(errors, path, f"optimizer stage {stage_index} learning rate differs")
        if len(stage_names) != len(set(stage_names)):
            add_error(errors, path, "optimizer stage contains duplicate parameter names")
    for name in ("best", "last"):
        if not set(manifest.get("changed_parameter_names", {}).get(name, [])).issubset(allowed):
            add_error(errors, path, f"{name} changed a frozen parameter")
    if method == "none":
        if stages or result.get("initial_trainable_params") != 0:
            add_error(errors, path, "none method has an optimizer")
    else:
        if not stages or stages[0].get("step") != 0:
            add_error(errors, path, "initial optimizer stage is missing")
        if result.get("initial_trainable_params") != manifest.get("receipt", {}).get("trainable_count"):
            add_error(errors, path, "initial trainable count differs")
        initial_names = initial_stage_names
        if initial_names != set(manifest.get("receipt", {}).get("trainable_paths", [])):
            add_error(errors, path, "initial optimizer allowlist differs from adaptation receipt")
        if (result.get("initial_trainable_params") != sum(state[name].numel() for name in initial_names if name in state) or
                result.get("peak_trainable_params") != sum(state[name].numel() for name in allowed if name in state)):
            add_error(errors, path, "optimizer trainable parameter count differs")
        if not manifest.get("changed_parameter_names", {}).get("last"):
            add_error(errors, path, "trainable method changed no parameter")
    if method == "full" and stage_steps != [0, 200, 400]:
        add_error(errors, path, "full optimizer stages differ")
    for audit in manifest.get("frozen_audits", []):
        if audit.get("status") != "passed":
            add_error(errors, path, "frozen parameter audit failed")

    values = [(item["step"], item["prefix_val_r2"]) for item in log if "prefix_val_r2" in item]
    if not values or any(not np.isfinite(value) for _, value in values):
        add_error(errors, path, "validation log is invalid")
    else:
        best_value = max(value for _, value in values)
        best_step = min(step for step, value in values if value == best_value)
        if best_step != result.get("best_step") or best_value != result.get("best_prefix_val_r2"):
            add_error(errors, path, "saved validation argmax differs")
    completed = result.get("completed_steps")
    if [item.get("step") for item in log] != list(range(completed + 1)):
        add_error(errors, path, "training log steps differ")
    for item in log[1:]:
        if not np.isfinite(item.get("loss", np.nan)) or not np.isfinite(item.get("gradient_norm", np.nan)):
            add_error(errors, path, "training log contains a nonfinite value")
            break
    expected_extensions = []
    expected_budget = 0 if method == "none" else matrix["steps"]
    while expected_budget and expected_budget < matrix["max_steps"]:
        available = [(step, value) for step, value in values if step <= expected_budget]
        if not available:
            break
        best_value = max(value for _, value in available)
        extension_best_step = min(step for step, value in available if value == best_value)
        if extension_best_step < 0.8 * expected_budget:
            break
        new_budget = min(matrix["max_steps"], expected_budget * 2)
        expected_extensions.append({"old_budget": expected_budget, "new_budget": new_budget,
                                    "best_step": extension_best_step,
                                    "reason": "best_step_at_least_80pct_budget"})
        expected_budget = new_budget
    if (result.get("extensions") != expected_extensions or manifest.get("extensions") != expected_extensions or
            result.get("final_budget") != expected_budget or completed != expected_budget):
        add_error(errors, path, "training budget extension receipt differs")
    expected_admissible = method == "none" or result.get("best_step") < 0.8 * result.get("final_budget")
    if result.get("selection_admissible") is not expected_admissible:
        add_error(errors, path, "selection admissibility differs")
    if result.get("convergence_satisfied") is not expected_admissible:
        add_error(errors, path, "convergence flag differs")
    normalizer = validate_normalizer(path, info, source["normalizer_path"], manifest, checkpoints, errors)
    return {
        **identity, "path": str(path), "best_prefix_val_r2": result.get("best_prefix_val_r2"),
        "best_step": result.get("best_step"), "completed_steps": completed,
        "final_budget": result.get("final_budget"), "selection_admissible": result.get("selection_admissible"),
        "convergence_satisfied": result.get("convergence_satisfied"),
        "best_checkpoint_sha256": result.get("best_checkpoint_sha256"),
        "last_checkpoint_sha256": result.get("last_checkpoint_sha256"),
        "target_data_sha256": result.get("target_data_sha256"),
        "normalizer": normalizer, "fit_seconds": result.get("fit_seconds"),
        "peak_cuda_allocated_mb": result.get("peak_cuda_allocated_mb"),
        "initial_trainable_params": result.get("initial_trainable_params"),
        "peak_trainable_params": result.get("peak_trainable_params"),
    }


def validate_selections(root, matrix, provenance, fits, errors):
    path = root / "selections.json"
    if not path.is_file():
        return None, []
    saved = read_json(path)
    if (saved.get("schema") != "cross_session_iteration_selection_v1" or
            saved.get("provenance") != provenance or saved.get("query_metrics_read") is not False):
        add_error(errors, path, "selection provenance differs")
    if set(saved.get("tasks", {})) != set(matrix["tasks"]):
        add_error(errors, path, "selection task set differs")
    index = {(row["task"], row["method"], row["seed"], row["lr"]): row for row in fits}
    expected = {}
    selected_fits = []
    for task in matrix["tasks"]:
        expected[task] = {}
        saved_task = saved.get("tasks", {}).get(task, {})
        warmstarts_path = root / task / "ui_warmstarts.json"
        if not warmstarts_path.is_file() or saved_task.get("ui_warmstarts") != read_json(warmstarts_path):
            add_error(errors, path, f"UI warmstart selection receipt differs: {task}")
        if set(saved_task.get("methods", {})) != set(matrix["methods"]):
            add_error(errors, path, f"selection method set differs: {task}")
        task_target_hashes = set()
        for method, lrs in matrix["methods"].items():
            seeds = [0] if method == "none" else matrix["seeds"]
            candidates = []
            for lr in lrs:
                rows = [index.get((task, method, seed, lr)) for seed in seeds]
                if any(row is None for row in rows):
                    add_error(errors, path, f"selection fit is missing: {task}/{method}/{lr}")
                    continue
                admissible = all(row["selection_admissible"] is True and row["convergence_satisfied"] is True for row in rows)
                receipts = [{"seed": row["seed"], "fit": row["path"],
                             "best_pt_sha256": row["best_checkpoint_sha256"],
                             "best_prefix_val_r2": row["best_prefix_val_r2"],
                             "selection_admissible": row["selection_admissible"],
                             "convergence_satisfied": row["convergence_satisfied"],
                             "source_checkpoint_sha256": provenance["source_inputs"][task]["checkpoint_sha256"],
                             "source_normalizer_sha256": provenance["source_inputs"][task]["normalizer_sha256"],
                             "target_data_sha256": row["target_data_sha256"]} for row in rows]
                mean = sum(row["best_prefix_val_r2"] for row in rows) / len(rows)
                excluded = [f"seed {row['seed']}: selection_admissible={row['selection_admissible']}, "
                            f"convergence_satisfied={row['convergence_satisfied']}"
                            for row in rows if not (row["selection_admissible"] and row["convergence_satisfied"])]
                receipt = {"lr": lr, "mean_best_prefix_val_r2": mean, "fits": receipts,
                           "admissible": admissible, "excluded_reasons": excluded}
                candidates.append({"lr": lr, "rows": rows, "mean": mean,
                                   "admissible": admissible, "receipt": receipt})
                task_target_hashes.update(row["target_data_sha256"] for row in rows)
            eligible = [row for row in candidates if row["admissible"]]
            chosen = max(eligible, key=lambda row: (row["mean"], -row["lr"])) if eligible else None
            if chosen is None:
                add_error(errors, path, f"no admissible candidate: {task}/{method}")
                continue
            expected[task][method] = chosen["lr"]
            selected_fits.extend(chosen["rows"])
            saved_method = saved_task.get("methods", {}).get(method, {})
            expected_method = {"all_lr_validation_scores": [row["receipt"] for row in candidates],
                               "selected": chosen["receipt"]}
            if saved_method != expected_method:
                add_error(errors, path, f"complete selection receipt differs: {task}/{method}")
            saved_choice = saved_method.get("selected", {})
            if saved_choice.get("lr") != chosen["lr"]:
                add_error(errors, path, f"selected learning rate differs: {task}/{method}")
            if not np.isclose(saved_choice.get("mean_best_prefix_val_r2", np.nan), chosen["mean"], atol=1e-12, rtol=0):
                add_error(errors, path, f"selected validation mean differs: {task}/{method}")
            saved_rows = saved_choice.get("fits", [])
            if len(saved_rows) != len(chosen["rows"]):
                add_error(errors, path, f"selected seed count differs: {task}/{method}")
            for row in chosen["rows"]:
                matches = [item for item in saved_rows if item.get("seed") == row["seed"]]
                if len(matches) != 1 or matches[0].get("fit") != row["path"] or matches[0].get("best_pt_sha256") != row["best_checkpoint_sha256"]:
                    add_error(errors, path, f"selected fit receipt differs: {task}/{method}/{row['seed']}")
        if len(task_target_hashes) != 1 or saved_task.get("target_data_sha256") not in task_target_hashes:
            add_error(errors, path, f"selection target hash differs: {task}")
    return {"selected_lr": expected, "selected_fit_count": len(selected_fits)}, selected_fits


def validate_evaluation(row, matrix, provenance, info, runtime, errors):
    fit = Path(row["path"])
    path = fit / "query_evaluation"
    metrics_path, predictions_path = path / "metrics.json", path / "predictions.npz"
    if not metrics_path.is_file() or not predictions_path.is_file():
        add_error(errors, fit, "selected query evaluation is missing")
        return None
    metrics = read_json(metrics_path)
    for key in IDENTITY_KEYS:
        if metrics.get(key) != row[key]:
            add_error(errors, fit, f"evaluation identity differs: {key}")
    if metrics.get("status") != "completed" or metrics.get("checkpoint_sha256") != row["best_checkpoint_sha256"]:
        add_error(errors, fit, "evaluation checkpoint contract differs")
    if metrics.get("code_hashes") != expected_code(provenance):
        add_error(errors, fit, "evaluation code hashes differ")
    if runtime is not None and metrics.get("runtime_identity") != runtime:
        add_error(errors, fit, "evaluation runtime differs from current pinned runtime")
    if metrics.get("target_data_sha256") != info["target_sha256"]:
        add_error(errors, fit, "evaluation target hash differs")
    if metrics.get("held_out_accessed") is not False or metrics.get("query_used_for_selection") is not False:
        add_error(errors, fit, "evaluation access flag differs")
    expected_indices = {
        "support_indices": info["support_indices"], "legacy_indices": info["legacy_indices"],
        "allvalid_indices": info["allvalid_indices"],
    }
    score_rows = {}
    with np.load(predictions_path) as data:
        for key, expected in expected_indices.items():
            if not np.array_equal(data[key], expected):
                add_error(errors, fit, f"evaluation cohort differs: {key}")
        if np.intersect1d(data["support_indices"], data["allvalid_indices"]).size:
            add_error(errors, fit, "support and query cohorts overlap")
        truth_map = {
            "legacy": (data["truth_legacy"], info["record"].behavior[data["legacy_indices"]]),
            "all_valid": (data["truth_allvalid"], info["record"].behavior[data["allvalid_indices"]]),
        }
        for cohort, (truth, live_truth) in truth_map.items():
            if not np.array_equal(truth, live_truth):
                add_error(errors, fit, f"raw physical truth differs: {cohort}")
            for mode in ("zero", "ridge"):
                suffix = "legacy" if cohort == "legacy" else "allvalid"
                prediction = data[f"{mode}_{suffix}"]
                if not np.isfinite(prediction).all():
                    add_error(errors, fit, f"prediction is nonfinite: {mode}/{cohort}")
                score = physical_r2(truth, prediction)
                reported = metrics.get("final", {}).get(mode, {}).get(cohort, {}).get("r2_variance_weighted")
                if not isinstance(reported, (int, float)) or abs(score["r2"] - reported) > 1e-6:
                    add_error(errors, fit, f"physical R2 differs: {mode}/{cohort}")
                score["reported_r2"] = reported
                score["absolute_delta"] = None if not isinstance(reported, (int, float)) else abs(score["r2"] - reported)
                score_rows[f"{mode}_{cohort}"] = score
        with np.load(fit / "normalizer.npz") as normalizer:
            expected_support_truth = ((info["record"].behavior[data["support_indices"]] - normalizer["y_mean"]) /
                                      normalizer["y_std"])
        if not np.array_equal(data["support_truth_normalized"], expected_support_truth):
            add_error(errors, fit, "support normalized truth differs")
        hash_arrays = {
            "support_indices": data["support_indices"], "legacy_indices": data["legacy_indices"],
            "all_valid_indices": data["allvalid_indices"],
            "support_truth_normalized": data["support_truth_normalized"],
            "legacy_truth_physical": data["truth_legacy"],
            "all_valid_truth_physical": data["truth_allvalid"],
        }
        for key in ("support_indices", "legacy_indices", "all_valid_indices"):
            if metrics.get("cohort_index_hashes", {}).get(key) != array_sha(hash_arrays[key]):
                add_error(errors, fit, f"cohort index hash differs: {key}")
        for key in ("support_truth_normalized", "legacy_truth_physical", "all_valid_truth_physical"):
            if metrics.get("cohort_truth_hashes", {}).get(key) != array_sha(hash_arrays[key]):
                add_error(errors, fit, f"cohort truth hash differs: {key}")
        hashes = {key: array_sha(value) for key, value in hash_arrays.items()}
    return {
        **{key: row[key] for key in IDENTITY_KEYS}, "fit": str(fit),
        "checkpoint_sha256": row["best_checkpoint_sha256"],
        "predictions_sha256": file_sha(predictions_path), "cohort_hashes": hashes,
        "scores": score_rows,
    }


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_path, provenance_path = root / "matrix.json", root / "provenance.json"
    matrix, provenance = read_json(matrix_path), read_json(provenance_path)
    errors = []
    validate_root(root, matrix_path, matrix, provenance, errors)
    runtime = pinned_runtime(errors, root)
    info = {task: task_data(matrix, task) for task in matrix["tasks"]}
    fits = []
    for task, method, seed, lr, path in expected_fits(root, matrix):
        result = validate_fit(task, method, seed, lr, path, matrix, provenance, info[task], runtime, errors)
        if result is not None:
            fits.append(result)
    selections, selected = validate_selections(root, matrix, provenance, fits, errors)
    evaluations = []
    if selections is not None:
        selected_paths = {row["path"] for row in selected}
        for row in selected:
            result = validate_evaluation(row, matrix, provenance, info[row["task"]], runtime, errors)
            if result is not None:
                evaluations.append(result)
        for row in fits:
            if row["path"] not in selected_paths and (Path(row["path"]) / "query_evaluation").exists():
                add_error(errors, row["path"], "unselected fit has a query evaluation")
    for task in matrix["tasks"]:
        rows = [row for row in evaluations if row["task"] == task]
        for key in ("support_indices", "legacy_indices", "all_valid_indices",
                    "support_truth_normalized", "legacy_truth_physical", "all_valid_truth_physical"):
            if rows and len({row["cohort_hashes"][key] for row in rows}) != 1:
                add_error(errors, root, f"evaluation cohort hashes differ: {task}/{key}")
    expected_fit_count = sum((1 if method == "none" else len(matrix["seeds"])) * len(lrs)
                             for method, lrs in matrix["methods"].items()) * len(matrix["tasks"])
    expected_eval_count = sum(1 if method == "none" else len(matrix["seeds"])
                              for method in matrix["methods"]) * len(matrix["tasks"])
    complete = (len(fits) == expected_fit_count and len(evaluations) == expected_eval_count and not errors)
    grouped = []
    groups = defaultdict(list)
    for row in evaluations:
        groups[(row["task"], row["method"])].append(row)
    for (task, method), rows in sorted(groups.items()):
        item = {"task": task, "method": method, "n": len(rows)}
        for score in ("zero_all_valid", "ridge_all_valid", "zero_legacy", "ridge_legacy"):
            values = np.asarray([row["scores"][score]["r2"] for row in rows], dtype=np.float64)
            item[f"{score}_mean"] = float(values.mean())
            item[f"{score}_population_sd"] = float(values.std())
        grouped.append(item)
    report = {
        "schema": "independent_cross_session_iteration_review_v1",
        "status": "verified_complete" if complete else "in_progress_or_failed",
        "root": str(root), "matrix_sha256": provenance.get("matrix_sha256"),
        "launcher_sha256": provenance.get("launcher_sha256"),
        "current_pinned_runtime": runtime,
        "expected_fit_count": expected_fit_count, "completed_fit_count": len(fits),
        "expected_evaluation_count": expected_eval_count,
        "completed_evaluation_count": len(evaluations), "selection": selections,
        "task_data": {task: {"target_session": value["record"].session,
                              "target_sha256": value["target_sha256"],
                              "support_count": int(len(value["support_indices"])),
                              "all_valid_count": int(len(value["allvalid_indices"])),
                              "legacy_count": int(len(value["legacy_indices"]))}
                      for task, value in info.items()},
        "grouped_scores": grouped, "fits": fits, "evaluations": evaluations,
        "known_limitations": [
            "For io and full, root_input_bias_trainable and root_bias_behavior describe the LoRA bias option. "
            "Use trainable_paths, optimizer groups, and parameter hashes for the actual IO/full scope."
        ],
        "errors": errors,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(output.with_suffix(".fits.csv"), fits)
    flat_evaluations = []
    for row in evaluations:
        flat = {key: row[key] for key in (*IDENTITY_KEYS, "fit", "checkpoint_sha256", "predictions_sha256")}
        for name, score in row["scores"].items():
            for key, value in score.items():
                flat[f"{name}_{key}"] = value
        flat_evaluations.append(flat)
    write_csv(output.with_suffix(".evaluations.csv"), flat_evaluations)
    print(json.dumps({key: report[key] for key in ("status", "completed_fit_count",
                                                    "completed_evaluation_count", "errors")}, indent=2))
    if args.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
