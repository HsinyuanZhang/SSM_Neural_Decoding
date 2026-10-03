#!/usr/bin/env python3
"""Independent, read-only audit of the formal PEFT artifact matrix.

This oracle intentionally does not import the experiment runner, PEFT adapters,
or summarizer.  It recomputes scores and ridge calibration from saved arrays,
checks checkpoint tensors against the frozen source checkpoint, and closes the
matrix/provenance/receipt/log contracts directly from artifact bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED = {
    "best.pt", "last.pt", "manifest.json", "metrics.json", "normalizer.npz",
    "predictions.npz", "replay_args.json", "train_log.json", "code_hashes_start.json",
}
CODE_FILES = {
    "peft_experiment.py", "peft.py", "debug_experiment.py", "calibration.py",
    "sparse_state_tuning.py", "mamba3_research_adapters.py", "data.py",
    "mamba3_official.py",
}
RESEARCH = {"state_offset", "memba_causal"}


class AuditError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def json_normalize(value: Any) -> Any:
    """Apply the same key coercion that writing and reading JSON applies."""
    return json.loads(json.dumps(value))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    return sha_array(value.detach().cpu().numpy())


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def float_close(left: float, right: float, atol: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= atol


def r2_float64(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.square(truth - prediction).sum(axis=0).sum()
    centered = np.square(truth - truth.mean(axis=0)).sum(axis=0).sum()
    return float(1.0 - residual / max(1e-12, float(centered)))


def base_name(name: str) -> str:
    """Map wrapper/parametrization storage back to its source checkpoint key."""
    name = name.replace(".base.", ".")
    name = re.sub(r"\.parametrizations\.(B_bias|C_bias)\.original$", r".\1", name)
    return name


def expected_grid(matrix: dict[str, Any]) -> set[tuple[str, str, int]]:
    return {
        (task, method, seed)
        for method in matrix["methods"]
        for seed in ([0] if method == "none" else matrix["seeds"])
        for task in ("m1", "m2")
    }


def command_options(command: list[str]) -> dict[str, str]:
    require(len(command) >= 3 and command[1:3] == ["-m", "ssm_decode.peft_experiment"],
            f"unexpected launcher prefix: {command[:3]}")
    tail = command[3:]
    require(len(tail) % 2 == 0, "launcher command has a dangling option")
    result: dict[str, str] = {}
    for key, value in zip(tail[0::2], tail[1::2]):
        require(key.startswith("--") and key not in result, f"invalid/duplicate launcher option: {key}")
        result[key[2:].replace("-", "_")] = value
    return result


def audit_launcher(root: Path, matrix: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    copied = root / "matrix.json"
    launch_path = root / "launch_manifest.json"
    snapshot_dir = root / "code_snapshot"
    require(copied.exists() and launch_path.exists() and snapshot_dir.is_dir(),
            "missing matrix, launch manifest, or code snapshot")
    require(read_json(copied) == matrix, "copied matrix content differs from parsed matrix")
    launch = read_json(launch_path)
    require(launch["matrix_sha256"] == sha_file(copied), "launch matrix digest differs")
    require(sha_file(snapshot_dir / "run_peft_two_gpu.py") == launch["launcher_sha256"],
            "launcher snapshot digest differs")

    grid = expected_grid(matrix)
    seen: set[tuple[str, str, int]] = set()
    for job in launch["jobs"]:
        options = command_options(job["command"])
        identity = (job["task"], options["method"], int(options["seed"]))
        require(identity in grid and identity not in seen, f"unexpected/duplicate launch job: {identity}")
        seen.add(identity)
        task, method, seed = identity
        expected_output = str(root / task / f"{method}_seed{seed}")
        require(job["output"] == expected_output, f"job output mismatch: {identity}")
        require(options["output"] == expected_output, f"command output mismatch: {identity}")
        require(job["physical_gpu"] == {"m1": "0", "m2": "1"}[task],
                f"physical GPU mismatch: {identity}")
        expected = {
            "task": task, "device": "cuda:0", "pretrained": matrix["pretrained_by_task"][task],
            "method": method, "steps": str(matrix["steps"]), "context": str(matrix["context"]),
            "batch_size": str(matrix["batch_size"]), "rank": str(matrix["rank"]),
            "alpha": str(matrix["alpha"]), "seed": str(seed), "policy": matrix["policy"],
            "output": expected_output,
        }
        require(options == expected, f"launcher options differ for {identity}: {options}")
    require(seen == grid, f"launcher matrix is incomplete: missing={sorted(grid-seen)}")

    snapshot_manifest = read_json(snapshot_dir / "snapshot_manifest.json")
    frozen_hashes = snapshot_manifest["verified_against_fit_start_hashes"]
    require(set(frozen_hashes) == CODE_FILES, "snapshot code-file set differs")
    for name, expected in frozen_hashes.items():
        require(sha_file(snapshot_dir / name) == expected, f"code snapshot digest differs: {name}")
    return launch, frozen_hashes


def source_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict),
            f"invalid source checkpoint: {path}")
    return payload, payload["state_dict"]


def expected_method_paths(method: str, receipt: dict[str, Any]) -> set[str]:
    paths = set(receipt["trainable_paths"])
    if method == "none":
        require(not paths, "none has trainable paths")
    elif method == "io":
        require(paths == {
            "in_proj.weight", "in_proj.bias", "final_norm.weight", "final_norm.bias",
            "out_proj.weight", "out_proj.bias",
        }, f"io trainable paths differ: {sorted(paths)}")
    elif method in {"lora", "bc_lora", "bc_dt"}:
        lora_modules = {name.removesuffix(":BC") for name in receipt["lora_paths"]}
        expected = {f"{name}.{factor}" for name in lora_modules for factor in ("A", "B")}
        if method == "bc_dt":
            expected.update(receipt["groups"]["dt_bias"])
        require(paths == expected, f"{method} trainable path set differs")
        if method in {"bc_lora", "bc_dt"}:
            require(receipt.get("custom_bc_lora_not_sdlora") is True,
                    f"{method} is not explicitly labeled custom")
    elif method == "sparse_sdt_m3":
        require(receipt.get("custom_not_original_sdlora") is True,
                "sparse port is not explicitly labeled custom")
        allowed = (".parametrizations.B_bias.0.values", ".parametrizations.C_bias.0.values", ".A", ".B")
        require(paths and all(name.endswith(allowed) for name in paths),
                "sparse trainable paths escape compact adapters")
    elif method == "state_offset":
        require(receipt.get("upstream_reproduction") is False and receipt.get("functional") is True,
                "state_offset research labeling differs")
        require(paths and all(name in {"in_proj.A", "in_proj.B", "out_proj.A", "out_proj.B"}
                              or name.endswith((".research_adapter.U", ".research_adapter.V"))
                              for name in paths), "state_offset trainable paths differ")
    elif method == "memba_causal":
        require(receipt.get("upstream_reproduction") is False and receipt.get("functional") is True,
                "memba research labeling differs")
        require(paths and all(name in {"in_proj.A", "in_proj.B", "out_proj.A", "out_proj.B"}
                              or ".research_adapter.down." in name or ".research_adapter.up." in name
                              for name in paths), "memba trainable paths differ")
    elif method == "full":
        require(paths, "full has no trainable paths")
    else:
        raise AuditError(f"unknown method: {method}")
    return paths


def audit_optimizer(
    method: str,
    args: dict[str, Any],
    manifest: dict[str, Any],
    receipt_paths: set[str],
    tensor_sizes: dict[str, int],
) -> set[str]:
    stages = manifest["optimizer_stages"]
    if method == "none":
        require(stages == [], "none has optimizer stages")
        return set()
    expected_steps = [0, 200, 400] if method == "full" else [0]
    require([stage["step"] for stage in stages] == expected_steps,
            f"optimizer stage steps differ: {[stage['step'] for stage in stages]}")
    prior: set[str] = set()
    for stage in stages:
        flattened = [name for group in stage["groups"] for name in group["names"]]
        require(len(flattened) == len(set(flattened)), f"duplicate parameter in optimizer stage {stage['step']}")
        current = set(flattened)
        require(prior.issubset(current), f"optimizer lost parameters at stage {stage['step']}")
        prior = current
        for group in stage["groups"]:
            require(group["count"] == sum(tensor_sizes[name] for name in group["names"]),
                    f"optimizer count differs at stage {stage['step']}")
            require(finite_number(group["lr"]) and group["lr"] > 0, "invalid optimizer LR")
            require(finite_number(group["weight_decay"]) and group["weight_decay"] >= 0,
                    "invalid optimizer weight decay")
    require(prior == receipt_paths, "final optimizer authorization differs from receipt")

    if method == "bc_dt":
        require(len(stages[0]["groups"]) == 2, "bc_dt must have regular and dt groups")
        dt_names = set(manifest["receipt"]["groups"]["dt_bias"])
        groups = stages[0]["groups"]
        dt_group = next((group for group in groups if set(group["names"]) == dt_names), None)
        require(dt_group is not None, "bc_dt optimizer lacks exact dt group")
        require(float_close(dt_group["lr"], args["lr"] * 0.1, 1e-15)
                and dt_group["weight_decay"] == 0.0, "bc_dt optimizer policy differs")
        require(manifest["dt_bias_constraints"] == {
            "initial_relative_bounds": [-1.0, 1.0], "lr_multiplier": 0.1, "weight_decay": 0.0,
        }, "bc_dt constraint receipt differs")
    else:
        require(manifest["dt_bias_constraints"] is None, "non-bc_dt has dt constraint receipt")
    return prior


def audit_probe(probe: dict[str, Any], label: str) -> None:
    require(set(probe) == {"0", "1", "2", "3"}, f"{label}: probe layers differ")
    for layer, record in probe.items():
        require(all(finite_number(value) for value in record.values()), f"{label}: nonfinite DT probe layer {layer}")
        require(record["dt_min"] > 0 and record["dt_max"] > 0 and record["dt_mean"] > 0,
                f"{label}: nonpositive DT layer {layer}")
        require(record["adt_min"] < 0 and record["adt_max"] < 0,
                f"{label}: nonnegative ADT layer {layer}")


def audit_log(
    method: str,
    args: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    log: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    if method == "none":
        require(log == [], f"{label}: none training log is not empty")
        require(metrics["best_step"] == 0, f"{label}: none best step differs")
        audit_probe(manifest["selected_dt_probe"], f"{label}/selected")
        return {"min_grad_norm": None, "validation_points": 0}

    require(len(log) == args["steps"] + 1, f"{label}: training log length differs")
    require([entry["step"] for entry in log] == list(range(args["steps"] + 1)),
            f"{label}: training steps are not contiguous")
    gradient_norms: list[float] = []
    values: list[tuple[int, float]] = []
    expected_val_steps = {0, *range(args["val_interval"], args["steps"] + 1, args["val_interval"]), args["steps"]}
    actual_val_steps: set[int] = set()
    for entry in log:
        step = entry["step"]
        if step:
            require(finite_number(entry.get("loss")), f"{label}: nonfinite/missing loss at {step}")
            require(finite_number(entry.get("grad_norm")) and entry["grad_norm"] > 0,
                    f"{label}: nonpositive/nonfinite grad norm at {step}")
            gradient_norms.append(float(entry["grad_norm"]))
        if "prefix_val_r2" in entry:
            require(finite_number(entry["prefix_val_r2"]), f"{label}: nonfinite validation at {step}")
            actual_val_steps.add(step)
            values.append((step, float(entry["prefix_val_r2"])))
            audit_probe(entry["dt_probe"], f"{label}/step{step}")
    require(actual_val_steps == expected_val_steps, f"{label}: validation schedule differs")
    argmax = max(values, key=lambda pair: pair[1])
    require(metrics["best_step"] == argmax[0], f"{label}: best step is not first validation argmax")
    require(float_close(metrics["best_prefix_val_r2"], argmax[1], 1e-12),
            f"{label}: best validation score differs")
    audit_probe(manifest["selected_dt_probe"], f"{label}/selected")
    return {"min_grad_norm": min(gradient_norms), "validation_points": len(values)}


def audit_sparse(
    fit: Path,
    manifest: dict[str, Any],
    best: dict[str, Any],
    last: dict[str, Any],
) -> dict[str, Any]:
    sparse_path = fit / "sparse_selection.json"
    require(sparse_path.exists(), "sparse fit lacks selection artifact")
    sparse = read_json(sparse_path)
    receipt = sparse["selection_receipt"]
    require(receipt == manifest["sparse_selection"], "sparse manifest receipt differs")
    normalize = lambda value: {str(key): [int(item) for item in items] for key, items in value.items()}
    selections = normalize(receipt["selections"])
    require(selections == normalize(best["peft_replay"]["sparse_selections"]),
            "sparse best replay selections differ")
    require(selections == normalize(last["peft_replay"]["sparse_selections"]),
            "sparse last replay selections differ")
    require(set(selections) == {"0", "1", "2", "3"}, "sparse selection layers differ")
    details = {str(key): value for key, value in receipt["details"].items()}
    for layer, selected in selections.items():
        require(len(selected) == 8 and selected == sorted(set(selected))
                and min(selected) >= 0 and max(selected) < 32, f"sparse states invalid for layer {layer}")
        detail = details[layer]
        delta_b = np.asarray(detail["dense_delta_B"], dtype=np.float64)
        delta_c = np.asarray(detail["dense_delta_C"], dtype=np.float64)
        axes_b = tuple(range(delta_b.ndim - 1))
        axes_c = tuple(range(delta_c.ndim - 1))
        scores = np.square(delta_b).sum(axis=axes_b) + np.square(delta_c).sum(axis=axes_c)
        # Details were serialized from float32 reductions; recomputing after
        # decimal JSON round-trip can differ by roughly one float32 ULP.
        require(np.allclose(scores, np.asarray(detail["scores"]), atol=2e-10, rtol=2e-6),
                f"sparse scores differ for layer {layer}")
        chosen = np.sort(np.argsort(scores)[-8:]).tolist()
        require(chosen == selected == detail["selected"], f"sparse top-k differs for layer {layer}")
        mask = np.zeros(32, dtype=bool)
        mask[selected] = True
        require(np.array_equal(mask, np.asarray(detail["mask"], dtype=bool)),
                f"sparse mask differs for layer {layer}")
        for payload, name in ((best, "best"), (last, "last")):
            for bias in ("B_bias", "C_bias"):
                key = f"blocks.{layer}.ssm.parametrizations.{bias}.0.states"
                require(np.array_equal(payload["state_dict"][key].cpu().numpy(), np.asarray(selected)),
                        f"sparse {name} states differ: {key}")
    warm = sparse["warmup_log"]
    require(len(warm) == 100 and [entry["step"] for entry in warm] == list(range(1, 101)),
            "sparse warmup schedule differs")
    require(all(finite_number(entry["loss"]) for entry in warm), "sparse warmup contains nonfinite loss")
    require(finite_number(sparse["warmup_seconds"]) and sparse["warmup_seconds"] > 0,
            "sparse warmup runtime invalid")
    return {"selections": selections, "warmup_min_loss": min(entry["loss"] for entry in warm)}


def ridge_recompute(data: Any, normalizer: Any, scope: str) -> tuple[np.ndarray, float]:
    support_x32 = np.asarray(data["support_prediction"], dtype=np.float32)
    support_y32 = np.asarray(data["support_truth_normalized"], dtype=np.float32)
    # The experiment subtracts in float32, then promotes to float64 for the solve.
    x = support_x32.astype(np.float64)
    delta = (support_y32 - support_x32).astype(np.float64)
    xm, dm = x.mean(axis=0), delta.mean(axis=0)
    xc, dc = x - xm, delta - dm
    weight = np.linalg.solve(xc.T @ xc + np.eye(x.shape[1], dtype=np.float64), xc.T @ dc)
    intercept = dm - xm @ weight
    zero = np.asarray(data[f"zero_{scope}"], dtype=np.float32)
    mean = np.asarray(normalizer["y_mean"], dtype=np.float32)
    scale = np.asarray(normalizer["y_std"], dtype=np.float32)
    raw = ((zero - mean) / scale).astype(np.float32)
    correction = (raw.astype(np.float64) @ weight + intercept).astype(np.float32)
    expected = ((raw + correction) * scale + mean).astype(np.float32)
    actual = np.asarray(data[f"ridge_{scope}"], dtype=np.float32)
    return expected, float(np.max(np.abs(expected - actual), initial=0.0))


def audit_predictions(
    fit: Path,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    historic: Any,
    cohort_reference: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    data = np.load(fit / "predictions.npz", allow_pickle=False)
    normalizer = np.load(fit / "normalizer.npz", allow_pickle=False)
    require(set(data.files) == {
        "legacy_indices", "allvalid_indices", "truth_legacy", "truth_allvalid",
        "support_indices", "support_truth_normalized", "support_prediction",
        "zero_legacy", "ridge_legacy", "zero_allvalid", "ridge_allvalid",
    }, "prediction artifact schema differs")
    require(set(normalizer.files) == {"x_mean", "x_std", "y_mean", "y_std"},
            "normalizer schema differs")
    for key in data.files:
        require(np.isfinite(data[key]).all(), f"prediction array has nonfinite values: {key}")
    for scope in ("legacy", "allvalid", "support"):
        indices = data[f"{scope}_indices"]
        require(indices.dtype == np.int64 and indices.ndim == 1,
                f"{scope} indices dtype/shape differs")
        require(np.array_equal(indices, np.unique(indices)), f"{scope} indices are not sorted unique")
    require(np.intersect1d(data["support_indices"], data["allvalid_indices"]).size == 0,
            "support overlaps all-valid query")
    require(np.setdiff1d(data["legacy_indices"], data["allvalid_indices"]).size == 0,
            "legacy query is not a subset of all-valid query")

    historic_map = {
        "legacy_indices": "legacy_query_indices",
        "allvalid_indices": "all_valid_query_indices",
        "truth_legacy": "legacy_truth_physical",
        "truth_allvalid": "all_valid_truth_physical",
    }
    for current, old in historic_map.items():
        require(np.array_equal(data[current], historic[old]), f"historical cohort/truth differs: {current}")

    signature = {
        "legacy_indices": sha_array(data["legacy_indices"]),
        "allvalid_indices": sha_array(data["allvalid_indices"]),
        "support_indices": sha_array(data["support_indices"]),
        "truth_legacy": sha_array(data["truth_legacy"]),
        "truth_allvalid": sha_array(data["truth_allvalid"]),
        "support_truth_normalized": sha_array(data["support_truth_normalized"]),
    }
    if cohort_reference is not None:
        require(signature == cohort_reference, "cohort/truth signature differs within task")
    require(manifest["cohort_index_hashes"] == {
        "support_indices": signature["support_indices"],
        "legacy_indices": signature["legacy_indices"],
        "all_valid_indices": signature["allvalid_indices"],
    }, "manifest cohort index hashes differ")

    max_r2_error = 0.0
    max_ridge_error = 0.0
    scores: dict[str, float] = {}
    for scope in ("legacy", "allvalid"):
        truth = data[f"truth_{scope}"]
        require(truth.ndim == 2, f"{scope} truth shape differs")
        for adaptation in ("zero", "ridge"):
            prediction = data[f"{adaptation}_{scope}"]
            require(prediction.shape == truth.shape, f"{adaptation}/{scope} prediction shape differs")
            value = r2_float64(truth, prediction)
            reported = metrics["final"][adaptation]["all_valid" if scope == "allvalid" else "legacy"]
            reported_value = float(reported["r2_variance_weighted"])
            error = abs(value - reported_value)
            max_r2_error = max(max_r2_error, error)
            require(error <= 1e-5, f"R2 mismatch {adaptation}/{scope}: {value} vs {reported_value}")
            require(reported["n_query_bins"] == len(truth), f"query count differs {adaptation}/{scope}")
            scores[f"{adaptation}_{scope}_r2"] = value
        expected, ridge_error = ridge_recompute(data, normalizer, scope)
        del expected
        max_ridge_error = max(max_ridge_error, ridge_error)
        require(ridge_error <= 1e-6, f"independent ridge prediction mismatch {scope}: {ridge_error}")

    for adaptation in ("zero", "ridge"):
        record = metrics["final"][adaptation]
        require(record["n_support_bins"] == len(data["support_indices"]), "support count differs")
        require(record["n_query_bins"] == len(data["legacy_indices"]), "legacy count differs")
        require(record["all_valid_query_trial_n_bins"] == len(data["allvalid_indices"]),
                "all-valid count differs")
        require(record["r2_variance_weighted"] == record["legacy"]["r2_variance_weighted"],
                "top-level legacy R2 differs")
        require(record["all_valid_query_trial_r2_variance_weighted"]
                == record["all_valid"]["r2_variance_weighted"], "top-level all-valid R2 differs")
    return {
        **scores,
        "max_r2_abs_error": max_r2_error,
        "max_ridge_prediction_abs_error": max_ridge_error,
        "n_support": len(data["support_indices"]),
        "n_legacy": len(data["legacy_indices"]),
        "n_allvalid": len(data["allvalid_indices"]),
    }, signature


def audit_fit(
    root: Path,
    fit: Path,
    identity: tuple[str, str, int],
    matrix: dict[str, Any],
    launch_job: dict[str, Any],
    frozen_hashes: dict[str, str],
    source_payload: dict[str, Any],
    source_state: dict[str, torch.Tensor],
    historic: Any,
    cohort_reference: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    task, method, seed = identity
    label = f"{task}/{method}_seed{seed}"
    existing = {path.name for path in fit.iterdir() if path.is_file()}
    require(REQUIRED.issubset(existing), f"{label}: missing artifacts {sorted(REQUIRED-existing)}")
    metrics = read_json(fit / "metrics.json")
    manifest = read_json(fit / "manifest.json")
    args = read_json(fit / "replay_args.json")
    start_hashes = read_json(fit / "code_hashes_start.json")
    log = read_json(fit / "train_log.json")
    best = torch.load(fit / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(fit / "last.pt", map_location="cpu", weights_only=False)

    require(metrics["status"] == "completed", f"{label}: status is not completed")
    require((args["task"], args["method"], int(args["seed"])) == identity, f"{label}: args identity differs")
    require((manifest["datamanifest"]["task"], manifest["method"], int(manifest["seed"])) == identity,
            f"{label}: manifest identity differs")
    require((metrics["method"], int(metrics["seed"])) == (method, seed), f"{label}: metrics identity differs")
    require(Path(args["output"]) == fit, f"{label}: output path differs")
    require(args["pretrained"] == matrix["pretrained_by_task"][task], f"{label}: source path differs")
    for key in ("steps", "context", "batch_size", "rank", "alpha", "policy"):
        require(args[key] == matrix[key], f"{label}: matrix argument differs: {key}")
    require(args["device"] == "cuda:0" and args["eval_batch_size"] == 256
            and args["val_interval"] == 50 and args["lr"] == 1e-4
            and args["weight_decay"] == 1e-4, f"{label}: fixed runner arguments differ")
    require(launch_job["output"] == str(fit), f"{label}: launch output differs")

    require(start_hashes == frozen_hashes == manifest["code_hashes"], f"{label}: code hashes differ")
    require(set(start_hashes) == CODE_FILES, f"{label}: code hash file set differs")
    require(manifest["cuda_visible_devices"] == {"m1": "0", "m2": "1"}[task],
            f"{label}: visible GPU differs")
    official = manifest["official"]
    require(official["expected_official_commit"] == official["actual_official_commit"]
            == "e9594ce1c732d97440f0332fdc43170a2294dbfa", f"{label}: official commit differs")
    require(official["triton"]["version"] == "3.5.0", f"{label}: Triton version differs")
    require(official["pythonpath"]
            == "/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM",
            f"{label}: isolated PYTHONPATH differs")

    source_path = Path(args["pretrained"])
    source_normalizer = source_path.parent / "normalizer.npz"
    require(sha_file(source_path) == manifest["pretrained_sha256"], f"{label}: source digest differs")
    require(Path(manifest["statsfile"]) == source_normalizer, f"{label}: source normalizer path differs")
    require(sha_file(source_normalizer) == manifest["statsfile_sha256"],
            f"{label}: source normalizer digest differs")
    require(sha_file(fit / "normalizer.npz") == manifest["statsfile_sha256"],
            f"{label}: copied normalizer bytes differ")
    copied_normalizer = np.load(fit / "normalizer.npz", allow_pickle=False)
    source_normalizer_npz = np.load(source_normalizer, allow_pickle=False)
    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        require(np.array_equal(copied_normalizer[key], source_normalizer_npz[key]),
                f"{label}: copied normalizer array differs: {key}")
        require(np.array_equal(copied_normalizer[key], best["normalizer"][key])
                and np.array_equal(copied_normalizer[key], last["normalizer"][key]),
                f"{label}: checkpoint normalizer differs: {key}")

    require(best["args"] == args and last["args"] == args, f"{label}: checkpoint args differ")
    # Sparse selection maps use integer layer keys inside torch checkpoints and
    # string keys in JSON.  Compare their JSON meanings rather than Python key
    # types, while still requiring best and last torch receipts to be identical.
    require(best["receipt"] == last["receipt"], f"{label}: best/last receipt differs")
    require(json_normalize(best["receipt"]) == manifest["receipt"],
            f"{label}: checkpoint/manifest receipt content differs")
    require(best["peft_replay"] == last["peft_replay"], f"{label}: replay metadata differs")
    replay = best["peft_replay"]
    require(replay["method"] == method and replay["rank"] == matrix["rank"]
            and replay["alpha"] == float(matrix["alpha"]), f"{label}: replay method arguments differ")
    require(replay["cfg"] == source_payload["args"], f"{label}: replay source config differs")
    require(best["best_step"] == metrics["best_step"], f"{label}: best checkpoint step differs")
    require(last["best_step"] == (0 if method == "none" else args["steps"]),
            f"{label}: last checkpoint step differs")

    initial_hashes = manifest["initial_parameter_hashes"]
    final_hashes = manifest["final_parameter_hashes"]
    require(set(initial_hashes) == set(final_hashes), f"{label}: parameter hash key sets differ")
    require(set(initial_hashes).issubset(best["state_dict"]), f"{label}: best lacks hashed parameters")
    require(set(initial_hashes).issubset(last["state_dict"]), f"{label}: last lacks hashed parameters")
    tensor_sizes = {name: best["state_dict"][name].numel() for name in initial_hashes}
    for name, expected in final_hashes.items():
        require(sha_tensor(best["state_dict"][name]) == expected, f"{label}: final tensor hash differs: {name}")
    changed = {name for name in initial_hashes if initial_hashes[name] != final_hashes[name]}
    require(changed == set(manifest["changed_parameter_names"]), f"{label}: changed-name receipt differs")

    receipt = manifest["receipt"]
    if method == "sparse_sdt_m3":
        require(receipt.get("stage") == "sparse" and receipt["rank"] == matrix["rank"],
                f"{label}: sparse receipt stage/rank differs")
    else:
        require(receipt["method"] == method and receipt["rank"] == matrix["rank"],
                f"{label}: receipt method/rank differs")
    receipt_paths = expected_method_paths(method, receipt)
    require(receipt_paths == set(manifest["trainable_parameters"]),
            f"{label}: trainable manifest differs from receipt")
    require(receipt["trainable_count"] == sum(tensor_sizes[name] for name in receipt_paths),
            f"{label}: receipt parameter count differs")
    authorized = audit_optimizer(method, args, manifest, receipt_paths, tensor_sizes)
    require(changed.issubset(authorized), f"{label}: unauthorized changed parameters: {sorted(changed-authorized)}")
    last_changed = {
        name for name in initial_hashes
        if sha_tensor(last["state_dict"][name]) != initial_hashes[name]
    }
    require(last_changed == authorized,
            f"{label}: last-checkpoint updates differ from optimizer authorization: "
            f"missing={sorted(authorized-last_changed)}, extra={sorted(last_changed-authorized)}")
    zero_initialized = {
        name for name in receipt_paths
        if initial_hashes[name] == sha_tensor(torch.zeros_like(last["state_dict"][name]))
    }
    require(all(torch.count_nonzero(last["state_dict"][name]).item() > 0 for name in zero_initialized),
            f"{label}: a zero-initialized trainable tensor remained inert")

    source_count = sum(value.numel() for value in source_state.values())
    require(metrics["base_parameters"] == source_count, f"{label}: base parameter count differs")
    require(metrics["total_parameters"] == sum(tensor_sizes.values()), f"{label}: total parameter count differs")
    require(metrics["trainable_params"] == receipt["trainable_count"],
            f"{label}: trainable count differs")
    initial_optimizer_count = 0 if method == "none" else sum(
        group["count"] for group in manifest["optimizer_stages"][0]["groups"]
    )
    require(metrics["initial_trainable_params"] == initial_optimizer_count,
            f"{label}: initial trainable count differs")

    # Every port starts from the exact source tensors, including parameters that
    # are later authorized to change.  Wrapper names are mapped back above.
    initial_source_names: set[str] = set()
    for name in initial_hashes:
        mapped = base_name(name)
        if mapped not in source_state:
            continue
        require(mapped not in initial_source_names, f"{label}: duplicate mapped source parameter: {mapped}")
        initial_source_names.add(mapped)
        require(initial_hashes[name] == sha_tensor(source_state[mapped]),
                f"{label}: initial tensor hash differs from source: {name}")
    require(initial_source_names == set(source_state),
            f"{label}: initial checkpoint does not cover source state exactly")

    frozen_exact = 0
    for name in initial_hashes:
        mapped = base_name(name)
        if mapped not in source_state or name in authorized:
            continue
        expected = sha_tensor(source_state[mapped])
        require(initial_hashes[name] == expected == final_hashes[name],
                f"{label}: frozen hash differs from source: {name}")
        require(torch.equal(best["state_dict"][name].cpu(), source_state[mapped].cpu()),
                f"{label}: frozen best tensor differs from source: {name}")
        require(torch.equal(last["state_dict"][name].cpu(), source_state[mapped].cpu()),
                f"{label}: frozen last tensor differs from source: {name}")
        frozen_exact += 1
    if method == "none":
        require(frozen_exact == len(source_state) == len(initial_hashes),
                f"{label}: none is not exact source state")
    elif method != "full":
        require(frozen_exact > 0, f"{label}: no independently checked frozen tensors")

    dt_update_report = None
    if method == "bc_dt":
        dt_update_report = {}
        for name in receipt["groups"]["dt_bias"]:
            source = source_state[name].cpu()
            require(name in changed, f"{label}: optimized dt_bias has no best-checkpoint update: {name}")
            values: dict[str, Any] = {}
            for payload_name, payload in (("best", best), ("last", last)):
                raw_delta = payload["state_dict"][name].cpu() - source
                delta = raw_delta.abs().max().item()
                nonzero = int(torch.count_nonzero(raw_delta).item())
                require(nonzero == raw_delta.numel(),
                        f"{label}: {payload_name} dt_bias did not update every head: {name}")
                require(0.0 < delta <= 1.0 + 1e-6,
                        f"{label}: {payload_name} dt update/clamp invalid: {name} {delta}")
                values[payload_name] = {
                    "max_abs_delta": delta,
                    "l2_delta": float(raw_delta.float().norm()),
                    "nonzero_elements": nonzero,
                    "elements": raw_delta.numel(),
                }
            dt_update_report[name] = values

    log_report = audit_log(method, args, metrics, manifest, log, label)
    prediction_report, signature = audit_predictions(
        fit, metrics, manifest, historic, cohort_reference
    )
    sparse_report = audit_sparse(fit, manifest, best, last) if method == "sparse_sdt_m3" else None
    require((fit / "sparse_selection.json").exists() == (method == "sparse_sdt_m3"),
            f"{label}: unexpected/missing sparse selection file")
    require(manifest["sparse_selection"] is not None if method == "sparse_sdt_m3"
            else manifest["sparse_selection"] is None, f"{label}: sparse manifest marker differs")

    require(manifest["policy"] == matrix["policy"], f"{label}: policy differs")
    require(manifest["prefix_train_count"] == 26 and len(manifest["train_bounds"]) == 26,
            f"{label}: prefix train split differs")
    require(manifest["prefix_val_count"] == 7 and len(manifest["val_bounds"]) == 7,
            f"{label}: prefix validation split differs")
    require(manifest["query_cutoff_trials"] == 33
            and manifest["support_bounds"] == manifest["train_bounds"] + manifest["val_bounds"],
            f"{label}: support/query split receipt differs")
    require(manifest["datamanifest"]["held_out_accessed"] is False,
            f"{label}: held-out access marker differs")
    require(manifest["datamanifest"]["authorized_splits"] == ["held_in", "minival"],
            f"{label}: authorized split receipt differs")
    require(finite_number(metrics["fit_seconds"]) and metrics["fit_seconds"] >= 0,
            f"{label}: fit runtime invalid")
    require(finite_number(metrics["peak_cuda_allocated_mb"]) and metrics["peak_cuda_allocated_mb"] > 0,
            f"{label}: peak GPU allocation invalid")
    log_path = fit.parent / f"{fit.name}.log"
    require(log_path.exists(), f"{label}: launcher log missing")
    log_text = log_path.read_text(errors="replace")
    require("Traceback (most recent call last)" not in log_text, f"{label}: traceback in launcher log")
    require(log_text.rstrip().endswith(str(fit)), f"{label}: launcher log lacks completion path")

    nonzero_changed = sum(
        int(torch.count_nonzero(best["state_dict"][name]).item() > 0)
        for name in changed
    )
    return {
        "fit": str(fit), "task": task, "method": method, "seed": seed,
        "best_step": metrics["best_step"], "best_prefix_val_r2": metrics["best_prefix_val_r2"],
        "trainable_params": metrics["trainable_params"], "changed_parameter_count": len(changed),
        "last_changed_parameter_count": len(last_changed),
        "zero_initialized_updated_tensor_count": len(zero_initialized),
        "nonzero_changed_tensor_count": nonzero_changed,
        "initial_source_exact_tensor_count": len(initial_source_names),
        "frozen_source_exact_tensor_count": frozen_exact,
        "fit_seconds": metrics["fit_seconds"], "peak_cuda_allocated_mb": metrics["peak_cuda_allocated_mb"],
        **log_report, **prediction_report, "sparse": sparse_report,
        "dt_bias_updates": dt_update_report,
    }, signature


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["method"]), []).append(row)
    result: list[dict[str, Any]] = []
    numeric = (
        "zero_legacy_r2", "ridge_legacy_r2", "zero_allvalid_r2", "ridge_allvalid_r2",
        "trainable_params", "fit_seconds", "peak_cuda_allocated_mb",
    )
    for (task, method), values in sorted(groups.items()):
        record: dict[str, Any] = {"task": task, "method": method, "n": len(values)}
        for key in numeric:
            array = np.asarray([row[key] for row in values], dtype=np.float64)
            record[f"{key}_mean"] = float(array.mean())
            record[f"{key}_std_population"] = float(array.std(ddof=0))
        result.append(record)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    options = parser.parse_args(argv)
    root = options.root
    matrix_path = options.matrix or (root / "matrix.json")
    matrix = read_json(matrix_path)
    launch, frozen_hashes = audit_launcher(root, matrix)
    grid = expected_grid(matrix)
    jobs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for job in launch["jobs"]:
        parsed = command_options(job["command"])
        jobs[(job["task"], parsed["method"], int(parsed["seed"]))] = job

    sources: dict[str, tuple[dict[str, Any], dict[str, torch.Tensor]]] = {}
    historic: dict[str, Any] = {}
    for task in ("m1", "m2"):
        source_path = Path(matrix["pretrained_by_task"][task])
        sources[task] = source_checkpoint(source_path)
        historic[task] = np.load(source_path.parent / "target_query_predictions.npz", allow_pickle=False)

    completed: set[tuple[str, str, int]] = set()
    partial: set[tuple[str, str, int]] = set()
    absent: set[tuple[str, str, int]] = set()
    for identity in grid:
        task, method, seed = identity
        fit = root / task / f"{method}_seed{seed}"
        if (fit / "metrics.json").exists():
            completed.add(identity)
        elif fit.exists():
            partial.add(identity)
        else:
            absent.add(identity)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    cohort_reference: dict[str, dict[str, str]] = {}
    for identity in sorted(completed):
        task, method, seed = identity
        fit = root / task / f"{method}_seed{seed}"
        try:
            row, signature = audit_fit(
                root, fit, identity, matrix, jobs[identity], frozen_hashes,
                *sources[task], historic[task], cohort_reference.get(task),
            )
            cohort_reference.setdefault(task, signature)
            rows.append(row)
        except Exception as error:  # retain every independent failure in the report
            failures.append({"fit": str(fit), "error": f"{type(error).__name__}: {error}"})

    if options.require_complete and (partial or absent):
        failures.append({
            "fit": str(root),
            "error": f"matrix incomplete: completed={len(completed)}/{len(grid)}, "
                     f"partial={sorted(partial)}, absent={sorted(absent)}",
        })
    grouped = summarize(rows)
    status = "failed" if failures else ("verified_complete" if len(completed) == len(grid) else "verified_partial")
    report = {
        "status": status,
        "matrix_expected_fits": len(grid),
        "completed_artifacts_seen": len(completed),
        "verified_fits": len(rows),
        "partial_fits": [list(item) for item in sorted(partial)],
        "absent_fits": [list(item) for item in sorted(absent)],
        "failures": failures,
        "max_r2_abs_error": max((row["max_r2_abs_error"] for row in rows), default=None),
        "max_ridge_prediction_abs_error": max(
            (row["max_ridge_prediction_abs_error"] for row in rows), default=None
        ),
        "code_hashes": frozen_hashes,
        "rows": rows,
        "grouped": grouped,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "matrix_expected_fits", "completed_artifacts_seen", "verified_fits",
        "max_r2_abs_error", "max_ridge_prediction_abs_error", "failures",
    )}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
