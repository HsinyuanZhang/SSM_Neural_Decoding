#!/usr/bin/env python3
"""Offline, query-free verifier for frozen unit/session fit completions.

This verifier never imports a decoder, reads score archives, or calls the data
reader.  It reads only fit artifacts, the frozen contract, approved calibration
files for their byte digests, and CPU-mapped checkpoints.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[4]
HEX = re.compile(r"[0-9a-f]{64}\Z")
MAP_KEYS = ("input_weight", "input_bias", "output_weight", "output_bias")
FORBIDDEN_FILES = frozenset({"summary.json"})
TASK_DIMS = {"m1": (64, 16), "m2": (96, 2)}
SOURCE_SESSION_COUNTS = {"m1": 3, "m2": 6}
SOURCE_FILES = frozenset({"best.pt", "last.pt", "progress.json", "train_log.json", "effective_maps.npz", "fit_result.json"})
TARGET_FILES = frozenset({"best.pt", "last.pt", "train_log.json", "effective_maps.npz", "fit_result.json", "normalizer.npz", "folded.pt"})


class AuditError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def require_digest(value: Any, label: str) -> str:
    if not is_digest(value):
        fail(f"{label} is not a lowercase SHA-256 digest")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def relative_child(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        fail(f"{label} is not a relative path")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        fail(f"{label} escapes its task root")
    return path


def json_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        fail(f"{label} differs")


def tensor_hash(tensor: Any, label: str) -> str:
    if not isinstance(tensor, torch.Tensor):
        fail(f"{label} is not a tensor")
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def checkpoint_hashes(
    path: Path, expected: dict[str, Any], expected_metadata: dict[str, Any], expected_step: Any, label: str
) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exercised against actual checkpoints
        raise AuditError(f"cannot CPU-load {label}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        fail(f"{label} lacks checkpoint metadata")
    json_equal(payload["metadata"], expected_metadata, f"{label} metadata")
    require(payload.get("best_step") == expected_step, f"{label} selected step differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        fail(f"{label} lacks state_dict")
    actual = {name: tensor_hash(value, f"{label}:{name}") for name, value in state.items()}
    json_equal(actual, expected, f"{label} selected parameter hashes")
    return state


def checkpoint_metadata(path: Path, expected_metadata: dict[str, Any], expected_step: Any, label: str) -> None:
    """CPU-only provenance check for non-selected checkpoints."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exercised against actual checkpoints
        raise AuditError(f"cannot CPU-load {label}") from error
    require(isinstance(payload, dict) and isinstance(payload.get("metadata"), dict), f"{label} lacks checkpoint metadata")
    json_equal(payload["metadata"], expected_metadata, f"{label} metadata")
    if expected_step is not None:
        require(payload.get("best_step") == expected_step, f"{label} step differs")
    state = payload.get("state_dict")
    require(isinstance(state, dict) and state, f"{label} lacks state_dict")
    for name, tensor in state.items():
        tensor_hash(tensor, f"{label}:{name}")


def parse_mamba_constants() -> tuple[str, Path]:
    path = REPO / "ssm_decode" / "mamba3_official.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    commit = None
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "COMMIT" and isinstance(node.value, ast.Constant):
                commit = node.value.value
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        fail("cannot parse pinned Mamba-3 commit")
    return commit, REPO / ".tools" / "mamba_official"


def validate_frozen_runtime(value: Any) -> dict[str, Any]:
    """Validate provenance without requiring the audit host to be the GPU host."""
    require(isinstance(value, dict) and set(value) == {"torch", "cuda", "official_expected", "official_actual", "triton"},
            "frozen runtime schema differs")
    commit, official = parse_mamba_constants()
    try:
        actual = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError("cannot read pinned official Mamba-3 revision") from error
    triton_init = REPO / ".tools" / "mamba_deps" / "triton" / "__init__.py"
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", triton_init.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        fail("cannot parse isolated Triton version")
    require(isinstance(value["torch"], str) and value["torch"], "frozen Torch runtime is malformed")
    require(isinstance(value["cuda"], str) and value["cuda"], "frozen CUDA runtime is malformed")
    require(value["official_expected"] == commit and value["official_actual"] == actual and value["triton"] == match.group(1),
            "frozen official Mamba-3 or Triton runtime differs")
    return value


def check_no_query_files(task_root: Path) -> None:
    for path in task_root.rglob("*"):
        relative = path.relative_to(task_root)
        lowered = tuple(part.lower() for part in relative.parts)
        if any("query" in part or "score" in part for part in lowered) or relative.name.lower() in FORBIDDEN_FILES:
            fail(f"query or score artifact is present: {relative}")


def source_session_count(task: str) -> int:
    try:
        return SOURCE_SESSION_COUNTS[task]
    except KeyError as error:  # defensive: config task roster is checked by main
        raise AuditError(f"unknown task for source roster: {task}") from error


def task_dims(task: str) -> tuple[int, int]:
    try:
        return TASK_DIMS[task]
    except KeyError as error:  # defensive: config task roster is checked by main
        raise AuditError(f"unknown task for dimensions: {task}") from error


def expected_validation_ids(raw_trials: int) -> list[int]:
    expected = [4, 9] if raw_trials == 10 else [4, 9, 14, 19, 24, 29, 32] if raw_trials == 33 else None
    if expected is None:
        fail(f"unsupported raw support-trial budget: {raw_trials}")
    return expected


def require_file_roster(directory: Path, expected: frozenset[str], label: str) -> None:
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    require(actual == expected, f"{label} file roster differs")
    require(not any(path.is_dir() for path in directory.iterdir()), f"{label} contains an unexpected subdirectory")


def require_target_directory_name(directory: Path, method: Any, seed: Any, lr: Any, label: str) -> None:
    require(isinstance(method, str) and method in {"none", "ui"}, f"{label} method is malformed")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, f"{label} seed is malformed")
    require(isinstance(lr, (int, float)) and not isinstance(lr, bool) and math.isfinite(float(lr)),
            f"{label} learning rate is malformed")
    require(directory.name == f"{method}_seed{seed}_lr{float(lr):g}", f"{label} directory identity differs")


def require_finite_json(value: Any, label: str) -> None:
    """Reject non-finite numbers anywhere in a fit receipt or log."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str), f"{label} has a non-string key")
            require_finite_json(item, f"{label}.{key}")
        return
    fail(f"{label} contains an unsupported JSON value")


def require_exact_bounds(value: Any, expected: list[Any], label: str) -> None:
    require(isinstance(value, list), f"{label} is not a list")
    json_equal(value, expected, label)


def support_partition(contract: dict[str, Any], task: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[int], list[Any], list[Any]]:
    """Rebuild the permitted raw-trial split from the frozen calibration receipt."""
    receipt = contract["calibration_binding"]["receipt"]
    raw_trials = cfg["tasks"][task]["raw_support_trials"]
    expected_ids = expected_validation_ids(raw_trials)
    require(receipt.get("raw_nwb_trial_ids_first_n") == list(range(raw_trials)),
            f"{task} receipt raw support trial IDs differ")
    bounds = receipt.get("realtrialbounds")
    require(isinstance(bounds, list) and len(bounds) == raw_trials, f"{task} receipt real-trial bounds differ")
    for index, bound in enumerate(bounds):
        require(isinstance(bound, list) and len(bound) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) for item in bound)
                and bound[0] < bound[1], f"{task} receipt real-trial bound {index} is malformed")
    require(contract.get("raw_validation_trial_indices") == expected_ids, f"{task} contract raw validation split differs")
    selected = set(expected_ids)
    return receipt, expected_ids, [bound for index, bound in enumerate(bounds) if index not in selected], [bounds[index] for index in expected_ids]


def check_matrix(root: Path, cfg: dict[str, Any], config_path: Path, launcher_sha: str) -> None:
    matrix = read_json(root / "matrix.json", "frozen matrix")
    require_digest(matrix.get("config_sha256"), "matrix config SHA")
    require_digest(matrix.get("launcher_sha256"), "matrix launcher SHA")
    json_equal(matrix.get("config"), cfg, "matrix config")
    require(matrix["config_sha256"] == sha256(config_path), "matrix config bytes changed")
    require(matrix["launcher_sha256"] == launcher_sha, "matrix launcher SHA differs from frozen input")


def check_contract(
    task_root: Path, task: str, cfg: dict[str, Any], config_path: Path, launcher_sha: str, runtime: dict[str, Any]
) -> dict[str, Any]:
    contract = read_json(task_root / "contract.json", f"{task} contract")
    require(contract.get("schema") == "unit_session_contract_v1", f"{task} contract schema differs")
    require(contract.get("task") == task, f"{task} contract task differs")
    json_equal(contract.get("cfg"), cfg, f"{task} frozen config")
    require(contract.get("config_sha256") == sha256(config_path), f"{task} contract config SHA differs")
    json_equal(contract.get("runtime"), runtime, f"{task} runtime")
    code = contract.get("code_hashes")
    require(isinstance(code, dict) and code, f"{task} code hashes are missing")
    snapshots = task_root / "code_snapshot"
    actual_snapshot_names = {p.name for p in snapshots.iterdir() if p.is_file()} if snapshots.is_dir() else set()
    require(actual_snapshot_names == set(code), f"{task} code snapshot roster differs")
    for name, expected in code.items():
        require(isinstance(name, str) and "/" not in name and "\\" not in name, f"{task} code name is unsafe")
        require_digest(expected, f"{task} code hash {name}")
        source, snapshot = REPO / "ssm_decode" / name, snapshots / name
        require(source.is_file() and snapshot.is_file(), f"{task} code source or snapshot is missing: {name}")
        require(sha256(source) == expected and sha256(snapshot) == expected, f"{task} code changed: {name}")
    require(sha256(REPO / "scripts" / "run_unit_session_iteration.py") == launcher_sha, "launcher changed")

    plan = contract.get("plan")
    require(isinstance(plan, dict), f"{task} plan is missing")
    require(plan.get("task") == task and plan.get("held_out_accessed") is False, f"{task} plan scope differs")
    source_sessions = plan.get("source_held_in_sessions")
    target = plan.get("cross_session_local_dev")
    expected_source_sessions = source_session_count(task)
    require(isinstance(source_sessions, list) and len(source_sessions) == expected_source_sessions
            and len(set(source_sessions)) == expected_source_sessions
            and all(isinstance(session, str) and session for session in source_sessions),
            f"{task} source roster has the wrong session count")
    require(isinstance(target, dict) and isinstance(target.get("target_session"), str), f"{task} target session is missing")
    require(target["target_session"] not in source_sessions, f"{task} source and target overlap")
    held = plan.get("held_in")
    require(isinstance(held, list) and {row.get("session") for row in held if isinstance(row, dict)} == set(source_sessions) | {target["target_session"]},
            f"{task} held-in roster differs")
    held_by_session: dict[str, dict[str, Any]] = {}
    for row in held:
        require(isinstance(row, dict) and isinstance(row.get("session"), str), f"{task} held-in binding malformed")
        path = Path(row.get("path", ""))
        expected = require_digest(row.get("sha256"), f"{task} held-in data SHA")
        require(path.is_file() and sha256(path) == expected, f"{task} held-in data changed: {row['session']}")
        held_by_session[row["session"]] = row
    calibration = contract.get("calibration_binding")
    require(isinstance(calibration, dict) and isinstance(calibration.get("receipt"), dict), f"{task} calibration binding missing")
    calibration_path = Path(calibration.get("path", ""))
    receipt = calibration["receipt"]
    require(calibration_path.is_file(), f"{task} calibration file missing")
    require(receipt.get("rawfile_sha256") == sha256(calibration_path), f"{task} calibration data changed")
    require(calibration_path == Path(held_by_session[target["target_session"]]["path"]), f"{task} calibration target differs")
    require(receipt.get("calibration_trials") == cfg["tasks"][task]["raw_support_trials"], f"{task} calibration budget differs")
    require(receipt.get("query_labels_used") is False and receipt.get("publiccalibration_only") is True,
            f"{task} calibration is not query-free")
    reader = REPO.parent / "APST" / "src" / "apst" / "data" / "load.py"
    require(reader.is_file() and contract.get("reader_sha256") == sha256(reader), f"{task} official reader changed")
    require(contract.get("query_labels_used_for_selection") is False and contract.get("source_target_disjoint") is True,
            f"{task} contract query/disjoint flags differ")
    support_partition(contract, task, cfg)
    return contract


def check_artifact_tree(task_root: Path, completion: dict[str, Any], code_names: set[str]) -> int:
    artifacts = completion.get("artifacts")
    require(isinstance(artifacts, dict), "fit artifact manifest is missing")
    expected_artifacts: set[str] = set()
    for relative, expected in artifacts.items():
        path = relative_child(task_root, relative, "fit artifact path")
        require_digest(expected, f"fit artifact SHA {relative}")
        require(path.is_file() and sha256(path) == expected, f"fit artifact changed: {relative}")
        expected_artifacts.add(relative)
    actual = {str(path.relative_to(task_root)) for path in task_root.rglob("*") if path.is_file()}
    expected = expected_artifacts | {"fits_complete.json"} | {f"code_snapshot/{name}" for name in code_names}
    json_equal(actual, expected, "task tree and artifact manifest")
    return len(expected_artifacts)


def load_maps(path: Path, result: dict[str, Any], sessions: int, width: int, inputs: int, outputs: int, label: str) -> dict[str, np.ndarray]:
    require(path.is_file() and sha256(path) == result.get("effective_maps_sha256"), f"{label} effective-map digest differs")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(MAP_KEYS), f"{label} effective-map keys differ")
        maps = {name: np.asarray(archive[name]) for name in MAP_KEYS}
    expected_shapes = {
        "input_weight": (sessions, width, inputs), "input_bias": (sessions, width),
        "output_weight": (sessions, outputs, width), "output_bias": (sessions, outputs),
    }
    norms = result.get("effective_map_norms")
    require(isinstance(norms, dict) and set(norms) == set(MAP_KEYS), f"{label} effective-map norms missing")
    for name, value in maps.items():
        require(value.shape == expected_shapes[name] and np.isfinite(value).all(), f"{label} map shape/finite check failed: {name}")
        expected_norm = norms[name]
        actual_norm = float(np.linalg.norm(value.astype(np.float64)))
        require(isinstance(expected_norm, (int, float)) and math.isfinite(expected_norm)
                and math.isclose(actual_norm, float(expected_norm), rel_tol=1e-9, abs_tol=1e-9),
                f"{label} map norm differs: {name}")
    return maps


def exact_number(left: Any, right: Any, label: str) -> None:
    require(isinstance(left, (int, float)) and not isinstance(left, bool) and math.isfinite(float(left)),
            f"{label} left value is malformed")
    require(isinstance(right, (int, float)) and not isinstance(right, bool) and math.isfinite(float(right)),
            f"{label} right value is malformed")
    require(left == right, f"{label} differs")


def check_fold(result: dict[str, Any], label: str, source_sessions: list[str] | None = None) -> list[dict[str, Any]]:
    """Check fold receipts without treating failed folding as failed unmerged research."""
    source = source_sessions is not None
    if source:
        receipts = result.get("source_fold_replays")
        require(isinstance(receipts, dict) and set(receipts) == set(source_sessions), f"{label} source fold receipts differ")
        metrics = result.get("selected_full_validation_r2_by_session")
        require(isinstance(metrics, dict) and set(metrics) == set(source_sessions), f"{label} source fold metric roster differs")
        require(result.get("source_selection_metric") == "macro mean physical R2 across source sessions",
                f"{label} source selection metric differs")
        values = [(session, receipts[session]) for session in source_sessions]
    else:
        receipts = result.get("fold_replay")
        require(isinstance(receipts, dict), f"{label} fold receipt missing")
        values = [(None, receipts)]
    failures: list[dict[str, Any]] = []
    for session, receipt in values:
        require(isinstance(receipt, dict), f"{label} fold receipt is malformed")
        tolerance = receipt.get("r2_tolerance")
        require(isinstance(receipt.get("point_allclose_pass"), bool)
                and isinstance(receipt.get("score_pass"), bool)
                and isinstance(receipt.get("fold_accepted"), bool), f"{label} fold receipt flags malformed")
        require(receipt["fold_accepted"] == (receipt["point_allclose_pass"] and receipt["score_pass"]),
                f"{label} fold acceptance gate differs")
        require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and math.isfinite(float(tolerance))
                and float(tolerance) > 0, f"{label} fold tolerance differs")
        for field in ("unmerged_r2", "folded_r2", "r2_delta", "normalized_max_abs"):
            require(isinstance(receipt.get(field), (int, float)) and not isinstance(receipt.get(field), bool)
                    and math.isfinite(float(receipt[field])), f"{label} fold {field} is malformed")
        exact_number(receipt["r2_delta"], receipt["folded_r2"] - receipt["unmerged_r2"], f"{label} fold R2 delta")
        require(receipt["score_pass"] == (abs(float(receipt["r2_delta"])) <= float(tolerance)),
                f"{label} fold score gate differs")
        point_tolerance = receipt.get("normalized_point_tolerance")
        require(isinstance(point_tolerance, dict) and set(point_tolerance) == {"atol", "rtol"}
                and all(isinstance(point_tolerance[key], (int, float)) and not isinstance(point_tolerance[key], bool)
                        and math.isfinite(float(point_tolerance[key])) and float(point_tolerance[key]) > 0
                        for key in point_tolerance), f"{label} normalized fold tolerance differs")
        require(isinstance(receipt.get("validation_endpoints"), int) and not isinstance(receipt.get("validation_endpoints"), bool)
                and receipt["validation_endpoints"] > 0, f"{label} fold endpoint count differs")
        if source:
            exact_number(result["selected_full_validation_r2_by_session"][session], receipt["unmerged_r2"],
                         f"{label} selected full-validation R2 for {session}")
        else:
            exact_number(result.get("fold_prefix_r2"), receipt["folded_r2"], f"{label} folded prefix R2")
            exact_number(result.get("fold_prefix_r2_delta"), receipt["r2_delta"], f"{label} folded prefix R2 delta")
            require(result.get("fold_prefix_r2_pass") is receipt["score_pass"], f"{label} folded prefix pass differs")
            require(result.get("deployment_fold_accepted") is receipt["fold_accepted"],
                    f"{label} deployment fold gate differs")
        if not receipt["fold_accepted"]:
            failures.append({"session": session, "folded_deployment_ineligible": True,
                             "point_allclose_pass": receipt["point_allclose_pass"], "score_pass": receipt["score_pass"]})
    return failures


def check_training_log(path: Path, result: dict[str, Any], cfg: dict[str, Any], label: str) -> None:
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {label} training log") from error
    require(isinstance(log, list) and log, f"{label} training log is malformed")
    require_finite_json(log, f"{label} training log")
    trainable, best, budget = result.get("trainable_parameters"), result.get("best_step"), result.get("final_budget")
    completed = result.get("completed_steps")
    require(all(isinstance(value, int) and not isinstance(value, bool) for value in (trainable, best, budget, completed))
            and trainable >= 0 and budget >= 0 and completed >= 0, f"{label} convergence fields malformed")
    require(completed == budget, f"{label} completed steps differ from final budget")
    require(len(log) == completed + 1, f"{label} training log length differs from completed steps")
    require(log[0] == {"step": 0, "val_r2": log[0].get("val_r2")}, f"{label} initial log event differs")
    require(isinstance(log[0]["val_r2"], (int, float)) and not isinstance(log[0]["val_r2"], bool),
            f"{label} initial validation differs")
    scores: list[tuple[int, float]] = [(0, float(log[0]["val_r2"]))]
    for step, event in enumerate(log[1:], start=1):
        require(isinstance(event, dict) and event.get("step") == step, f"{label} training log step order differs")
        require({"step", "data_loss", "penalty", "gradient_norm"}.issubset(event), f"{label} training event fields differ")
        allowed = {"step", "data_loss", "penalty", "gradient_norm", "val_r2"}
        require(set(event).issubset(allowed), f"{label} training event has unknown fields")
        if "val_r2" in event:
            require(step % cfg["val_interval"] == 0 or step == budget, f"{label} validation log cadence differs")
            scores.append((step, float(event["val_r2"])))
    if completed == 0:
        require(len(log) == 1 and best == 0, f"{label} untrained log differs")
    else:
        require(log[-1]["step"] == completed and "val_r2" in log[-1], f"{label} terminal validation is missing")
    observed_best, observed_best_step = scores[0][1], 0
    for step, score in scores[1:]:
        if score > observed_best:  # matches the runner's strict tie rule
            observed_best, observed_best_step = score, step
    exact_number(result.get("best_val_r2"), observed_best, f"{label} best validation R2")
    require(best == observed_best_step, f"{label} best validation step differs")
    extensions = result.get("extensions")
    require(isinstance(extensions, list), f"{label} extension receipt differs")
    current_budget = 0 if trainable == 0 else cfg["steps"]
    for extension in extensions:
        require(isinstance(extension, dict) and set(extension) == {"old", "new", "best_step"}, f"{label} extension schema differs")
        old, new, at_step = extension["old"], extension["new"], extension["best_step"]
        require(all(isinstance(value, int) and not isinstance(value, bool) for value in (old, new, at_step)),
                f"{label} extension values are malformed")
        require(old == current_budget and old in {2000, 4000} and new == old * 2 and new <= cfg["max_steps"],
                f"{label} extension chain differs")
        prefix_best, prefix_step = scores[0][1], 0
        for step, score in scores[1:]:
            if step > old:
                break
            if score > prefix_best:
                prefix_best, prefix_step = score, step
        require(at_step == prefix_step and at_step >= .8 * old, f"{label} extension justification differs")
        current_budget = new
    require(budget == current_budget and budget in {0, 2000, 4000, 8000}, f"{label} final budget differs")
    expected = trainable == 0 or best < 0.8 * budget
    require(result.get("converged") is expected, f"{label} convergence receipt differs")
    require(result.get("frozen_parameter_audit_pass") is True, f"{label} frozen parameter audit failed")


def check_trainable_scope(result: dict[str, Any], variant: dict[str, Any], label: str, source: bool,
                          selected_state: dict[str, torch.Tensor]) -> None:
    selected = result.get("selected_parameter_hashes")
    names = result.get("trainable_names")
    count = result.get("trainable_parameters")
    total = result.get("total_parameters")
    require(isinstance(selected, dict) and selected and all(isinstance(name, str) and require_digest(value, f"{label} selected hash {name}")
            for name, value in selected.items()), f"{label} selected parameter hashes are malformed")
    require(isinstance(names, list) and all(isinstance(name, str) for name in names) and len(names) == len(set(names)),
            f"{label} trainable names are malformed")
    require(isinstance(count, int) and not isinstance(count, bool) and isinstance(total, int) and not isinstance(total, bool),
            f"{label} parameter counts are malformed")
    require(set(selected_state) == set(selected), f"{label} selected checkpoint parameter roster differs")
    require(all(name in selected_state for name in names), f"{label} trainable parameter is absent from checkpoint")
    require(count == sum(selected_state[name].numel() for name in names), f"{label} trainable parameter count differs")
    require(total == sum(value.numel() for value in selected_state.values()), f"{label} total parameter count differs")
    if source:
        require(count == total and names == list(selected), f"{label} source training scope differs")
        return
    if result.get("method") == "none":
        require(count == 0 and names == [], f"{label} none training scope differs")
        return
    expected = ["bank.gain", "bank.bias", "bank.embedding"]
    if variant["unit_residual"]:
        expected.append("bank.unit_delta")
    if variant["session_readout"]:
        expected.extend(["bank.readout_log_gain", "bank.readout_bias"])
    require(names == expected and all(not name.startswith("base.") for name in names), f"{label} UI training scope differs")


def check_normalizer(path: Path, result: dict[str, Any], source_result: dict[str, Any], inputs: int, outputs: int, label: str) -> str:
    require(path.is_file() and sha256(path) == result.get("normalizer_sha256"), f"{label} normalizer differs")
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"x_mean", "x_std", "y_mean", "y_std"}, f"{label} normalizer keys differ")
            values = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise AuditError(f"cannot load {label} normalizer") from error
    for name, shape in (("x_mean", (inputs,)), ("x_std", (inputs,)), ("y_mean", (outputs,)), ("y_std", (outputs,))):
        value = values[name]
        require(value.dtype == np.dtype("float32") and value.shape == shape and np.isfinite(value).all(),
                f"{label} normalizer {name} shape, dtype, or finite check failed")
        if name.endswith("_std"):
            require(np.all(value > 0), f"{label} normalizer {name} is not positive")
    source_stats = source_result.get("source_statistics")
    require(isinstance(source_stats, list) and source_stats and isinstance(source_stats[0], dict),
            f"{label} source statistics are missing")
    for name in ("y_mean", "y_std"):
        expected = np.asarray(source_stats[0].get(name), dtype=np.float32)
        require(expected.shape == values[name].shape and np.array_equal(values[name], expected),
                f"{label} normalizer {name} differs from source statistics")
    return sha256(path)


def check_source(task_root: Path, task: str, contract: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    variants = cfg["variants"]
    expected_names = [variant["name"] for variant in variants]
    actual_names = sorted(path.name for path in (task_root / "source").iterdir() if path.is_dir())
    require(actual_names == sorted(expected_names), f"{task} source variant roster differs")
    source_results: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    fold_failures_all: list[dict[str, Any]] = []
    initial_bases: list[dict[str, Any]] = []
    plan = contract["plan"]
    source_sessions = plan["source_held_in_sessions"]
    held = {row["session"]: row["sha256"] for row in plan["held_in"]}
    task_cfg = cfg["tasks"][task]
    inputs, outputs = task_dims(task)
    for variant in variants:
        name = variant["name"]
        directory = task_root / "source" / name
        require_file_roster(directory, SOURCE_FILES, f"{task}/{name} source")
        result = read_json(directory / "fit_result.json", f"{task}/{name} source result")
        require(result.get("task") == task and result.get("variant") == variant, f"{task}/{name} source identity differs")
        require(result.get("target_recording_loaded") is False and result.get("query_labels_used") is False,
                f"{task}/{name} source loaded target/query")
        json_equal(result.get("source_sessions"), source_sessions, f"{task}/{name} source session roster")
        json_equal(result.get("source_data_sha256"), {session: held[session] for session in source_sessions},
                   f"{task}/{name} source data hashes")
        check_training_log(directory / "train_log.json", result, cfg, f"{task}/{name} source")
        fold_failures = check_fold(result, f"{task}/{name} source", source_sessions=source_sessions)
        fold_failures_all.extend({"variant": name, **failure} for failure in fold_failures)
        metadata = {"task": task, "task_cfg": task_cfg, "variant": variant, "cfg": cfg, "inputs": inputs,
                    "outputs": outputs, "sessions": source_session_count(task),
                    "contract_sha256": sha256(task_root / "contract.json")}
        state = checkpoint_hashes(directory / "best.pt", result.get("selected_parameter_hashes"), metadata, result.get("best_step"),
                                  f"{task}/{name} source best checkpoint")
        checkpoint_metadata(directory / "last.pt", metadata, result.get("completed_steps"), f"{task}/{name} source last checkpoint")
        require(sha256(directory / "best.pt") == result.get("best_checkpoint_sha256"), f"{task}/{name} source checkpoint digest differs")
        require(sha256(directory / "train_log.json") == result.get("train_log_sha256"), f"{task}/{name} source log digest differs")
        check_trainable_scope(result, variant, f"{task}/{name} source", source=True, selected_state=state)
        maps = load_maps(directory / "effective_maps.npz", result, len(source_sessions), task_cfg["width"],
                         inputs, outputs, f"{task}/{name} source")
        statistics = result.get("source_statistics")
        require(isinstance(statistics, list) and len(statistics) == len(source_sessions),
                f"{task}/{name} source statistics roster differs")
        for index, statistic in enumerate(statistics):
            require(isinstance(statistic, dict) and set(statistic) == {"x_mean", "x_std", "y_mean", "y_std"},
                    f"{task}/{name} source statistic {index} schema differs")
            for field, length in (("x_mean", inputs), ("x_std", inputs), ("y_mean", outputs), ("y_std", outputs)):
                values = np.asarray(statistic[field], dtype=np.float32)
                require(values.shape == (length,) and np.isfinite(values).all(),
                        f"{task}/{name} source statistic {index}/{field} differs")
                if field.endswith("_std"):
                    require(np.all(values > 0), f"{task}/{name} source statistic {index}/{field} is not positive")
        require_digest(result.get("source_validation_selections_sha256"), f"{task}/{name} source validation selection SHA")
        initial = result.get("initial_base_parameter_hashes")
        require(isinstance(initial, dict) and initial, f"{task}/{name} source initial base hashes missing")
        initial_bases.append(initial)
        source_results[name] = result
        evidence.append({"variant": name, "best_prefix_val_r2": result["best_val_r2"], "best_step": result["best_step"],
                         "completed_steps": result["completed_steps"], "effective_maps_sha256": result["effective_maps_sha256"],
                         "source_fold_accepted": {session: receipt["fold_accepted"]
                                                   for session, receipt in result["source_fold_replays"].items()},
                         "folded_deployment_ineligible": fold_failures})
        del maps
    require(all(base == initial_bases[0] for base in initial_bases[1:]), f"{task} source initial base differs across variants")
    return source_results, evidence, fold_failures_all


def check_target(
    task_root: Path, task: str, contract: dict[str, Any], cfg: dict[str, Any], source_results: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    variants = cfg["variants"]
    expected_names = [variant["name"] for variant in variants]
    actual_names = sorted(path.name for path in (task_root / "adapt").iterdir() if path.is_dir())
    require(actual_names == sorted(expected_names), f"{task} target variant roster differs")
    task_cfg = cfg["tasks"][task]
    inputs, outputs = task_dims(task)
    receipt, validation_ids, train_bounds, val_bounds = support_partition(contract, task, cfg)
    all_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    fold_failures: list[dict[str, Any]] = []
    normalizer_hashes: list[str] = []
    prefix_selection_hashes: list[str] = []
    for variant in variants:
        name = variant["name"]
        source_dir = task_root / "source" / name
        with np.load(source_dir / "effective_maps.npz", allow_pickle=False) as archive:
            source_maps = {key: np.asarray(archive[key]) for key in MAP_KEYS}
        rows: list[dict[str, Any]] = []
        for directory in sorted((task_root / "adapt" / name).iterdir()):
            require(directory.is_dir(), f"{task}/{name} target artifact is not a directory")
            result = read_json(directory / "fit_result.json", f"{task}/{name} target result")
            rows.append(result)
            require(result.get("task") == task and result.get("variant") == name, f"{task}/{name} target identity differs")
            require(result.get("method") in {"none", "ui"} and result.get("query_labels_used") is False,
                    f"{task}/{name} target method/query flag differs")
            require_target_directory_name(directory, result.get("method"), result.get("seed"), result.get("lr"),
                                          f"{task}/{name} target")
            expected_files = TARGET_FILES | ({"progress.json"} if result["method"] == "ui" else set())
            require_file_roster(directory, frozenset(expected_files), f"{task}/{name}/{directory.name} target")
            require(result.get("backbone_frozen") is True and result.get("source_initializer") == "arithmetic mean of effective source input and normalized-output maps",
                    f"{task}/{name} target provenance differs")
            json_equal(result.get("calibration_receipt"), receipt, f"{task}/{name} target calibration")
            require(result.get("source_checkpoint_sha256") == sha256(source_dir / "best.pt"), f"{task}/{name} target source checkpoint differs")
            require_exact_bounds(result.get("validation_raw_trial_indices"), validation_ids,
                                 f"{task}/{name}/{directory.name} raw validation trial IDs")
            require_exact_bounds(result.get("train_bounds"), train_bounds, f"{task}/{name}/{directory.name} train bounds")
            require_exact_bounds(result.get("val_bounds"), val_bounds, f"{task}/{name}/{directory.name} validation bounds")
            prefix_digest = require_digest(result.get("prefix_val_indices_sha256"),
                                           f"{task}/{name}/{directory.name} prefix validation indices SHA")
            prefix_selection_hashes.append(prefix_digest)
            check_training_log(directory / "train_log.json", result, cfg, f"{task}/{name}/{directory.name}")
            fold_failures.extend({"variant": name, "method": result["method"], "seed": result["seed"], "lr": result["lr"], **failure}
                                 for failure in check_fold(result, f"{task}/{name}/{directory.name}"))
            normalizer_hashes.append(check_normalizer(directory / "normalizer.npz", result, source_results[name], inputs, outputs,
                                                      f"{task}/{name}/{directory.name}"))
            require(sha256(directory / "folded.pt") == result.get("folded_checkpoint_sha256"), f"{task}/{name}/{directory.name} folded checkpoint differs")
            require(sha256(directory / "best.pt") == result.get("best_checkpoint_sha256"), f"{task}/{name}/{directory.name} best checkpoint differs")
            require(sha256(directory / "train_log.json") == result.get("train_log_sha256"), f"{task}/{name}/{directory.name} train log differs")
            metadata = {"task": task, "task_cfg": task_cfg, "variant": variant, "cfg": cfg, "inputs": inputs,
                        "outputs": outputs, "sessions": 1, "contract_sha256": sha256(task_root / "contract.json"),
                        "method": result["method"], "seed": result["seed"], "lr": result["lr"],
                        "source_checkpoint_sha256": result["source_checkpoint_sha256"]}
            state = checkpoint_hashes(directory / "best.pt", result.get("selected_parameter_hashes"), metadata, result.get("best_step"),
                                      f"{task}/{name}/{directory.name} best checkpoint")
            checkpoint_metadata(directory / "last.pt", metadata, result.get("completed_steps"),
                                f"{task}/{name}/{directory.name} last checkpoint")
            checkpoint_metadata(directory / "folded.pt", metadata, None, f"{task}/{name}/{directory.name} folded checkpoint")
            initial = result.get("initial_parameter_hashes")
            selected = result.get("selected_parameter_hashes")
            require(isinstance(initial, dict) and isinstance(selected, dict), f"{task}/{name}/{directory.name} parameter hashes missing")
            base_initial = {key: value for key, value in initial.items() if key.startswith("base.")}
            base_selected = {key: value for key, value in selected.items() if key.startswith("base.")}
            source_base = {key: value for key, value in source_results[name]["selected_parameter_hashes"].items() if key.startswith("base.")}
            json_equal(base_initial, source_base, f"{task}/{name}/{directory.name} initial base provenance")
            json_equal(base_selected, base_initial, f"{task}/{name}/{directory.name} frozen base")
            check_trainable_scope(result, variant, f"{task}/{name}/{directory.name}", source=False, selected_state=state)
            target_maps = load_maps(directory / "effective_maps.npz", result, 1, task_cfg["width"], inputs, outputs,
                                    f"{task}/{name}/{directory.name}")
            if result["method"] == "none":
                for key in MAP_KEYS:
                    require(np.allclose(target_maps[key][0], source_maps[key].mean(0), rtol=1e-6, atol=1e-6),
                            f"{task}/{name} none map is not the source effective-map mean: {key}")
            all_rows.append(result)
        expected = [("none", 0, 0.0)] + [("ui", seed, lr) for lr in cfg["adapt_lrs"] for seed in cfg["adapt_seeds"]]
        actual = sorted((row.get("method"), row.get("seed"), float(row.get("lr"))) for row in rows)
        require(actual == sorted(expected), f"{task}/{name} target fit roster differs")
        for field in ("initial_parameter_hashes", "normalizer_sha256", "calibration_receipt"):
            values = [json.dumps(row[field], sort_keys=True) if isinstance(row[field], (dict, list)) else row[field] for row in rows]
            require(len(set(values)) == 1, f"{task}/{name} target {field} differs across fits")
        evidence.extend({"variant": name, "method": row["method"], "seed": row["seed"], "lr": row["lr"],
                         "best_prefix_val_r2": row["best_val_r2"], "best_step": row["best_step"]} for row in rows)
    require(len(all_rows) == 21, f"{task} target roster is not exactly 21 fits")
    require(len(set(normalizer_hashes)) == 1, f"{task} target normalizer differs across variants")
    require(len(set(prefix_selection_hashes)) == 1, f"{task} target prefix validation split differs across fits")
    return all_rows, evidence, fold_failures


def check_selection(task: str, cfg: dict[str, Any], rows: list[dict[str, Any]], completion: dict[str, Any]) -> list[dict[str, Any]]:
    expected_selected: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for variant in cfg["variants"]:
        name = variant["name"]
        family_rows = [row for row in rows if row["variant"] == name]
        none = [row for row in family_rows if row["method"] == "none"]
        require(len(none) == 1, f"{task}/{name} none roster differs")
        expected_selected.extend(none)
        candidates = []
        for lr in cfg["adapt_lrs"]:
            trial = [row for row in family_rows if row["method"] == "ui" and row["lr"] == lr]
            admissible = len(trial) == len(cfg["adapt_seeds"]) and all(row["converged"] for row in trial)
            candidates.append({"lr": lr, "admissible": admissible,
                               "mean_prefix_val_r2": float(np.mean([row["best_val_r2"] for row in trial]))})
        available = [row for row in candidates if row["admissible"]]
        if not available:
            decisions[name] = {"candidates": candidates, "selected_lr": None, "excluded": True}
            continue
        best = max(available, key=lambda row: (row["mean_prefix_val_r2"], -row["lr"]))
        decisions[name] = {"candidates": candidates, "selected_lr": best["lr"], "excluded": False}
        expected_selected.extend(row for row in family_rows if row["method"] == "ui" and row["lr"] == best["lr"])
    json_equal(completion.get("decisions"), decisions, f"{task} LR decisions")
    json_equal(completion.get("selected"), expected_selected, f"{task} selected roster")
    return [{"variant": row["variant"], "method": row["method"], "seed": row["seed"], "lr": row["lr"],
             "best_prefix_val_r2": row["best_val_r2"], "best_step": row["best_step"]} for row in expected_selected]


def audit_task(root: Path, task: str, cfg: dict[str, Any], config_path: Path, launcher_sha: str, runtime: dict[str, Any]) -> dict[str, Any]:
    task_root = root / task
    require(task_root.is_dir(), f"{task} task root missing")
    check_no_query_files(task_root)
    contract = check_contract(task_root, task, cfg, config_path, launcher_sha, runtime)
    completion = read_json(task_root / "fits_complete.json", f"{task} fit completion")
    require(completion.get("all_fits_completed_before_query") is True and completion.get("query_metrics_read") is False,
            f"{task} completion query flags differ")
    require(completion.get("source_fits") == 3 and completion.get("target_fits") == 21, f"{task} completion fit counts differ")
    artifact_count = check_artifact_tree(task_root, completion, set(contract["code_hashes"]))
    source_results, source_evidence, source_fold_failures = check_source(task_root, task, contract, cfg)
    target_rows, target_evidence, target_fold_failures = check_target(task_root, task, contract, cfg, source_results)
    selected = check_selection(task, cfg, target_rows, completion)
    return {
        "contract_sha256": sha256(task_root / "contract.json"),
        "fits_complete_sha256": sha256(task_root / "fits_complete.json"),
        "artifact_count": artifact_count,
        "source_fit_count": len(source_results),
        "target_fit_count": len(target_rows),
        "selected_fit_count": len(selected),
        "source_prefix_metrics": source_evidence,
        "source_folded_deployment_ineligible": source_fold_failures,
        "source_folded_deployment_ineligible_count": len(source_fold_failures),
        "target_prefix_metrics": target_evidence,
        "selected_prefix_metrics": selected,
        "target_folded_deployment_ineligible": target_fold_failures,
        "target_folded_deployment_ineligible_count": len(target_fold_failures),
        "query_artifacts_present": False,
    }


def check_seal(root: Path, tasks: dict[str, dict[str, Any]], expected_sha: str) -> dict[str, Any]:
    seal_path = root / "fit_seal.json"
    require(seal_path.is_file(), "top-level fit seal is missing")
    require(sha256(seal_path) == expected_sha, "top-level fit seal SHA differs from external input")
    seal = read_json(seal_path, "top-level fit seal")
    require(seal.get("schema") == "unit_session_fit_seal_v1" and set(seal) == {"schema", "tasks"}, "fit seal schema differs")
    bindings = seal.get("tasks")
    require(isinstance(bindings, dict) and set(bindings) == set(tasks), "fit seal task roster differs")
    for task, report in tasks.items():
        binding = bindings[task]
        require(isinstance(binding, dict) and set(binding) == {"completion_sha256", "contract_sha256"},
                f"fit seal {task} binding differs")
        require(binding["completion_sha256"] == report["fits_complete_sha256"], f"fit seal {task} completion differs")
        require(binding["contract_sha256"] == report["contract_sha256"], f"fit seal {task} contract differs")
    return {"fit_seal_sha256": expected_sha, "task_bindings_verified": sorted(tasks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="top-level fit output root")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-launcher-sha256", required=True)
    parser.add_argument("--expected-fit-seal-sha256")
    parser.add_argument("--dry-run-task", choices=("m1", "m2"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        root, config_path = args.root.resolve(), args.config.resolve()
        require(root.is_dir() and config_path.is_file(), "root or config is missing")
        launcher_sha = require_digest(args.frozen_launcher_sha256, "frozen launcher SHA")
        cfg = read_json(config_path, "config")
        require(cfg.get("schema") == "unit_session_iteration_v1" and set(cfg.get("tasks", {})) == {"m1", "m2"},
                "config task schema differs")
        reference_task = args.dry_run_task or "m1"
        reference_contract = read_json(root / reference_task / "contract.json", f"{reference_task} contract")
        runtime = validate_frozen_runtime(reference_contract.get("runtime"))
        check_matrix(root, cfg, config_path, launcher_sha)
        if args.dry_run_task:
            require(args.report is None, "dry run must not write a final report")
            tasks = {args.dry_run_task: audit_task(root, args.dry_run_task, cfg, config_path, launcher_sha, runtime)}
            report = {"schema": "unit_session_fit_completion_audit_v1", "status": "PASS", "mode": "dry_run",
                      "root": str(root), "config_sha256": sha256(config_path), "launcher_sha256": launcher_sha,
                      "runtime": runtime, "tasks": tasks}
        else:
            expected_seal = require_digest(args.expected_fit_seal_sha256, "external fit seal SHA")
            tasks = {task: audit_task(root, task, cfg, config_path, launcher_sha, runtime) for task in cfg["tasks"]}
            report = {"schema": "unit_session_fit_completion_audit_v1", "status": "PASS", "mode": "final",
                      "root": str(root), "config_sha256": sha256(config_path), "launcher_sha256": launcher_sha,
                      "runtime": runtime, "tasks": tasks,
                      "seal": check_seal(root, tasks, expected_seal)}
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.report:
            output = args.report.resolve()
            require(output.parent == Path(__file__).resolve().parent, "report path must remain in this review directory")
            output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except AuditError as error:
        print(json.dumps({"schema": "unit_session_fit_completion_audit_v1", "status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
