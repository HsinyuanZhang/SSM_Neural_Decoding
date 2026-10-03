#!/usr/bin/env python3
"""Independent CPU-only verifier for frozen unit/session score artifacts.

The verifier intentionally imports neither the experiment runner nor the
analysis program.  It reads only frozen JSON/CSV/NPZ artifacts and uses NumPy
to independently reconstruct all scalar results.  It never opens NWB files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
HEX = re.compile(r"[0-9a-f]{64}\Z")
TASK_DIMS = {"m1": 16, "m2": 2}
SOURCE_SESSION_COUNTS = {"m1": 3, "m2": 6}
EXPECTED_ANALYZER_SHA256 = "4f40a41b9411c7b965738139911605e8f483a58498558532f77d5e2eba6697fe"
TASK_SUMMARY_KEYS = frozenset({
    "schema", "task", "rows", "all_fits_completed_before_query",
    "query_used_for_checkpoint_or_lr_selection", "architecture_comparison_uses_local_query",
    "official_heldout_result",
})
SCORE_ROW_KEYS = frozenset({
    "task", "variant", "method", "seed", "lr", "r2", "query_bins", "best_step",
    "best_prefix_val_r2", "trainable_parameters", "query_seconds", "prediction_archive",
    "prediction_archive_sha256", "best_checkpoint_sha256", "indices_sha256", "truth_sha256",
    "folded_prefix_r2_delta",
})
ANALYSIS_SUMMARY_KEYS = frozenset({
    "schema", "status", "local_development_only", "official_result",
    "query_used_for_checkpoint_lr_or_seed_selection",
    "query_used_for_local_architecture_exploration_comparison", "source_seed", "target_seeds",
    "rows", "bindings", "directional_diagnostic",
})


class AuditError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AuditError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise AuditError(f"cannot hash file: {path}") from error


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def is_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def require_digest(value: Any, label: str) -> str:
    require(is_digest(value), f"{label} is not a lowercase SHA-256 digest")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"{label} is not a finite number")
    return float(value)


def require_int(value: Any, label: str, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} is not an integer")
    if minimum is not None:
        require(value >= minimum, f"{label} is below its minimum")
    return value


def fit_name(row: dict[str, Any]) -> str:
    return f"{row['method']}_seed{row['seed']}_lr{float(row['lr']):g}"


def row_key(row: dict[str, Any], label: str) -> tuple[str, str, int, float]:
    variant, method = row.get("variant"), row.get("method")
    require(isinstance(variant, str) and isinstance(method, str), f"{label} variant or method is malformed")
    return variant, method, require_int(row.get("seed"), f"{label} seed", 0), require_number(row.get("lr"), f"{label} lr")


def exact(left: Any, right: Any, label: str) -> None:
    if left != right:
        fail(f"{label} differs")


def exact_scalar(left: Any, right: Any, label: str) -> None:
    """Require literal equality after separately rejecting NaN and bool coercions."""
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        require_number(left, f"{label} left")
        require_number(right, f"{label} right")
    exact(left, right, label)


def physical_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth64, prediction64 = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    require(truth64.shape == prediction64.shape and truth64.ndim == 2 and np.isfinite(truth64).all()
            and np.isfinite(prediction64).all(), "physical R2 inputs differ or are non-finite")
    denominator = max(float(np.square(truth64 - truth64.mean(axis=0)).sum()), 1e-12)
    return float(1 - np.square(truth64 - prediction64).sum() / denominator)


def load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as error:
        raise AuditError(f"cannot load {label}: {path}") from error


def validate_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path, "config")
    require(config.get("schema") == "unit_session_iteration_v1" and set(config.get("tasks", {})) == set(TASK_DIMS),
            "config task schema differs")
    matrix = read_json(root / "matrix.json", "frozen matrix")
    exact(matrix.get("config"), config, "matrix frozen config")
    require(matrix.get("config_sha256") == sha256(config_path), "matrix config SHA differs")
    return config


def verify_fit_bindings(root: Path, config: dict[str, Any], config_path: Path, expected_seal_sha: str,
                        fit_audit_path: Path, expected_fit_audit_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every score input before touching a score summary or NPZ archive."""
    require_digest(expected_seal_sha, "external fit seal SHA")
    require_digest(expected_fit_audit_sha, "external fit audit SHA")
    seal_path = root / "fit_seal.json"
    require(sha256(seal_path) == expected_seal_sha, "external fit seal SHA differs")
    seal = read_json(seal_path, "fit seal")
    require(set(seal) == {"schema", "tasks"} and seal.get("schema") == "unit_session_fit_seal_v1",
            "fit seal schema differs")
    bindings = seal.get("tasks")
    require(isinstance(bindings, dict) and set(bindings) == set(TASK_DIMS), "fit seal task roster differs")

    require(sha256(fit_audit_path) == expected_fit_audit_sha, "external fit-completion audit SHA differs")
    audit = read_json(fit_audit_path, "fit-completion audit")
    require(audit.get("schema") == "unit_session_fit_completion_audit_v1" and audit.get("status") == "PASS"
            and audit.get("mode") == "final", "fit-completion audit status differs")
    audit_seal = audit.get("seal")
    require(isinstance(audit_seal, dict) and audit_seal.get("fit_seal_sha256") == expected_seal_sha
            and audit_seal.get("task_bindings_verified") == sorted(TASK_DIMS), "fit-completion audit seal binding differs")
    task_audits = audit.get("tasks")
    require(isinstance(task_audits, dict) and set(task_audits) == set(TASK_DIMS), "fit-completion audit task roster differs")

    for task in sorted(TASK_DIMS):
        binding = bindings[task]
        require(isinstance(binding, dict) and set(binding) == {"completion_sha256", "contract_sha256"},
                f"fit seal {task} schema differs")
        require(all(is_digest(value) for value in binding.values()), f"fit seal {task} digest differs")
        completion_path, contract_path = root / task / "fits_complete.json", root / task / "contract.json"
        require(sha256(completion_path) == binding["completion_sha256"], f"{task} frozen completion changed")
        require(sha256(contract_path) == binding["contract_sha256"], f"{task} frozen contract changed")
        task_audit = task_audits[task]
        require(isinstance(task_audit, dict) and task_audit.get("fits_complete_sha256") == binding["completion_sha256"]
                and task_audit.get("contract_sha256") == binding["contract_sha256"]
                and task_audit.get("query_artifacts_present") is False, f"{task} fit audit binding differs")
        contract = read_json(contract_path, f"{task} contract")
        require(contract.get("cfg") == config and contract.get("config_sha256") == sha256(config_path),
                f"{task} contract config differs")
    return seal, audit


def validate_archive(path: Path, task: str, task_cfg: dict[str, Any], metric: dict[str, Any], label: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_npz(path, label)
    require(set(arrays) == {"indices", "truth", "prediction"}, f"{label} archive keys differ")
    indices, truth, prediction = arrays["indices"], arrays["truth"], arrays["prediction"]
    count, outputs = task_cfg["expected_query_bins"], TASK_DIMS[task]
    require(indices.dtype == np.dtype("int64") and indices.ndim == 1 and indices.shape == (count,), f"{label} indices differ")
    require(truth.dtype == np.dtype("float32") and prediction.dtype == np.dtype("float32")
            and truth.shape == (count, outputs) and prediction.shape == truth.shape
            and np.isfinite(truth).all() and np.isfinite(prediction).all(), f"{label} truth or prediction differs")
    indices_sha, truth_sha = array_sha256(indices), array_sha256(truth)
    require(indices_sha == task_cfg["expected_indices_sha256"] and truth_sha == task_cfg["expected_truth_sha256"],
            f"{label} canonical cohort or truth differs")
    exact(metric.get("indices_sha256"), indices_sha, f"{label} metric indices SHA")
    exact(metric.get("truth_sha256"), truth_sha, f"{label} metric truth SHA")
    return indices, truth


def directional_ratio(source_result: dict[str, Any], source_dir: Path, label: str) -> float:
    maps_path = source_dir / "effective_maps.npz"
    require(sha256(maps_path) == source_result.get("effective_maps_sha256"), f"{label} effective maps SHA differs")
    arrays = load_npz(maps_path, f"{label} effective maps")
    require(set(arrays) == {"input_weight", "input_bias", "output_weight", "output_bias"}, f"{label} effective map keys differ")
    values = np.asarray(arrays["input_weight"], np.float64)
    require(values.ndim == 3 and all(values.shape) and np.isfinite(values).all(), f"{label} effective input map differs")
    mean = values.mean(axis=0)
    orthogonal_energy = 0.0
    total_energy = float(np.square(values).sum())
    for channel in range(values.shape[2]):
        direction = mean[:, channel]
        norm = float(direction @ direction)
        channel_values = values[:, :, channel]
        projected = np.zeros_like(channel_values) if norm <= 1e-30 else np.outer(channel_values @ direction / norm, direction)
        orthogonal_energy += float(np.square(channel_values - projected).sum())
    return orthogonal_energy / max(total_energy, 1e-30)


def validate_fold(result: dict[str, Any], label: str) -> bool:
    receipt = result.get("fold_replay")
    require(isinstance(receipt, dict), f"{label} fold receipt is missing")
    point, score, accepted = receipt.get("point_allclose_pass"), receipt.get("score_pass"), receipt.get("fold_accepted")
    delta, tolerance = receipt.get("r2_delta"), receipt.get("r2_tolerance")
    require(isinstance(point, bool) and isinstance(score, bool) and isinstance(accepted, bool), f"{label} fold flags differ")
    require_number(delta, f"{label} fold delta"); require_number(tolerance, f"{label} fold tolerance")
    require(score == (abs(float(delta)) <= float(tolerance)) and accepted == (point and score), f"{label} fold receipt differs")
    exact_scalar(result.get("fold_prefix_r2_delta"), delta, f"{label} fold delta binding")
    require(result.get("deployment_fold_accepted") is accepted, f"{label} deployment fold binding differs")
    return accepted


def rebuild_selection(root: Path, task: str, config: dict[str, Any], completion: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    variants = config["variants"]
    decisions = completion.get("decisions")
    require(isinstance(decisions, dict) and set(decisions) == {variant["name"] for variant in variants}, f"{task} decisions roster differs")
    expected_selected: list[dict[str, Any]] = []
    candidates_by_variant: dict[str, list[dict[str, Any]]] = {}
    fold_failures: list[dict[str, Any]] = []
    for variant in variants:
        name = variant["name"]
        task_root = root / task / "adapt" / name
        none_path = task_root / "none_seed0_lr0" / "fit_result.json"
        none = read_json(none_path, f"{task}/{name} none fit")
        require(row_key(none, f"{task}/{name} none") == (name, "none", 0, 0.0), f"{task}/{name} none identity differs")
        expected_selected.append(none)
        if not validate_fold(none, f"{task}/{name}/none"):
            fold_failures.append({"task": task, "variant": name, "method": "none", "seed": 0, "lr": 0.0,
                                  "folded_deployment_ineligible": True})
        rebuilt: list[dict[str, Any]] = []
        for lr in config["adapt_lrs"]:
            family: list[dict[str, Any]] = []
            for seed in config["adapt_seeds"]:
                path = task_root / f"ui_seed{seed}_lr{float(lr):g}" / "fit_result.json"
                fit = read_json(path, f"{task}/{name} UI candidate")
                require(row_key(fit, f"{task}/{name} UI candidate") == (name, "ui", seed, float(lr)),
                        f"{task}/{name} UI candidate identity differs")
                require(fit.get("task") == task and fit.get("converged") in {True, False},
                        f"{task}/{name} UI candidate convergence differs")
                family.append(fit)
                if not validate_fold(fit, f"{task}/{name}/ui/{seed}/{lr}"):
                    fold_failures.append({"task": task, "variant": name, "method": "ui", "seed": seed, "lr": float(lr),
                                          "folded_deployment_ineligible": True})
            require({fit["seed"] for fit in family} == set(config["adapt_seeds"]), f"{task}/{name}/{lr} seed roster differs")
            rebuilt.append({"lr": float(lr), "admissible": all(fit["converged"] is True for fit in family),
                            "mean_prefix_val_r2": float(np.mean(
                                [require_number(fit.get("best_val_r2"), f"{task}/{name}/{lr} best validation") for fit in family],
                                dtype=np.float64)), "fits": family})
        decision = decisions[name]
        require(isinstance(decision, dict) and set(decision) == {"candidates", "selected_lr", "excluded"},
                f"{task}/{name} decision schema differs")
        saved = decision["candidates"]
        require(isinstance(saved, list) and len(saved) == len(rebuilt), f"{task}/{name} candidate count differs")
        saved_by_lr = {require_number(value.get("lr"), f"{task}/{name} saved LR"): value for value in saved if isinstance(value, dict)}
        require(len(saved_by_lr) == len(rebuilt) and set(saved_by_lr) == {value["lr"] for value in rebuilt},
                f"{task}/{name} candidate LR roster differs")
        for candidate in rebuilt:
            saved_candidate = saved_by_lr[candidate["lr"]]
            require(set(saved_candidate) == {"lr", "admissible", "mean_prefix_val_r2"}, f"{task}/{name} candidate schema differs")
            require(saved_candidate["admissible"] is candidate["admissible"], f"{task}/{name}/{candidate['lr']} admissibility differs")
            exact_scalar(saved_candidate["mean_prefix_val_r2"], candidate["mean_prefix_val_r2"],
                         f"{task}/{name}/{candidate['lr']} mean prefix R2")
        viable = [candidate for candidate in rebuilt if candidate["admissible"]]
        selected = max(viable, key=lambda candidate: (candidate["mean_prefix_val_r2"], -candidate["lr"])) if viable else None
        require(decision["excluded"] is (selected is None), f"{task}/{name} exclusion differs")
        exact(decision["selected_lr"], None if selected is None else selected["lr"], f"{task}/{name} selected LR")
        if selected is not None:
            expected_selected.extend(selected["fits"])
        candidates_by_variant[name] = rebuilt
    exact(completion.get("selected"), expected_selected, f"{task} selected fit roster")
    return expected_selected, candidates_by_variant, fold_failures


def verify_task(root: Path, task: str, config: dict[str, Any], seal: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_root = root / task
    binding = seal["tasks"][task]
    completion_path, contract_path = task_root / "fits_complete.json", task_root / "contract.json"
    require(sha256(completion_path) == binding["completion_sha256"] and sha256(contract_path) == binding["contract_sha256"],
            f"{task} completion or contract changed after seal")
    completion = read_json(completion_path, f"{task} completion")
    require(completion.get("all_fits_completed_before_query") is True and completion.get("query_metrics_read") is False,
            f"{task} completion query flags differ")
    selected, candidates, fold_failures = rebuild_selection(root, task, config, completion)
    selected_by_key = {row_key(row, f"{task} selected"): row for row in selected}
    require(len(selected_by_key) == len(selected), f"{task} selected fit identities duplicate")

    task_summary_path = task_root / "summary.json"
    summary = read_json(task_summary_path, f"{task} score summary")
    require(set(summary) == TASK_SUMMARY_KEYS and summary.get("schema") == "unit_session_local_results_v1" and summary.get("task") == task,
            f"{task} score summary schema differs")
    require(summary.get("all_fits_completed_before_query") is True
            and summary.get("query_used_for_checkpoint_or_lr_selection") is False
            and summary.get("architecture_comparison_uses_local_query") is True
            and summary.get("official_heldout_result") is False, f"{task} score summary flags differ")
    metric_rows = summary.get("rows")
    require(isinstance(metric_rows, list) and len(metric_rows) == len(selected), f"{task} score row count differs")
    metrics: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    expected_archives: set[Path] = set()
    canonical_indices: np.ndarray | None = None
    canonical_truth: np.ndarray | None = None
    scores_by_variant: dict[str, list[tuple[dict[str, Any], float]]] = {variant["name"]: [] for variant in config["variants"]}
    for metric in metric_rows:
        require(isinstance(metric, dict) and set(metric) == SCORE_ROW_KEYS, f"{task} score row schema differs")
        key = row_key(metric, f"{task} score row")
        require(key in selected_by_key and key not in metrics, f"{task} score row selection differs")
        selected_fit = selected_by_key[key]
        for field in ("task", "variant", "method", "seed", "lr", "best_step", "best_checkpoint_sha256", "trainable_parameters"):
            exact_scalar(metric.get(field), selected_fit.get(field), f"{task}/{key} {field}")
        exact_scalar(metric.get("best_prefix_val_r2"), selected_fit.get("best_val_r2"), f"{task}/{key} best prefix R2")
        exact_scalar(metric.get("folded_prefix_r2_delta"), selected_fit.get("fold_prefix_r2_delta"), f"{task}/{key} fold delta")
        require_int(selected_fit.get("best_step"), f"{task}/{key} selected best step", 0)
        require_int(selected_fit.get("trainable_parameters"), f"{task}/{key} selected trainable parameters", 0)
        require_number(selected_fit.get("fit_seconds"), f"{task}/{key} selected fit seconds")
        query_seconds = require_number(metric.get("query_seconds"), f"{task}/{key} query seconds")
        require(query_seconds >= 0 and metric.get("query_bins") == config["tasks"][task]["expected_query_bins"], f"{task}/{key} query timing or count differs")
        fit_dir = task_root / "adapt" / key[0] / fit_name(selected_fit)
        require(read_json(fit_dir / "fit_result.json", f"{task}/{key} selected fit result") == selected_fit,
                f"{task}/{key} selected fit changed")
        require(sha256(fit_dir / "best.pt") == selected_fit["best_checkpoint_sha256"], f"{task}/{key} best checkpoint changed")
        require(sha256(fit_dir / "normalizer.npz") == selected_fit["normalizer_sha256"], f"{task}/{key} normalizer changed")
        expected_archive = (task_root / "scores" / key[0] / f"{fit_name(selected_fit)}.npz").resolve()
        archive_text = metric.get("prediction_archive")
        require(isinstance(archive_text, str) and archive_text == str(expected_archive), f"{task}/{key} archive path differs")
        require(sha256(expected_archive) == metric.get("prediction_archive_sha256"), f"{task}/{key} archive SHA differs")
        indices, truth = validate_archive(expected_archive, task, config["tasks"][task], metric, f"{task}/{key}")
        archive = load_npz(expected_archive, f"{task}/{key} archive replay")
        score = physical_r2(truth, archive["prediction"])
        require(abs(score - require_number(metric.get("r2"), f"{task}/{key} saved R2")) <= 1e-12, f"{task}/{key} direct R2 differs")
        if canonical_indices is None:
            canonical_indices, canonical_truth = indices, truth
        else:
            require(np.array_equal(indices, canonical_indices) and np.array_equal(truth, canonical_truth),
                    f"{task} score cohort or truth differs across selected fits")
        metrics[key] = metric
        expected_archives.add(expected_archive)
        scores_by_variant[key[0]].append((selected_fit, score))
    require(set(metrics) == set(selected_by_key), f"{task} score row closure differs")

    scores_root = task_root / "scores"
    require(scores_root.is_dir() and not scores_root.is_symlink(), f"{task} scores root differs")
    actual_archives = {path.resolve() for path in scores_root.rglob("*") if path.is_file()}
    require(actual_archives == expected_archives, f"{task} scores file closure differs")
    require(not any(path.is_symlink() for path in scores_root.rglob("*")), f"{task} scores tree contains a symlink")
    expected_dirs = {scores_root.resolve()} | {path.parent for path in expected_archives}
    actual_dirs = {path.resolve() for path in scores_root.rglob("*") if path.is_dir()}
    require(actual_dirs == expected_dirs - {scores_root.resolve()}, f"{task} scores directory closure differs")

    rows: list[dict[str, Any]] = []
    for variant in config["variants"]:
        name = variant["name"]
        pairs = scores_by_variant[name]
        none = [(fit, score) for fit, score in pairs if fit["method"] == "none"]
        ui = [(fit, score) for fit, score in pairs if fit["method"] == "ui"]
        require(len(none) == 1, f"{task}/{name} none score roster differs")
        require(len({fit["normalizer_sha256"] for fit, _ in pairs}) == 1,
                f"{task}/{name} selected normalizer provenance differs")
        if ui:
            require(len(ui) == len(config["adapt_seeds"]) and {fit["seed"] for fit, _ in ui} == set(config["adapt_seeds"]),
                    f"{task}/{name} UI score seed roster differs")
        decision = completion["decisions"][name]
        require((not ui) is (decision["excluded"] is True), f"{task}/{name} selection score roster differs")
        source_dir = task_root / "source" / name
        source = read_json(source_dir / "fit_result.json", f"{task}/{name} source fit")
        source_variant = source.get("variant")
        source_variant = source_variant.get("name") if isinstance(source_variant, dict) else source_variant
        require(source.get("task") == task and source_variant == name, f"{task}/{name} source identity differs")
        source_scores = source.get("selected_full_validation_r2_by_session")
        require(isinstance(source_scores, dict) and len(source_scores) == SOURCE_SESSION_COUNTS[task],
                f"{task}/{name} source full-session scores differ")
        values = [require_number(value, f"{task}/{name} source score") for value in source_scores.values()]
        folds = source.get("source_fold_replays")
        require(isinstance(folds, dict) and set(folds) == set(source_scores), f"{task}/{name} source folds differ")
        source_fold_failed = 0
        for session, receipt in folds.items():
            require(isinstance(receipt, dict) and isinstance(receipt.get("point_allclose_pass"), bool)
                    and isinstance(receipt.get("score_pass"), bool) and isinstance(receipt.get("fold_accepted"), bool),
                    f"{task}/{name}/{session} source fold flags differ")
            require(receipt["fold_accepted"] == (receipt["point_allclose_pass"] and receipt["score_pass"]),
                    f"{task}/{name}/{session} source fold gate differs")
            exact_scalar(source_scores[session], receipt.get("unmerged_r2"), f"{task}/{name}/{session} source unmerged R2")
            source_fold_failed += int(receipt["fold_accepted"] is False)
        target_accepted = sum(int(validate_fold(fit, f"{task}/{name}/{fit_name(fit)}")) for fit, _ in pairs)
        require_int(source.get("trainable_parameters"), f"{task}/{name} source trainable parameters", 0)
        require_int(source.get("final_budget"), f"{task}/{name} source final budget", 0)
        require_number(source.get("fit_seconds"), f"{task}/{name} source fit seconds")
        candidate_rows = candidates[name]
        rejected = sum(int(candidate["admissible"] is False) for candidate in candidate_rows)
        late = sum(int(fit["converged"] is False) for candidate in candidate_rows for fit in candidate["fits"])
        ui_values = np.asarray([score for _, score in ui], dtype=np.float64)
        rows.append({
            "task": task, "variant": name, "none_r2": none[0][1],
            "selected_ui_r2_mean": None if not ui else float(np.mean(ui_values, dtype=np.float64)),
            "selected_ui_r2_population_sd": None if not ui else float(np.std(ui_values, dtype=np.float64)),
            "selected_ui_lr": None if not ui else float(ui[0][0]["lr"]), "ui_family_excluded": decision["excluded"],
            "lr_rejections": rejected, "family_late_count": late,
            "source_seed": config["source_seed"], "target_seeds": "" if not ui else ",".join(map(str, sorted(config["adapt_seeds"]))),
            "source_macro_full_validation_r2": float(np.mean(values, dtype=np.float64)),
            "source_full_session_scores": json.dumps(source_scores, sort_keys=True),
            "source_selection_macro_validation_r2": source["best_val_r2"],
            "source_directional_orthogonal_energy_ratio": directional_ratio(source, source_dir, f"{task}/{name}"),
            "source_directional_reference": "affine-only expected near zero; no biological alignment" if name == "affine" else "unit-direction residual enabled",
            "source_trainable_parameters": source["trainable_parameters"],
            "target_trainable_parameters": None if not ui else ui[0][0]["trainable_parameters"],
            "normalizer_sha256": none[0][0]["normalizer_sha256"], "source_fold_failed_count": source_fold_failed,
            "target_selected_fold_accepted_count": target_accepted, "target_selected_fold_count": len(pairs),
            "source_budget": source["final_budget"], "target_budgets": json.dumps([fit["final_budget"] for fit, _ in pairs]),
            "source_fit_seconds": source["fit_seconds"],
            "target_fit_seconds": float(sum(float(fit["fit_seconds"]) for fit, _ in pairs)),
        })
    return rows, fold_failures


def verify_analysis(analysis_dir: Path, root: Path, seal_sha: str, task_summary_sha: dict[str, str], expected_rows: list[dict[str, Any]]) -> None:
    analyzer_path = REPO / "scripts" / "analyze_unit_session_iteration.py"
    require(sha256(analyzer_path) == EXPECTED_ANALYZER_SHA256, "analyzer current SHA differs")
    summary = read_json(analysis_dir / "summary.json", "analyzer summary")
    require(set(summary) == ANALYSIS_SUMMARY_KEYS and summary.get("schema") == "unit_session_iteration_analysis_v1"
            and summary.get("status") == "verified" and summary.get("local_development_only") is True
            and summary.get("official_result") is False and summary.get("query_used_for_checkpoint_lr_or_seed_selection") is False
            and summary.get("query_used_for_local_architecture_exploration_comparison") is True,
            "analyzer summary schema or flags differ")
    bindings = summary.get("bindings")
    require(isinstance(bindings, dict) and set(bindings) == {"expected_fit_seal_sha256", "fit_seal_sha256", "task_summary_sha256", "analyzer_sha256"},
            "analyzer bindings schema differs")
    exact(bindings["expected_fit_seal_sha256"], seal_sha, "analyzer external seal binding")
    exact(bindings["fit_seal_sha256"], sha256(root / "fit_seal.json"), "analyzer fit seal binding")
    exact(bindings["task_summary_sha256"], task_summary_sha, "analyzer task summary bindings")
    exact(bindings["analyzer_sha256"], EXPECTED_ANALYZER_SHA256, "analyzer SHA binding")
    actual_rows = summary.get("rows")
    require(isinstance(actual_rows, list) and len(actual_rows) == 6, "analyzer row count differs")
    expected_by_key = {(row["task"], row["variant"]): row for row in expected_rows}
    actual_by_key = {(row.get("task"), row.get("variant")): row for row in actual_rows if isinstance(row, dict)}
    require(len(expected_by_key) == len(expected_rows) == len(actual_by_key) == 6 and set(actual_by_key) == set(expected_by_key),
            "analyzer row identity differs")
    for key, expected in expected_by_key.items():
        exact(actual_by_key[key], expected, f"analyzer scalar row {key}")

    csv_path = analysis_dir / "comparison.csv"
    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            csv_rows = list(reader)
    except OSError as error:
        raise AuditError("cannot read analyzer comparison CSV") from error
    expected_fields = sorted({field for row in expected_rows for field in row})
    require(header == expected_fields and len(csv_rows) == 6, "comparison CSV schema or row count differs")
    csv_by_key = {(row.get("task"), row.get("variant")): row for row in csv_rows}
    require(len(csv_by_key) == 6 and set(csv_by_key) == set(expected_by_key), "comparison CSV identities differ")
    for key, expected in expected_by_key.items():
        csv_row = csv_by_key[key]
        require(set(csv_row) == set(expected_fields), f"comparison CSV fields differ: {key}")
        for field, value in expected.items():
            encoded = "" if value is None else str(value)
            exact(csv_row[field], encoded, f"comparison CSV scalar {key}/{field}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-fit-seal-sha256", required=True)
    parser.add_argument("--fit-audit", type=Path, required=True)
    parser.add_argument("--expected-fit-audit-sha256", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="optional approved review-directory JSON output")
    args = parser.parse_args(argv)
    try:
        root, config_path = args.root.resolve(), args.config.resolve()
        require(root.is_dir() and config_path.is_file(), "root or config is missing")
        config = validate_config(root, config_path)
        seal, _audit = verify_fit_bindings(root, config, config_path, args.expected_fit_seal_sha256,
                                           args.fit_audit.resolve(), args.expected_fit_audit_sha256)
        all_rows: list[dict[str, Any]] = []
        fold_failures: list[dict[str, Any]] = []
        task_summary_sha: dict[str, str] = {}
        for task in sorted(TASK_DIMS):
            rows, failures = verify_task(root, task, config, seal)
            all_rows.extend(rows); fold_failures.extend(failures)
            task_summary_sha[task] = sha256(root / task / "summary.json")
        verify_analysis(args.analysis_dir.resolve(), root, args.expected_fit_seal_sha256, task_summary_sha, all_rows)
        report = {
            "schema": "unit_session_score_audit_v1", "status": "PASS", "root": str(root),
            "config_sha256": sha256(config_path), "fit_seal_sha256": args.expected_fit_seal_sha256,
            "fit_audit_sha256": args.expected_fit_audit_sha256, "analyzer_sha256": EXPECTED_ANALYZER_SHA256,
            "task_summary_sha256": task_summary_sha, "rows": all_rows,
            "folded_deployment_ineligible": fold_failures,
            "folded_deployment_ineligible_count": len(fold_failures),
            "arrays_in_report": False,
        }
        if args.report is not None:
            output = args.report.resolve()
            require(output.parent == Path(__file__).resolve().parent, "report must remain in this review directory")
            output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except AuditError as error:
        print(json.dumps({"schema": "unit_session_score_audit_v1", "status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
