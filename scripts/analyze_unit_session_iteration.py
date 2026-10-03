"""Read-only verification and scalar aggregation for a unit/session iteration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def sha(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def physical_r2(truth, prediction) -> float:
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    if truth.shape != prediction.shape or truth.ndim < 1 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("saved truth and prediction must have equal finite shapes")
    return float(1 - np.square(truth - prediction).sum() /
                 max(float(np.square(truth - truth.mean(axis=0)).sum()), 1e-12))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-12))


def is_digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_seal(root: Path, cfg: dict, expected_sha: str) -> dict:
    path = root / "fit_seal.json"
    require(is_digest(expected_sha), "external fit-seal SHA must be lowercase SHA-256")
    require(sha(path) == expected_sha, "external fit-seal SHA differs")
    seal = read(path)
    tasks = set(cfg["tasks"])
    require(set(seal) == {"schema", "tasks"} and seal["schema"] == "unit_session_fit_seal_v1", "invalid fit-seal schema")
    require(set(seal["tasks"]) == tasks, "fit-seal task set differs")
    for task, binding in seal["tasks"].items():
        require(set(binding) == {"completion_sha256", "contract_sha256"}, f"invalid fit-seal binding: {task}")
        require(all(is_digest(value) for value in binding.values()),
                f"invalid fit-seal digest: {task}")
    return seal


def verify_completion_artifacts(task_root: Path, completion: dict):
    artifacts = completion.get("artifacts")
    require(isinstance(artifacts, dict) and artifacts, f"completion artifacts missing: {task_root}")
    root = task_root.resolve()
    for relative, digest in artifacts.items():
        candidate = Path(relative)
        require(isinstance(relative, str) and not candidate.is_absolute() and ".." not in candidate.parts,
                f"completion artifact path escapes task root: {relative}")
        path = (root / candidate).resolve()
        require(path.is_relative_to(root), f"completion artifact path escapes task root: {relative}")
        require(is_digest(digest) and sha(path) == digest,
                f"completion artifact SHA differs: {relative}")


def load_npz(path: Path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    except Exception as exc:
        raise ValueError(f"invalid NPZ: {path}") from exc


def effective_directional_ratio(path: Path, expected_sha: str) -> float:
    require(sha(path) == expected_sha, f"source effective-map SHA differs: {path}")
    arrays = load_npz(path)
    require(set(arrays) == {"input_weight", "input_bias", "output_weight", "output_bias"},
            f"unexpected effective-map arrays: {path}")
    a = np.asarray(arrays["input_weight"], np.float64)
    require(a.ndim == 3 and all(a.shape), f"invalid effective input map shape: {path}")
    require(np.isfinite(a).all(), f"nonfinite effective input map: {path}")
    mean = a.mean(axis=0)
    numerator = 0.0
    denominator = float(np.square(a).sum())
    for channel in range(a.shape[2]):
        direction = mean[:, channel]
        norm = float(direction @ direction)
        values = a[:, :, channel]
        projected = np.zeros_like(values) if norm <= 1e-30 else np.outer(values @ direction / norm, direction)
        numerator += float(np.square(values - projected).sum())
    return numerator / max(denominator, 1e-30)


def fit_name(row):
    return f"{row['method']}_seed{row['seed']}_lr{float(row['lr']):g}"


def expected_archive_path(task_root: Path, variant: str, row: dict) -> Path:
    return (task_root / "scores" / variant / (fit_name(row) + ".npz")).resolve()


def verify_scores_closure(task_root: Path, expected_archives: set[Path]):
    scores = task_root / "scores"
    found = {path.resolve() for path in scores.rglob("*") if path.is_file()} if scores.is_dir() else set()
    require(found == expected_archives, f"scores archive closure differs: {task_root.name}")


def selected_fit(root: Path, task: str, row: dict) -> tuple[Path, dict]:
    path = root / task / "adapt" / row["variant"] / fit_name(row)
    result = read(path / "fit_result.json")
    require(result == row, f"completion selected row is not the complete fit result: {path}")
    for key in ("task", "variant", "method", "seed", "lr"):
        require(result.get(key) == row.get(key), f"selected fit identity differs: {path}/{key}")
    require(result.get("best_checkpoint_sha256") == row.get("best_checkpoint_sha256"),
            f"selected checkpoint binding differs: {path}")
    require(sha(path / "best.pt") == row["best_checkpoint_sha256"], f"selected checkpoint SHA differs: {path}")
    require(sha(path / "normalizer.npz") == row["normalizer_sha256"], f"selected normalizer SHA differs: {path}")
    return path, result


def validate_fold_receipt(receipt: dict, label: str):
    require(isinstance(receipt, dict), f"missing fold receipt: {label}")
    accepted = receipt.get("fold_accepted")
    point = receipt.get("point_allclose_pass")
    delta = receipt.get("r2_delta")
    require(isinstance(accepted, bool) and isinstance(point, bool) and np.isfinite(delta), f"invalid fold receipt: {label}")
    require(accepted == (point and abs(float(delta)) <= .001), f"inconsistent fold receipt: {label}")


def validate_query_archive(path: Path, task_cfg: dict, metric: dict, label: str):
    data = load_npz(path)
    require(set(data) == {"indices", "truth", "prediction"}, f"unexpected query archive arrays: {path}")
    indices, truth, prediction = data["indices"], data["truth"], data["prediction"]
    require(indices.dtype == np.int64 and indices.ndim == 1 and len(indices) == task_cfg["expected_query_bins"],
            f"invalid query indices: {label}")
    require(truth.dtype == np.float32 and prediction.dtype == np.float32 and truth.ndim == 2 and prediction.shape == truth.shape
            and truth.shape[0] == len(indices) and truth.shape[1] > 0 and np.isfinite(truth).all() and np.isfinite(prediction).all(),
            f"invalid query truth or prediction: {label}")
    require(array_sha(indices) == task_cfg["expected_indices_sha256"] and array_sha(truth) == task_cfg["expected_truth_sha256"],
            f"query cohort SHA differs: {label}")
    require(metric.get("indices_sha256") == array_sha(indices) and metric.get("truth_sha256") == array_sha(truth),
            f"summary array SHA differs: {label}")
    return data


def source_summary(root: Path, task: str, variant: str) -> dict:
    path = root / task / "source" / variant
    result = read(path / "fit_result.json")
    identity = result.get("variant")
    identity = identity.get("name") if isinstance(identity, dict) else identity
    require(result.get("task") == task and identity == variant,
            f"source identity differs: {path}")
    maps = path / "effective_maps.npz"
    ratio = effective_directional_ratio(maps, result.get("effective_maps_sha256"))
    scores = result.get("selected_full_validation_r2_by_session")
    require(isinstance(scores, dict) and scores, f"missing source full-session scores: {path}")
    values = np.asarray(list(scores.values()), np.float64)
    require(np.isfinite(values).all(), f"nonfinite source full-session score: {path}")
    folds = result.get("source_fold_replays")
    require(isinstance(folds, dict) and folds, f"missing source fold receipts: {path}")
    for session, receipt in folds.items():
        validate_fold_receipt(receipt, f"{path}/source/{session}")
    failed_folds = sum(receipt["fold_accepted"] is not True for receipt in folds.values())
    return dict(path=path, macro=float(values.mean()), selection_macro=result.get("best_val_r2"), full_scores=scores, directional_ratio=ratio,
                trainable_parameters=result.get("trainable_parameters"), fit_seconds=result.get("fit_seconds"),
                budget=result.get("final_budget"), source_seed=0, fold_failed_count=failed_folds)


def verify_task(root: Path, task: str, cfg: dict, seal: dict) -> list[dict]:
    task_root = root / task
    completion = read(task_root / "fits_complete.json")
    contract = task_root / "contract.json"
    binding = seal.get("tasks", {}).get(task)
    require(isinstance(binding, dict), f"fit seal lacks task: {task}")
    require(sha(task_root / "fits_complete.json") == binding.get("completion_sha256"), f"completion SHA differs: {task}")
    require(sha(contract) == binding.get("contract_sha256"), f"contract SHA differs: {task}")
    contract_value = read(contract)
    require(contract_value.get("cfg") == cfg and contract_value.get("config_sha256") == read(root / "matrix.json").get("config_sha256"),
            f"contract configuration binding differs: {task}")
    require(completion.get("all_fits_completed_before_query") is True and completion.get("query_metrics_read") is False,
            f"fit completion is not query-clean: {task}")
    verify_completion_artifacts(task_root, completion)
    summary = read(task_root / "summary.json")
    require(set(summary) == {"schema", "task", "rows", "all_fits_completed_before_query", "query_used_for_checkpoint_or_lr_selection", "architecture_comparison_uses_local_query", "official_heldout_result"},
            f"invalid query summary schema: {task}")
    require(summary["schema"] == "unit_session_local_results_v1" and summary["task"] == task
            and summary["all_fits_completed_before_query"] is True and summary["query_used_for_checkpoint_or_lr_selection"] is False
            and summary["architecture_comparison_uses_local_query"] is True and summary["official_heldout_result"] is False,
            f"invalid query summary flags: {task}")
    variants = {v["name"]: v for v in cfg["variants"]}
    selected = completion.get("selected")
    require(isinstance(selected, list), f"selected fits missing: {task}")
    rows_by_variant = {name: [] for name in variants}
    for row in selected:
        require(row.get("variant") in variants, f"unknown selected variant: {task}")
        rows_by_variant[row["variant"]].append(row)
    summary_rows = summary.get("rows")
    require(isinstance(summary_rows, list), f"query rows missing: {task}")
    query_by_fit = {}
    row_keys = {"task", "variant", "method", "seed", "lr", "r2", "query_bins", "best_step", "best_prefix_val_r2",
                "trainable_parameters", "query_seconds", "prediction_archive", "prediction_archive_sha256", "best_checkpoint_sha256",
                "indices_sha256", "truth_sha256", "folded_prefix_r2_delta"}
    for row in summary_rows:
        require(set(row) == row_keys, f"invalid query row schema: {task}")
        key = (row.get("variant"), row.get("method"), row.get("seed"), float(row.get("lr")))
        require(key not in query_by_fit, f"duplicate query row: {task}/{key}")
        query_by_fit[key] = row
    output = []
    expected_seeds = set(cfg["adapt_seeds"])
    canonical_indices = canonical_truth = None
    expected_archives = set()
    for variant, picked in rows_by_variant.items():
        decision = completion.get("decisions", {}).get(variant, {})
        none = [r for r in picked if r.get("method") == "none"]
        ui = [r for r in picked if r.get("method") == "ui"]
        require(len(none) == 1 and none[0].get("seed") == 0 and float(none[0].get("lr")) == 0.,
                f"expected one none seed0 fit: {task}/{variant}")
        excluded = decision.get("excluded") is True
        if excluded:
            require(not ui, f"excluded UI family is selected: {task}/{variant}")
        else:
            require(len(ui) == len(expected_seeds) and {r.get("seed") for r in ui} == expected_seeds,
                    f"UI seed family differs: {task}/{variant}")
            require(len({float(r["lr"]) for r in ui}) == 1 and float(ui[0]["lr"]) == float(decision.get("selected_lr")),
                    f"selected UI LR differs: {task}/{variant}")
        candidate_results = []
        for lr in cfg["adapt_lrs"]:
            family = []
            for seed in sorted(expected_seeds):
                candidate_path = root / task / "adapt" / variant / f"ui_seed{seed}_lr{float(lr):g}"
                candidate = read(candidate_path / "fit_result.json")
                require((candidate.get("task"), candidate.get("variant"), candidate.get("method"),
                         candidate.get("seed"), float(candidate.get("lr"))) == (task, variant, "ui", seed, float(lr)),
                        f"UI candidate identity differs: {candidate_path}")
                family.append(candidate)
            require({row["seed"] for row in family} == expected_seeds, f"UI candidate seeds differ: {task}/{variant}/{lr}")
            candidate_results.append((float(lr), family))
        rebuilt_candidates = []
        for lr, family in candidate_results:
            admissible = all(candidate.get("converged") is True for candidate in family)
            mean = float(np.mean([candidate["best_val_r2"] for candidate in family], dtype=np.float64))
            rebuilt_candidates.append(dict(lr=lr, admissible=admissible, mean_prefix_val_r2=mean))
        saved_candidates = decision.get("candidates")
        require(isinstance(saved_candidates, list) and len(saved_candidates) == len(rebuilt_candidates),
                f"decision candidate count differs: {task}/{variant}")
        saved_by_lr = {float(candidate.get("lr")): candidate for candidate in saved_candidates}
        require(set(saved_by_lr) == {candidate["lr"] for candidate in rebuilt_candidates},
                f"decision candidate LR set differs: {task}/{variant}")
        for candidate in rebuilt_candidates:
            saved = saved_by_lr[candidate["lr"]]
            require(saved.get("admissible") is candidate["admissible"] and close(saved.get("mean_prefix_val_r2"), candidate["mean_prefix_val_r2"]),
                    f"decision candidate differs: {task}/{variant}/{candidate['lr']}")
        admissible = [candidate for candidate in rebuilt_candidates if candidate["admissible"]]
        recomputed = max(admissible, key=lambda candidate: (candidate["mean_prefix_val_r2"], -candidate["lr"])) if admissible else None
        require(excluded == (recomputed is None), f"decision exclusion differs: {task}/{variant}")
        require(decision.get("selected_lr") == (None if recomputed is None else recomputed["lr"]),
                f"decision selected LR differs: {task}/{variant}")
        query_scores = []
        all_selected = none + ui
        for selected_row in all_selected:
            path, fit = selected_fit(root, task, selected_row)
            key = (variant, selected_row["method"], selected_row["seed"], float(selected_row["lr"]))
            metric = query_by_fit.pop(key, None)
            require(metric is not None, f"missing query metric: {task}/{key}")
            require(metric["task"] == task and metric["query_bins"] == cfg["tasks"][task]["expected_query_bins"], f"query row task/count differs: {task}/{key}")
            for name in ("variant", "method", "seed", "lr", "best_step", "trainable_parameters", "best_checkpoint_sha256"):
                require(metric[name] == selected_row[name], f"query row differs from selected fit: {task}/{key}/{name}")
            require(close(metric["best_prefix_val_r2"], selected_row["best_val_r2"]) and close(metric["folded_prefix_r2_delta"], selected_row["fold_prefix_r2_delta"])
                    and np.isfinite(metric["query_seconds"]) and float(metric["query_seconds"]) >= 0, f"query scalar differs: {task}/{key}")
            expected = expected_archive_path(task_root, variant, selected_row)
            archive = Path(metric["prediction_archive"]).resolve()
            require(archive == expected and archive.is_relative_to((task_root / "scores").resolve()), f"query archive path differs: {task}/{key}")
            require(is_digest(metric["prediction_archive_sha256"]) and sha(archive) == metric["prediction_archive_sha256"], f"query archive SHA differs: {task}/{key}")
            expected_archives.add(expected)
            data = validate_query_archive(archive, cfg["tasks"][task], metric, f"{task}/{key}")
            score = physical_r2(data["truth"], data["prediction"])
            require(abs(score - float(metric.get("r2"))) <= 1e-12, f"saved R2 differs: {task}/{key}")
            if canonical_indices is None:
                canonical_indices, canonical_truth = data["indices"], data["truth"]
            else:
                require(np.array_equal(data["indices"], canonical_indices) and np.array_equal(data["truth"], canonical_truth),
                        f"query cohort differs across variants: {task}")
            query_scores.append((selected_row, fit, score, data["indices"], data["truth"]))
        require(not query_by_fit or all(key[0] != variant for key in query_by_fit), f"unselected query metric: {task}/{variant}")
        baseline = next(score for row, _, score, _, _ in query_scores if row["method"] == "none")
        ui_scores = [score for row, _, score, _, _ in query_scores if row["method"] == "ui"]
        source = source_summary(root, task, variant)
        rejected = [x for x in decision.get("candidates", []) if not x.get("admissible", False)]
        late = sum(1 for _, family in candidate_results for candidate in family if candidate.get("converged") is False)
        selected_ui = [fit for row, fit, _, _, _ in query_scores if row["method"] == "ui"]
        accepted_target_folds = sum(fit.get("deployment_fold_accepted") is True for _, fit, _, _, _ in query_scores)
        for _, fit, _, _, _ in query_scores:
            validate_fold_receipt(fit.get("fold_replay"), f"{task}/{variant}/{fit_name(fit)}")
            require(fit["deployment_fold_accepted"] == fit["fold_replay"]["fold_accepted"]
                    and close(fit["fold_prefix_r2_delta"], fit["fold_replay"]["r2_delta"]), f"target fold fields differ: {task}/{variant}")
        output.append(dict(
            task=task, variant=variant, none_r2=baseline,
            selected_ui_r2_mean=None if not ui_scores else float(np.mean(ui_scores, dtype=np.float64)),
            selected_ui_r2_population_sd=None if not ui_scores else float(np.std(ui_scores, dtype=np.float64)),
            selected_ui_lr=None if not ui else float(ui[0]["lr"]), ui_family_excluded=excluded,
            lr_rejections=len(rejected), family_late_count=late,
            source_seed=source["source_seed"], target_seeds="" if not ui else ",".join(map(str, sorted(expected_seeds))),
            source_macro_full_validation_r2=source["macro"], source_full_session_scores=json.dumps(source["full_scores"], sort_keys=True),
            source_selection_macro_validation_r2=source["selection_macro"],
            source_directional_orthogonal_energy_ratio=source["directional_ratio"],
            source_directional_reference="affine-only expected near zero; no biological alignment" if variant == "affine" else "unit-direction residual enabled",
            source_trainable_parameters=source["trainable_parameters"], target_trainable_parameters=None if not selected_ui else selected_ui[0].get("trainable_parameters"),
            normalizer_sha256=none[0]["normalizer_sha256"], source_fold_failed_count=source["fold_failed_count"],
            target_selected_fold_accepted_count=accepted_target_folds, target_selected_fold_count=len(query_scores),
            source_budget=source["budget"], target_budgets=json.dumps([fit.get("final_budget") for _, fit, _, _, _ in query_scores]),
            source_fit_seconds=source["fit_seconds"], target_fit_seconds=float(sum(float(fit.get("fit_seconds", 0.)) for _, fit, _, _, _ in query_scores)),
        ))
    require(not query_by_fit, f"unknown query metrics remain: {task}")
    verify_scores_closure(task_root, expected_archives)
    return output


def write_csv(path: Path, rows: list[dict]):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def plot(rows, path: Path):
    tasks = sorted({row["task"] for row in rows})
    figure, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        values = [row for row in rows if row["task"] == task]
        names = [row["variant"] for row in values]
        means = [row["selected_ui_r2_mean"] for row in values]
        errors = [row["selected_ui_r2_population_sd"] for row in values]
        positions = np.arange(len(values))
        finite = np.array([value is not None for value in means])
        axis.bar(positions[finite], np.asarray(means, float)[finite], yerr=np.asarray(errors, float)[finite], capsize=4, label="selected UI: 3-seed mean ± population SD")
        axis.scatter(positions, [row["none_r2"] for row in values], color="black", zorder=3, label="none")
        axis.set_xticks(positions, names, rotation=20); axis.set_title(task.upper()); axis.set_ylabel("physical float64 R²")
        axis.axhline(0, color="0.7", linewidth=.8); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/unit_session_iteration_20261004"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis/unit_session_iteration_20261004"))
    parser.add_argument("--expected-fit-seal-sha256", required=True,
                        help="externally recorded SHA-256 of root/fit_seal.json")
    args = parser.parse_args(argv)
    root = args.root.resolve(); output = args.output.resolve()
    matrix = read(root / "matrix.json")
    cfg = matrix.get("config")
    require(isinstance(cfg, dict) and set(cfg.get("tasks", {})) == {"m1", "m2"}, "matrix has no complete M1/M2 config")
    require(is_digest(matrix.get("config_sha256")),
            "matrix configuration digest is invalid")
    seal = verify_seal(root, cfg, args.expected_fit_seal_sha256)
    # Both score summaries must exist before any aggregation or output write.
    require(all((root / task / "summary.json").is_file() for task in cfg["tasks"]), "both task summaries must complete before analysis")
    rows = [row for task in sorted(cfg["tasks"]) for row in verify_task(root, task, cfg, seal)]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", rows)
    summary = dict(schema="unit_session_iteration_analysis_v1", status="verified", local_development_only=True,
                   official_result=False, query_used_for_checkpoint_lr_or_seed_selection=False,
                   query_used_for_local_architecture_exploration_comparison=True,
                   source_seed=cfg["source_seed"], target_seeds=cfg["adapt_seeds"], rows=rows,
                   bindings=dict(expected_fit_seal_sha256=args.expected_fit_seal_sha256,
                                 fit_seal_sha256=sha(root / "fit_seal.json"),
                                 task_summary_sha256={task: sha(root / task / "summary.json") for task in sorted(cfg["tasks"])},
                                 analyzer_sha256=sha(Path(__file__).resolve())),
                   directional_diagnostic="effective input A: per-column projection onto source mean; orthogonal-energy / total-A-energy; affine-only is expected near zero and does not imply biological alignment")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(rows, output / "comparison.png")


if __name__ == "__main__":
    main()
