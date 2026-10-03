"""Fit the declared legal M2 M33 development candidates, then score local held-in only.

No fit sees local query labels: all four public-bank fits complete before this
script loads the full held-in recording or opens the historical cohort archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COHORT_N = 14115
sys.path.insert(0, str(ROOT))
from scripts.run_cross_session_iteration import _evaluation_is_current, _select
from scripts.run_session_pretraining_iteration import source_is_current, source_path, verify_live
from ssm_decode import debug_experiment as d
from ssm_decode.data import load_recording
from ssm_decode.official_calibration import load_public_calibration
from ssm_decode.cross_session_iteration import _runtime_identity
from ssm_decode.official_payload import CODE, KEYS, fit_bank, reader_identity, sha


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, value):
    if path.exists() and read(path) != value:
        raise RuntimeError(f"frozen probe contract differs: {path}")
    write(path, value)


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    return float(1 - np.square(truth - prediction).sum() / max(np.square(truth - truth.mean(0, keepdims=True)).sum(), 1e-12))


def canonical_allvalid_indices(record, query_cutoff_trials: int) -> np.ndarray:
    usable = [(a, b) for a, b in record.trial_bounds if b - a >= 50]
    if len(usable) <= query_cutoff_trials:
        raise RuntimeError("frozen query cutoff leaves no usable trial")
    pieces = [np.arange(a, b, dtype=np.int64) for a, b in usable[query_cutoff_trials:]]
    return np.concatenate(pieces)[np.asarray(record.eval_mask, dtype=bool)[np.concatenate(pieces)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", required=True, type=Path)
    parser.add_argument("--stage-b-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    stage_a, stage_b, output = (value.resolve() for value in (args.stage_a_root, args.stage_b_root, args.output_root))
    if (output.exists() or output == stage_a or output == stage_b or stage_a in output.parents or stage_b in output.parents
            or output in stage_a.parents or output in stage_b.parents):
        raise RuntimeError("probe output must be a new root outside formal Stage A/B roots")
    config_path = ROOT / "configs/official_m2_m33_probe_20261003.json"
    cfg, bcfg = read(config_path), read(stage_b / "matrix.json")
    frozen_b = read(stage_b / "provenance.json")
    verify_live(bcfg, frozen_b, stage_a)
    authenticated_selection = read(stage_a / "selections.json")
    if authenticated_selection != _select(read(stage_a / "matrix.json"), stage_a, read(stage_a / "provenance.json"), persist=False):
        raise RuntimeError("Stage A prefix selection does not authenticate")
    selected = authenticated_selection["tasks"]["m2"]["methods"]
    if selected["none"]["selected"]["lr"] != 0.0 or selected["lora"]["selected"]["lr"] != 0.003:
        raise RuntimeError("authenticated Stage A M2 LR selection drift")
    expected_methods = [{"method": "none", "seed": 0, "lr": selected["none"]["selected"]["lr"]}]
    expected_methods += [{"method": "lora", "seed": seed, "lr": selected["lora"]["selected"]["lr"]} for seed in (0, 1, 2)]
    if cfg["methods"] != expected_methods:
        raise RuntimeError("declared M2 probe methods/LRs differ from authenticated selection")
    capacity = next(row for row in bcfg["capacities"] if row["name"] == cfg["source_capacity"])
    source_pathname = source_path(stage_b, "m2", capacity)
    if not source_is_current(source_pathname, bcfg, frozen_b, "m2", capacity):
        raise RuntimeError("Stage B wide M2 source is stale or incomplete")
    source_file, source_norm = source_pathname / "best.pt", source_pathname / "normalizer.npz"
    source = torch.load(source_file, map_location="cpu", weights_only=False)
    with np.load(source_norm) as normalizer:
        source_stats = tuple(normalizer[name].astype(np.float32) for name in KEYS)
    target_session = read(stage_a / "m2/none/seed0/lr0/manifest.json")["target_session"]
    data_root = Path(read(stage_a / "matrix.json")["data_root"])
    calibration = load_public_calibration("m2", "held_in", target_session, data_root)
    if len(calibration.receipt.get("realtrialbounds", [])) != cfg["public_calibration_trials"] or calibration.receipt.get("query_labels_used") is not False:
        raise RuntimeError("M2 calibration is not exactly the legal first 33 raw trials")
    formal_fit = stage_b / "adapt/wide/m2/m2/none/seed0/lr0"
    formal_eval = formal_fit / "query_evaluation"
    formal_metrics, formal_archive = formal_eval / "metrics.json", formal_eval / "predictions.npz"
    # Existing launcher validator authenticates the historical comparison artifact.
    wide_matrix, wide_provenance = read(stage_b / "adapt/wide/m2/matrix.json"), read(stage_b / "adapt/wide/m2/provenance.json")
    wide_selection = read(stage_b / "adapt/wide/m2/selections.json")
    if wide_selection != _select(wide_matrix, stage_b / "adapt/wide/m2", wide_provenance, persist=False):
        raise RuntimeError("Stage B wide M2 selection does not authenticate")
    selected_none = wide_selection["tasks"]["m2"]["methods"]["none"]["selected"]
    selected_none_fits = selected_none.get("fits", [])
    if not isinstance(selected_none_fits, list) or len(selected_none_fits) != 1:
        raise RuntimeError("Stage B wide M2 none selection must contain exactly one fit")
    selected_none_fit = selected_none_fits[0]
    if not _evaluation_is_current(formal_fit, formal_eval, wide_matrix, wide_provenance, "m2", selected_none_fit):
        raise RuntimeError("Stage B wide M2 none evaluation is stale")
    metrics = read(formal_metrics)
    if metrics.get("target_data_sha256") != calibration.receipt["rawfile_sha256"]:
        raise RuntimeError("formal M2 target hash differs from calibration raw NWB")
    formal_binding = dict(metrics_sha256=digest(formal_metrics), archive_sha256=digest(formal_archive),
        target_data_sha256=metrics["target_data_sha256"], cohort_index_hash=metrics["cohort_index_hashes"]["all_valid_indices"],
        cohort_truth_hash=metrics["cohort_truth_hashes"]["all_valid_truth_physical"], query_cutoff_trials=metrics["query_cutoff_trials"])
    frozen = dict(schema="official_m2_m33_probe_contract_v1", config=cfg, config_sha256=digest(config_path),
        launcher_sha256=digest(Path(__file__)), stage_a_provenance_sha256=digest(stage_a / "provenance.json"),
        stage_b_provenance_sha256=digest(stage_b / "provenance.json"), source_checkpoint_sha256=digest(source_file),
        source_normalizer_sha256=digest(source_norm), source_manifest_sha256=digest(source_pathname / "manifest.json"),
        target_session=target_session, public_calibration_receipt=calibration.receipt,
        stage_a_authenticated_selection=authenticated_selection, runtime_identity=_runtime_identity(),
        reader_identity=reader_identity(), code_sha256={name: sha(ROOT / "ssm_decode" / name) for name in CODE},
        formal_binding=formal_binding, query_used_for_checkpoint_or_lr_selection=False, calibration_raw_trials=33, calibration_filtering="none")
    freeze(output / "probe_contract.json", frozen)
    (output / "code_snapshot").mkdir(exist_ok=True)
    shutil.copyfile(__file__, output / "code_snapshot" / Path(__file__).name)
    def verify_frozen():
        checks = dict(config_sha256=digest(config_path), launcher_sha256=digest(Path(__file__)),
            stage_a_provenance_sha256=digest(stage_a / "provenance.json"), stage_b_provenance_sha256=digest(stage_b / "provenance.json"),
            source_checkpoint_sha256=digest(source_file), source_normalizer_sha256=digest(source_norm),
            source_manifest_sha256=digest(source_pathname / "manifest.json"), runtime_identity=_runtime_identity(),
            reader_identity=reader_identity(), code_sha256={name: sha(ROOT / "ssm_decode" / name) for name in CODE})
        if any(frozen[key] != value for key, value in checks.items()):
            raise RuntimeError("frozen probe input/code/runtime changed")
        if digest(Path(calibration.path)) != calibration.receipt["rawfile_sha256"]:
            raise RuntimeError("public calibration raw file changed")
        verify_live(bcfg, frozen_b, stage_a)
        if not source_is_current(source_pathname, bcfg, frozen_b, "m2", capacity):
            raise RuntimeError("Stage B wide M2 source changed")
        if digest(formal_metrics) != formal_binding["metrics_sha256"] or digest(formal_archive) != formal_binding["archive_sha256"]:
            raise RuntimeError("formal M2 metrics/archive changed")

    # Fitting is deliberately complete before local held-in query labels are opened.
    receipts = []
    for row in cfg["methods"]:
        verify_frozen()
        fit_dir = output / "fits" / f"{row['method']}_seed{row['seed']}"
        fit_args = SimpleNamespace(device="cuda:0", seed=row["seed"], method=row["method"], lr=row["lr"],
                                   steps=cfg["steps"], max_steps=cfg["max_steps"])
        receipt = fit_bank(calibration, source, source_stats, fit_args, fit_dir, frozen)
        receipts.append({"method": row["method"], "seed": row["seed"], "fit": str(fit_dir),
                         "model_sha256": digest(fit_dir / "model.pt"), "normalizer_sha256": digest(fit_dir / "normalizer.npz"), "receipt_sha256": digest(fit_dir / "receipt.json"),
                         "best_step": receipt["best_step"], "fold_r2_abs_delta": receipt["fold_r2_abs_delta"]})
    verify_frozen()
    # This is the first local query-label read, after every model and LR is frozen.
    record = load_recording("m2", "held_in", target_session, root=data_root)
    if Path(record.path) != Path(calibration.path) or digest(Path(record.path)) != formal_binding["target_data_sha256"]:
        raise RuntimeError("full record raw target binding differs")
    archive_path = formal_archive
    with np.load(archive_path) as archive:
        archive_indices = archive["allvalid_indices"]
        if archive_indices.ndim != 1 or archive_indices.dtype.kind not in "iu":
            raise RuntimeError("local held-in cohort indices must be a 1-D integer array")
        indices = archive_indices.astype(np.int64, copy=False)
        archived_truth = archive["truth_allvalid"].astype(np.float32)
        archived_zero = archive["zero_allvalid"].astype(np.float32)
    if (len(indices) != COHORT_N or not np.all(indices[1:] > indices[:-1]) or len(np.unique(indices)) != len(indices)
            or indices.min() < 0 or indices.max() >= len(record.neural) or not np.array_equal(archived_truth, record.behavior[indices])
            or hashlib.sha256(indices.tobytes()).hexdigest() != formal_binding["cohort_index_hash"]
            or hashlib.sha256(archived_truth.tobytes()).hexdigest() != formal_binding["cohort_truth_hash"]
            or not np.array_equal(indices, canonical_allvalid_indices(record, formal_binding["query_cutoff_trials"]))):
        raise RuntimeError("local held-in cohort/truth archive mismatch")
    rows = []
    for fit in receipts:
        payload = torch.load(Path(fit["fit"]) / "model.pt", map_location="cpu", weights_only=False)
        model = d._model(SimpleNamespace(**source["args"]), record.neural.shape[1], record.behavior.shape[1]).to("cuda:0").eval()
        model.load_state_dict(payload["state_dict"], strict=True)
        with np.load(Path(fit["fit"]) / "normalizer.npz") as normalizer:
            stats = tuple(normalizer[name].astype(np.float32) for name in KEYS)
        x = ((record.neural - stats[0]) / stats[1]).astype(np.float32)
        normalized = d._predict_endpoints(model, x, [(0, len(x))], indices, "cuda:0", 128, batch=256)
        prediction = normalized * stats[3] + stats[2]
        save_path = output / "scores" / f"{fit['method']}_seed{fit['seed']}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not np.isfinite(archived_truth).all() or not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite local held-in truth or prediction")
        physical = prediction.astype(np.float32)
        np.savez(save_path, allvalid_indices=indices, truth_allvalid=archived_truth, zero_allvalid=physical,
                 prediction_allvalid=physical, historical_b_zero_allvalid=archived_zero)
        rows.append(dict(**fit, score_npz_sha256=digest(save_path), allvalid_count=len(indices), r2=r2(archived_truth, physical),
                         cohort_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), prediction_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                         runtime_identity=_runtime_identity()))
    none = next(row["r2"] for row in rows if row["method"] == "none")
    lora = np.asarray([row["r2"] for row in rows if row["method"] == "lora"], dtype=np.float64)
    verify_frozen()
    for fit in receipts:
        if digest(Path(fit["fit"]) / "model.pt") != fit["model_sha256"] or digest(Path(fit["fit"]) / "normalizer.npz") != fit["normalizer_sha256"]:
            raise RuntimeError("merged probe model or normalizer changed after scoring")
    write(output / "summary.json", dict(schema="official_m2_m33_local_heldin_probe_v1", rows=rows,
        lora_mean=float(lora.mean()), lora_population_sd=float(lora.std(ddof=0)), lora_seed_gains_vs_none=(lora-none).tolist(),
        truth_archive_sha256=digest(archive_path), raw_truth_sha256=hashlib.sha256(record.behavior[indices].astype(np.float32).tobytes()).hexdigest(),
        query_used_for_checkpoint_or_lr_selection=False, all_fits_completed_before_query_decode=True, runtime_identity=_runtime_identity()))


if __name__ == "__main__":
    main()
