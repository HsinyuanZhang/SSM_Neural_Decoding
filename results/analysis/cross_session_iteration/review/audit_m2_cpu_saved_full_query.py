"""Compare one caller-selected merged M2 model with its saved GPU output.

This program does not rank candidates and does not start CUDA work. It checks
the complete frozen all-valid query cohort from the legal M33 probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssm_decode.cross_session_iteration import KEYS, _runtime_identity
from ssm_decode.data import load_recording
from ssm_decode.mamba3_cpu import CPUDecoder


SCORE_DELTA_LIMIT = 1e-3
SAVED_SCORE_ORACLE_LIMIT = 2e-6
EXPECTED_QUERY_BINS = 14_115


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def physical_r2(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    sse = float(np.square(truth64 - prediction64).sum())
    mean = truth64.mean(axis=0, keepdims=True)
    sst = float(np.square(truth64 - mean).sum())
    if not np.isfinite(sse) or not np.isfinite(sst) or sst <= 0:
        raise FloatingPointError("The physical R-squared inputs are not valid")
    return float(1.0 - sse / sst), sse, sst


def make_windows(neural: np.ndarray, endpoints: np.ndarray, context: int) -> np.ndarray:
    windows = np.zeros((len(endpoints), context, neural.shape[1]), dtype=np.float32)
    for row, endpoint in enumerate(endpoints):
        left = max(0, int(endpoint) - context + 1)
        length = int(endpoint) - left + 1
        windows[row, context - length :] = neural[left : int(endpoint) + 1]
    return windows


def canonical_allvalid_indices(record, cutoff: int) -> np.ndarray:
    usable = [(a, b) for a, b in record.trial_bounds if b - a >= 50]
    if len(usable) <= cutoff:
        raise RuntimeError("The frozen query cutoff leaves no usable trial")
    full = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in usable[cutoff:]])
    return full[np.asarray(record.eval_mask, dtype=bool)[full]]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--stage-b-root", required=True, type=Path)
    parser.add_argument("--fit-name", default="lora_seed0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--cpu-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    if args.cpu_batch_size < 1:
        raise ValueError("CPU batch size must be positive")
    if args.fit_name != "lora_seed0":
        raise ValueError("This audit is restricted to the caller-selected lora_seed0 fit")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    probe_root = args.probe_root.resolve()
    stage_b_root = args.stage_b_root.resolve()
    fit_dir = probe_root / "fits" / args.fit_name
    checkpoint = fit_dir / "model.pt"
    unmerged_checkpoint = fit_dir / "unmerged_best.pt"
    normalizer_path = fit_dir / "normalizer.npz"
    receipt_path = fit_dir / "receipt.json"
    score_path = probe_root / "scores" / f"{args.fit_name}.npz"
    summary_path = probe_root / "summary.json"
    contract_path = probe_root / "probe_contract.json"
    source_checkpoint = stage_b_root / "source/m2/wide/best.pt"
    source_normalizer = stage_b_root / "source/m2/wide/normalizer.npz"
    source_manifest = stage_b_root / "source/m2/wide/manifest.json"
    required = (
        checkpoint,
        unmerged_checkpoint,
        normalizer_path,
        receipt_path,
        score_path,
        summary_path,
        contract_path,
        source_checkpoint,
        source_normalizer,
        source_manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    receipt = read_json(receipt_path)
    summary = read_json(summary_path)
    contract = read_json(contract_path)
    rows = [row for row in summary.get("rows", []) if row.get("method") == "lora" and row.get("seed") == 0]
    if len(rows) != 1:
        raise RuntimeError("The probe summary does not identify one lora seed 0 row")
    row = rows[0]
    if Path(row["fit"]).resolve() != fit_dir:
        raise RuntimeError("The selected summary row refers to another fit")
    expected_hashes = {
        checkpoint: row["model_sha256"],
        normalizer_path: row["normalizer_sha256"],
        receipt_path: row["receipt_sha256"],
        score_path: row["score_npz_sha256"],
        source_checkpoint: contract["source_checkpoint_sha256"],
        source_normalizer: contract["source_normalizer_sha256"],
        source_manifest: contract["source_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact hash changed: {path}")

    unmerged = torch.load(unmerged_checkpoint, map_location="cpu", weights_only=False)
    fit_args = unmerged.get("args", {})
    if fit_args.get("method") != "lora" or fit_args.get("seed") != 0 or fit_args.get("lr") != 0.003:
        raise RuntimeError("The selected fit metadata is not the fixed lora seed 0 candidate")
    if unmerged.get("best_step") != row.get("best_step") or receipt.get("best_step") != row.get("best_step"):
        raise RuntimeError("The selected best step is inconsistent")
    adaptation = receipt.get("adaptation", {})
    if adaptation.get("method") != "lora" or adaptation.get("trainable_count") != 37_000:
        raise RuntimeError("The selected receipt does not describe the expected LoRA fit")
    if not receipt.get("convergence_satisfied") or not receipt.get("fold_allclose"):
        raise RuntimeError("The selected fit did not pass convergence or folding")
    if receipt.get("query_labels_used") is not False or receipt.get("query_used_for_selection") is not False:
        raise RuntimeError("The selected fit does not deny query use")
    if receipt.get("calibration_raw_trials") != 33 or receipt.get("calibration_filtering") != "none":
        raise RuntimeError("The selected fit does not use the legal raw M33 budget")
    if summary.get("query_used_for_checkpoint_or_lr_selection") is not False:
        raise RuntimeError("The probe summary does not deny query selection")
    if summary.get("all_fits_completed_before_query_decode") is not True:
        raise RuntimeError("The probe did not complete all fits before query decode")
    if contract.get("query_used_for_checkpoint_or_lr_selection") is not False:
        raise RuntimeError("The frozen probe contract does not deny query selection")

    runtime = _runtime_identity()
    if contract.get("runtime_identity") != runtime or row.get("runtime_identity") != runtime:
        raise RuntimeError("The current runtime differs from the frozen GPU probe")
    if receipt.get("runtime_identity") != runtime:
        raise RuntimeError("The fit receipt runtime differs from the current runtime")

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_args = source.get("args", {})
    if source_args.get("task") != "m2" or source_args.get("context") != 128:
        raise RuntimeError("The frozen source is not the required M2 context-128 model")
    if source_args.get("width") != 256 or source_args.get("layers") != 4:
        raise RuntimeError("The frozen source is not the required wide model")
    if contract.get("target_session") != receipt.get("target_session"):
        raise RuntimeError("The target-session binding changed")
    record = load_recording(
        "m2",
        "held_in",
        contract["target_session"],
        root=Path(source_args["data_root"]),
    )
    if file_sha256(Path(record.path)) != contract["formal_binding"]["target_data_sha256"]:
        raise RuntimeError("The raw M2 target recording changed")
    cutoff = int(contract["formal_binding"]["query_cutoff_trials"])
    canonical_indices = canonical_allvalid_indices(record, cutoff)

    with np.load(score_path) as saved:
        saved_indices = saved["allvalid_indices"].astype(np.int64, copy=False)
        saved_truth = saved["truth_allvalid"].astype(np.float64, copy=False)
        saved_gpu = saved["prediction_allvalid"].astype(np.float64, copy=False)
        saved_zero = saved["zero_allvalid"].astype(np.float64, copy=False)
    cohort_exact = bool(np.array_equal(canonical_indices, saved_indices))
    truth = record.behavior[canonical_indices].astype(np.float64, copy=False)
    truth_exact = bool(np.array_equal(truth, saved_truth))
    prediction_fields_exact = bool(np.array_equal(saved_gpu, saved_zero))
    if not cohort_exact or not truth_exact or not prediction_fields_exact:
        raise RuntimeError("The saved GPU score archive differs from the canonical query")
    if len(canonical_indices) != EXPECTED_QUERY_BINS:
        raise RuntimeError("The M2 all-valid query cohort does not contain 14,115 bins")
    if array_sha256(saved_indices) != contract["formal_binding"]["cohort_index_hash"]:
        raise RuntimeError("The canonical cohort hash differs from the frozen formal binding")
    if array_sha256(saved_truth.astype(np.float32)) != contract["formal_binding"]["cohort_truth_hash"]:
        raise RuntimeError("The canonical truth hash differs from the frozen formal binding")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(payload) != {"state_dict"}:
        raise RuntimeError("The deployment checkpoint is not a plain merged state export")
    cpu = CPUDecoder.from_state_dict(payload["state_dict"])
    if cpu.input_size != record.neural.shape[1] or cpu.output_size != record.behavior.shape[1]:
        raise RuntimeError("The merged model geometry differs from the target recording")
    with np.load(normalizer_path) as archive:
        stats = tuple(archive[key].astype(np.float32, copy=False) for key in KEYS)
    if receipt.get("normalizer_sha256") != file_sha256(normalizer_path):
        raise RuntimeError("The fit receipt normalizer hash changed")
    neural = ((record.neural - stats[0]) / stats[1]).astype(np.float32)

    predictions = []
    first_inputs = None
    began = time.perf_counter()
    for start in range(0, len(canonical_indices), args.cpu_batch_size):
        selected = canonical_indices[start : start + args.cpu_batch_size]
        inputs = torch.from_numpy(make_windows(neural, selected, cpu.context))
        if first_inputs is None:
            first_inputs = inputs[:1].clone()
        output = cpu.forward(inputs)[:, -1]
        if not torch.isfinite(output).all():
            raise FloatingPointError("A CPU query prediction is not finite")
        predictions.append(output.numpy())
        done = min(start + args.cpu_batch_size, len(canonical_indices))
        if start == 0 or done == len(canonical_indices) or start % 800 == 0:
            print(f"CPU full-query endpoints {done}/{len(canonical_indices)}", flush=True)
    cpu_seconds = time.perf_counter() - began
    cpu_normalized = np.concatenate(predictions).astype(np.float32, copy=False)
    y_mean, y_std = stats[2].astype(np.float64), stats[3].astype(np.float64)
    cpu_physical = cpu_normalized.astype(np.float64) * y_std + y_mean

    saved_gpu_r2, saved_gpu_sse, sst = physical_r2(truth, saved_gpu)
    cpu_r2, cpu_sse, cpu_sst = physical_r2(truth, cpu_physical)
    if cpu_sst != sst:
        raise AssertionError("The GPU and CPU score denominators differ")
    saved_score_oracle_delta = abs(saved_gpu_r2 - float(row["r2"]))
    score_delta = cpu_r2 - saved_gpu_r2

    altered = first_inputs.clone()
    generator = torch.Generator().manual_seed(20261003)
    altered[:, 96:] += 3 * torch.randn(altered[:, 96:].shape, generator=generator)
    first_output = cpu.forward(first_inputs).numpy()
    altered_output = cpu.forward(altered).numpy()
    prefix_equal = bool(np.array_equal(first_output[:, :96], altered_output[:, :96]))
    prefix_max_delta = float(np.abs(first_output[:, :96] - altered_output[:, :96]).max())

    finite = bool(
        np.isfinite(truth).all()
        and np.isfinite(saved_gpu).all()
        and np.isfinite(cpu_physical).all()
        and np.isfinite(saved_gpu_r2)
        and np.isfinite(cpu_r2)
    )
    passed = bool(
        finite
        and cohort_exact
        and truth_exact
        and prediction_fields_exact
        and prefix_equal
        and saved_score_oracle_delta <= SAVED_SCORE_ORACLE_LIMIT
        and abs(score_delta) <= SCORE_DELTA_LIMIT
    )

    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.arrays,
        all_valid_query_indices=canonical_indices,
        truth_physical=truth,
        saved_gpu_prediction_physical=saved_gpu,
        cpu_prediction_physical=cpu_physical,
    )
    delta = cpu_physical - saved_gpu
    report = {
        "schema": "mamba3_cpu_saved_gpu_full_query_parity_v1",
        "status": "PASS" if passed else "FAIL",
        "scope": "full_query",
        "selection_boundary": "fit supplied by caller; this audit does not rank candidates",
        "reference": "saved official GPU prediction archive",
        "gpu_calls": 0,
        "task": "m2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "unmerged_checkpoint_sha256": file_sha256(unmerged_checkpoint),
        "receipt": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "probe_contract_sha256": file_sha256(contract_path),
        "probe_summary_sha256": file_sha256(summary_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_normalizer_sha256": file_sha256(source_normalizer),
        "target_normalizer_sha256": file_sha256(normalizer_path),
        "target_data_sha256": file_sha256(Path(record.path)),
        "saved_predictions": str(score_path),
        "saved_predictions_sha256": file_sha256(score_path),
        "method": "lora",
        "seed": 0,
        "lr": 0.003,
        "best_step": row["best_step"],
        "calibration_raw_trials": receipt["calibration_raw_trials"],
        "query_cutoff_trials": cutoff,
        "policy": "recording_causal_fixed_window",
        "target_session": record.session,
        "n_query_bounds": len([(a, b) for a, b in record.trial_bounds if b - a >= 50][cutoff:]),
        "n_all_valid_query_bins": int(len(canonical_indices)),
        "cohort_exact_to_saved_gpu": cohort_exact,
        "truth_exact_to_saved_gpu": truth_exact,
        "saved_prediction_fields_exact": prediction_fields_exact,
        "cohort_sha256": array_sha256(canonical_indices),
        "truth_sha256": array_sha256(saved_truth.astype(np.float32)),
        "saved_gpu_prediction_sha256": array_sha256(saved_gpu.astype(np.float32)),
        "cpu_prediction_sha256": array_sha256(cpu_physical),
        "saved_gpu_physical_r2": saved_gpu_r2,
        "cpu_physical_r2": cpu_r2,
        "score_delta": score_delta,
        "absolute_score_delta": abs(score_delta),
        "saved_score_oracle_delta": saved_score_oracle_delta,
        "saved_gpu_sse": saved_gpu_sse,
        "cpu_sse": cpu_sse,
        "sst": sst,
        "pointwise": {
            "mean_signed": float(delta.mean()),
            "mean_absolute": float(np.abs(delta).mean()),
            "rms": float(np.sqrt(np.square(delta).mean())),
            "max_absolute": float(np.abs(delta).max()),
        },
        "all_values_finite": finite,
        "causality": {
            "perturbed_input_start": 96,
            "checked_output_stop_exclusive": 96,
            "prefix_bitwise_equal": prefix_equal,
            "prefix_max_absolute_delta": prefix_max_delta,
        },
        "cpu": {
            "decoder_class": "ssm_decode.mamba3_cpu.CPUDecoder",
            "decoder_sha256": file_sha256(Path(__file__).resolve().parents[4] / "ssm_decode/mamba3_cpu.py"),
            "threads": torch.get_num_threads(),
            "batch_size": args.cpu_batch_size,
            "seconds": cpu_seconds,
        },
        "runtime_identity": runtime,
        "torch": torch.__version__,
        "script_sha256": file_sha256(Path(__file__)),
        "arrays": str(args.arrays.resolve()),
        "arrays_sha256": file_sha256(args.arrays),
        "acceptance": {
            "score_delta_absolute_max": SCORE_DELTA_LIMIT,
            "saved_score_oracle_absolute_max": SAVED_SCORE_ORACLE_LIMIT,
            "exact_cohort_and_truth_required": True,
            "finite_predictions_required": True,
            "causal_prefix_required": True,
            "passed": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
