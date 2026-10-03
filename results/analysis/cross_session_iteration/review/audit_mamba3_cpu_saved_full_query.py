"""Compare a selected plain M1 CPU checkpoint with saved official GPU output.

The caller selects the checkpoint. This program does not rank candidates. It
does not start CUDA work. It checks the complete frozen all-valid query cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssm_decode.cross_session_iteration import KEYS, _code_hashes, _file_hash, _runtime_identity, _target
from ssm_decode.mamba3_cpu import CPUDecoder
from ssm_decode.session_normalization import normalize_neural


SCORE_DELTA_LIMIT = 1e-3
SAVED_SCORE_ORACLE_LIMIT = 2e-6


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def tensor_state_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key].detach().cpu(), right[key].detach().cpu())
        for key in left
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--cpu-batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    if args.cpu_batch_size < 1:
        raise ValueError("CPU batch size must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    checkpoint = args.checkpoint.resolve()
    fit_dir = checkpoint.parent
    metrics_path = fit_dir / "query_evaluation/metrics.json"
    saved_path = fit_dir / "query_evaluation/predictions.npz"
    manifest_path = fit_dir / "manifest.json"
    fit_result_path = fit_dir / "fit_result.json"
    normalizer_path = fit_dir / "normalizer.npz"
    for path in (checkpoint, metrics_path, saved_path, manifest_path, fit_result_path, normalizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = json.loads(metrics_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    fit_result = json.loads(fit_result_path.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fit_args = argparse.Namespace(**payload["args"])

    if fit_args.task != "m1" or fit_args.method != "none":
        raise ValueError("This CPU audit accepts only a selected M1 none checkpoint")
    if fit_args.policy != "recording_causal_fixed_window" or fit_args.context != 128:
        raise ValueError("The selected checkpoint does not use the CPU deployment policy")
    if fit_args.support_trials != 11 or fit_args.query_cutoff_trials != 33:
        raise ValueError("The selected checkpoint does not use the M1 ten-trial contract")
    if metrics.get("status") != "completed" or fit_result.get("status") != "completed":
        raise RuntimeError("The selected fit or evaluation is not complete")
    if metrics.get("query_used_for_selection") is not False:
        raise RuntimeError("The saved evaluation does not deny query selection")
    if file_sha256(checkpoint) != metrics.get("checkpoint_sha256"):
        raise RuntimeError("The query evaluation refers to a different checkpoint")
    if payload["code_hashes"] != _code_hashes():
        raise RuntimeError("The current replay implementation differs from the fitted code")
    current_runtime = _runtime_identity()
    if payload["runtime_identity"] != current_runtime or metrics.get("runtime_identity") != current_runtime:
        raise RuntimeError("The current runtime differs from the fitted or evaluated runtime")

    source_checkpoint = Path(manifest["args"]["pretrained"]).resolve()
    source_normalizer = source_checkpoint.parent / "normalizer.npz"
    if file_sha256(source_checkpoint) != manifest.get("source_checkpoint_sha256"):
        raise RuntimeError("The source checkpoint hash changed")
    if file_sha256(source_normalizer) != manifest.get("source_normalizer_sha256"):
        raise RuntimeError("The source normalizer hash changed")
    if file_sha256(normalizer_path) != manifest.get("normalizer_sha256"):
        raise RuntimeError("The selected target normalizer hash changed")
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_state_equal = tensor_state_equal(payload["state_dict"], source_payload["state_dict"])
    if not source_state_equal:
        raise RuntimeError("The none checkpoint state differs from its frozen source checkpoint")
    if manifest.get("changed_parameter_names") != {"best": [], "last": []}:
        raise RuntimeError("The none fit reports a changed parameter")

    with np.load(normalizer_path) as archive:
        stats = tuple(archive[key].astype(np.float32, copy=False) for key in KEYS)
    payload_stats = tuple(np.asarray(payload["normalizer"][key], dtype=np.float32) for key in KEYS)
    normalizer_exact = all(np.array_equal(left, right) for left, right in zip(stats, payload_stats))
    if not normalizer_exact:
        raise RuntimeError("The normalizer archive differs from the checkpoint normalizer")

    target, plan = _target(fit_args.task, fit_args.data_root, fit_args.target_session)
    target_hash = _file_hash(target.path)
    if target_hash != payload["target_data_sha256"] or target_hash != manifest.get("target_data_sha256"):
        raise RuntimeError("The current target file differs from the selected fit")
    usable = [tuple(bounds) for bounds in target.trial_bounds if bounds[1] - bounds[0] >= 50]
    query_bounds = usable[fit_args.query_cutoff_trials :]
    if not query_bounds:
        raise ValueError("The selected checkpoint has no query cohort")
    query_indices = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in query_bounds])
    query_indices = query_indices[target.eval_mask[query_indices]]
    if len(np.unique(query_indices)) != len(query_indices):
        raise RuntimeError("The rebuilt query cohort contains duplicate indices")
    truth = target.behavior[query_indices].astype(np.float64, copy=False)

    with np.load(saved_path) as saved:
        saved_indices = saved["allvalid_indices"].astype(np.int64, copy=False)
        saved_truth = saved["truth_allvalid"].astype(np.float64, copy=False)
        saved_gpu = saved["zero_allvalid"].astype(np.float64, copy=False)
    cohort_exact = bool(np.array_equal(query_indices, saved_indices))
    truth_exact = bool(np.array_equal(truth, saved_truth))
    if not cohort_exact or not truth_exact:
        raise RuntimeError("The rebuilt query cohort differs from the saved GPU evaluation")
    if len(query_indices) != 50_591:
        raise RuntimeError("The M1 all-valid query cohort does not contain 50,591 bins")

    cpu = CPUDecoder.from_state_dict(payload["state_dict"])
    neural = normalize_neural(target.neural, stats)
    rows = []
    first_inputs = None
    began = time.perf_counter()
    for start in range(0, len(query_indices), args.cpu_batch_size):
        selected = query_indices[start : start + args.cpu_batch_size]
        inputs = torch.from_numpy(make_windows(neural, selected, fit_args.context))
        if first_inputs is None:
            first_inputs = inputs[:1].clone()
        prediction = cpu.forward(inputs)[:, -1]
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("A CPU query prediction is not finite")
        rows.append(prediction.numpy())
        if start == 0 or start + args.cpu_batch_size >= len(query_indices) or start % 3200 == 0:
            print(f"CPU full-query endpoints {min(start + args.cpu_batch_size, len(query_indices))}/{len(query_indices)}", flush=True)
    cpu_seconds = time.perf_counter() - began
    cpu_normalized = np.concatenate(rows).astype(np.float32, copy=False)
    y_mean, y_std = stats[2].astype(np.float64), stats[3].astype(np.float64)
    cpu_physical = cpu_normalized.astype(np.float64) * y_std + y_mean

    saved_gpu_r2, saved_gpu_sse, sst = physical_r2(truth, saved_gpu)
    cpu_r2, cpu_sse, cpu_sst = physical_r2(truth, cpu_physical)
    if cpu_sst != sst:
        raise AssertionError("The GPU and CPU score denominators differ")
    metric_r2 = float(metrics["final"]["zero"]["all_valid_query_trial_r2_variance_weighted"])
    saved_score_oracle_delta = abs(saved_gpu_r2 - metric_r2)
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
        and source_state_equal
        and normalizer_exact
        and prefix_equal
        and saved_score_oracle_delta <= SAVED_SCORE_ORACLE_LIMIT
        and abs(score_delta) <= SCORE_DELTA_LIMIT
    )

    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.arrays,
        all_valid_query_indices=query_indices,
        truth_physical=truth,
        saved_gpu_prediction_physical=saved_gpu,
        cpu_prediction_physical=cpu_physical,
    )
    delta = cpu_physical - saved_gpu
    report = {
        "schema": "mamba3_cpu_saved_gpu_full_query_parity_v1",
        "status": "PASS" if passed else "FAIL",
        "scope": "full_query",
        "selection_boundary": "checkpoint supplied by caller; this audit does not rank candidates",
        "reference": "saved official GPU prediction archive",
        "gpu_calls": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_normalizer_sha256": file_sha256(source_normalizer),
        "target_normalizer_sha256": file_sha256(normalizer_path),
        "target_data_sha256": target_hash,
        "query_metrics": str(metrics_path.resolve()),
        "query_metrics_sha256": file_sha256(metrics_path),
        "saved_predictions": str(saved_path.resolve()),
        "saved_predictions_sha256": file_sha256(saved_path),
        "method": fit_args.method,
        "seed": fit_args.seed,
        "lr": fit_args.lr,
        "support_trials_field": fit_args.support_trials,
        "query_cutoff_trials_field": fit_args.query_cutoff_trials,
        "policy": fit_args.policy,
        "target_session": target.session,
        "n_query_bounds": len(query_bounds),
        "n_all_valid_query_bins": int(len(query_indices)),
        "cohort_exact_to_saved_gpu": cohort_exact,
        "truth_exact_to_saved_gpu": truth_exact,
        "cohort_sha256": array_sha256(query_indices),
        "truth_sha256": array_sha256(truth),
        "saved_gpu_prediction_sha256": array_sha256(saved_gpu),
        "cpu_prediction_sha256": array_sha256(cpu_physical),
        "source_state_tensor_exact": source_state_equal,
        "normalizer_exact_to_checkpoint": normalizer_exact,
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
        "runtime_identity": current_runtime,
        "torch": torch.__version__,
        "script_sha256": file_sha256(Path(__file__)),
        "arrays": str(args.arrays.resolve()),
        "arrays_sha256": file_sha256(args.arrays),
        "acceptance": {
            "score_delta_absolute_max": SCORE_DELTA_LIMIT,
            "saved_score_oracle_absolute_max": SAVED_SCORE_ORACLE_LIMIT,
            "exact_cohort_and_truth_required": True,
            "exact_source_state_required": True,
            "finite_predictions_required": True,
            "causal_prefix_required": True,
            "passed": passed,
        },
        "plan": plan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
