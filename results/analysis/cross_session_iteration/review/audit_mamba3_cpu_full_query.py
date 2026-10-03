"""Audit one selected M1 adaptation on its complete development-query cohort.

The caller must select the checkpoint before this audit. This program does not
rank candidates. It compares the selected adapted GPU model, its merged plain
GPU export, and the CPU decoder on the same frozen all-valid query endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.cross_session_iteration import KEYS, _file_hash, _runtime_identity, _target, load_saved_model
from ssm_decode.input_adaptation import merged_adapted_state_dict
from ssm_decode.mamba3_cpu import CPUDecoder
from ssm_decode.mamba3_official import build_mamba3_official
from ssm_decode.session_normalization import normalize_neural


R2_ABSOLUTE_DELTA_LIMIT = 1e-3
FOLD_ATOL = 1e-2
FOLD_RTOL = 1e-2
SAVED_REPLAY_R2_ATOL = 1e-6


def sha256_bytes(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    sse = float(np.square(truth64 - prediction64).sum())
    sst = float(np.square(truth64 - truth64.mean(0, keepdims=True)).sum())
    return float(1.0 - sse / sst), sse, sst


def describe_delta(left: np.ndarray, right: np.ndarray) -> dict:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "mean_signed": float(delta.mean()),
        "mean_absolute": float(np.abs(delta).mean()),
        "rms": float(np.sqrt(np.square(delta).mean())),
        "max_absolute": float(np.abs(delta).max()),
    }


def make_windows(neural: np.ndarray, endpoints: np.ndarray, context: int) -> np.ndarray:
    windows = np.zeros((len(endpoints), context, neural.shape[1]), dtype=np.float32)
    for row, endpoint in enumerate(endpoints):
        left = max(0, int(endpoint) - context + 1)
        length = int(endpoint) - left + 1
        windows[row, context - length :] = neural[left : int(endpoint) + 1]
    return windows


def causality(cpu: CPUDecoder, gpu_plain, gpu_adapted, inputs: torch.Tensor, device: torch.device) -> dict:
    original = inputs[:1].clone()
    changed = original.clone()
    generator = torch.Generator().manual_seed(20261003)
    changed[:, 96:] += 3 * torch.randn(changed[:, 96:].shape, generator=generator)
    cpu_a, cpu_b = cpu.forward(original).numpy(), cpu.forward(changed).numpy()
    with torch.inference_mode():
        plain_a = debug._forward(gpu_plain, original.to(device)).float().cpu().numpy()
        plain_b = debug._forward(gpu_plain, changed.to(device)).float().cpu().numpy()
        adapted_a = debug._forward(gpu_adapted, original.to(device)).float().cpu().numpy()
        adapted_b = debug._forward(gpu_adapted, changed.to(device)).float().cpu().numpy()
    rows = {}
    for name, first, second in (
        ("cpu_plain", cpu_a, cpu_b),
        ("gpu_plain", plain_a, plain_b),
        ("gpu_adapted", adapted_a, adapted_b),
    ):
        delta = np.abs(first[:, :96].astype(np.float64) - second[:, :96].astype(np.float64))
        rows[name] = {
            "prefix_bitwise_equal": bool(np.array_equal(first[:, :96], second[:, :96])),
            "prefix_max_absolute_delta": float(delta.max()),
        }
    return {
        "perturbed_input_start": 96,
        "checked_output_stop_exclusive": 96,
        "runtimes": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-batch-size", type=int, default=32)
    parser.add_argument("--gpu-batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The reference replay must use an official CUDA model")
    torch.cuda.set_device(device)

    checkpoint = args.checkpoint.resolve()
    evaluation = checkpoint.parent / "query_evaluation"
    metrics_path = evaluation / "metrics.json"
    predictions_path = evaluation / "predictions.npz"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("The selected checkpoint needs a completed query evaluation")
    metrics = json.loads(metrics_path.read_text())
    gpu_adapted, payload = load_saved_model(checkpoint, device)
    fit_args = argparse.Namespace(**payload["args"])
    replay = payload["iteration_replay"]
    if fit_args.task != "m1" or fit_args.policy != "recording_causal_fixed_window":
        raise ValueError("This audit accepts only an M1 recording-causal checkpoint")
    if fit_args.method not in {"none", "io", "lora", "affine", "affine_lora"}:
        raise ValueError("The selected adaptation cannot be merged into the plain CPU decoder")
    if _file_hash(checkpoint) != metrics["checkpoint_sha256"]:
        raise RuntimeError("The saved query evaluation refers to a different checkpoint")
    if payload["runtime_identity"] != _runtime_identity():
        raise RuntimeError("The current official runtime differs from the fitted runtime")

    target, plan = _target(fit_args.task, fit_args.data_root, fit_args.target_session)
    if _file_hash(target.path) != payload["target_data_sha256"]:
        raise RuntimeError("The current target file differs from the fitted target")
    stats = tuple(np.asarray(payload["normalizer"][key], dtype=np.float32) for key in KEYS)
    usable = [tuple(bounds) for bounds in target.trial_bounds if bounds[1] - bounds[0] >= 50]
    query_bounds = usable[fit_args.query_cutoff_trials :]
    if not query_bounds:
        raise ValueError("The checkpoint has no development-query cohort")
    query_indices = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in query_bounds])
    query_indices = query_indices[target.eval_mask[query_indices]]
    neural = normalize_neural(target.neural, stats)
    truth = target.behavior[query_indices].astype(np.float64, copy=False)

    with np.load(predictions_path) as saved:
        saved_indices = saved["allvalid_indices"].astype(np.int64, copy=False)
        saved_truth = saved["truth_allvalid"].astype(np.float64, copy=False)
        saved_prediction = saved["zero_allvalid"].astype(np.float64, copy=False)
    cohort_exact = bool(np.array_equal(query_indices, saved_indices))
    truth_exact = bool(np.array_equal(truth, saved_truth))
    if not cohort_exact or not truth_exact:
        raise RuntimeError("The independently rebuilt query cohort differs from the saved evaluation")

    merged = {key: value.detach().cpu() for key, value in merged_adapted_state_dict(gpu_adapted).items()}
    base_cfg = replay["base_cfg"]
    gpu_plain = build_mamba3_official(
        replay["input_size"], replay["output_size"], base_cfg["width"],
        base_cfg["layers"], base_cfg["state_size"], base_cfg["dropout"],
    ).to(device).eval()
    gpu_plain.load_state_dict(merged, strict=True)
    cpu_plain = CPUDecoder.from_state_dict(merged)

    gpu_adapted_rows = []
    gpu_plain_rows = []
    cpu_rows = []
    first_inputs = None
    gpu_seconds = 0.0
    cpu_seconds = 0.0
    combined_batch = min(args.cpu_batch_size, args.gpu_batch_size)
    for start in range(0, len(query_indices), combined_batch):
        selected = query_indices[start : start + combined_batch]
        inputs = torch.from_numpy(make_windows(neural, selected, fit_args.context))
        if first_inputs is None:
            first_inputs = inputs.clone()
        began = time.perf_counter()
        with torch.inference_mode():
            adapted = debug._forward(gpu_adapted, inputs.to(device))[:, -1].float().cpu()
            plain = debug._forward(gpu_plain, inputs.to(device))[:, -1].float().cpu()
        torch.cuda.synchronize(device)
        gpu_seconds += time.perf_counter() - began
        began = time.perf_counter()
        cpu = cpu_plain.forward(inputs)[:, -1]
        cpu_seconds += time.perf_counter() - began
        if not torch.isfinite(adapted).all() or not torch.isfinite(plain).all() or not torch.isfinite(cpu).all():
            raise FloatingPointError("A full-query prediction is not finite")
        gpu_adapted_rows.append(adapted.numpy())
        gpu_plain_rows.append(plain.numpy())
        cpu_rows.append(cpu.numpy())
        if start == 0 or start + combined_batch >= len(query_indices) or (start // combined_batch) % 100 == 0:
            print(f"full-query endpoints {min(start + combined_batch, len(query_indices))}/{len(query_indices)}", flush=True)

    gpu_adapted_normalized = np.concatenate(gpu_adapted_rows).astype(np.float32, copy=False)
    gpu_plain_normalized = np.concatenate(gpu_plain_rows).astype(np.float32, copy=False)
    cpu_normalized = np.concatenate(cpu_rows).astype(np.float32, copy=False)
    y_mean, y_std = stats[2].astype(np.float64), stats[3].astype(np.float64)
    gpu_adapted_physical = gpu_adapted_normalized.astype(np.float64) * y_std + y_mean
    gpu_plain_physical = gpu_plain_normalized.astype(np.float64) * y_std + y_mean
    cpu_physical = cpu_normalized.astype(np.float64) * y_std + y_mean

    adapted_r2, adapted_sse, sst = r2(truth, gpu_adapted_physical)
    plain_r2, plain_sse, plain_sst = r2(truth, gpu_plain_physical)
    cpu_r2, cpu_sse, cpu_sst = r2(truth, cpu_physical)
    if plain_sst != sst or cpu_sst != sst:
        raise AssertionError("R2 denominators differ")
    saved_r2 = float(metrics["final"]["zero"]["all_valid_query_trial_r2_variance_weighted"])
    saved_replay_r2_delta = abs(adapted_r2 - saved_r2)
    saved_prediction_max_delta = float(np.abs(gpu_adapted_physical - saved_prediction).max())
    fold_r2_delta = abs(adapted_r2 - plain_r2)
    cpu_gpu_r2_delta = abs(cpu_r2 - plain_r2)
    fold_allclose = bool(np.allclose(
        gpu_adapted_normalized, gpu_plain_normalized, atol=FOLD_ATOL, rtol=FOLD_RTOL,
    ))
    causal = causality(cpu_plain, gpu_plain, gpu_adapted, first_inputs, device)
    causal_pass = all(row["prefix_bitwise_equal"] for row in causal["runtimes"].values())
    finite = bool(
        np.isfinite(truth).all() and np.isfinite(gpu_adapted_physical).all()
        and np.isfinite(gpu_plain_physical).all() and np.isfinite(cpu_physical).all()
    )
    passed = bool(
        finite and cohort_exact and truth_exact and causal_pass and fold_allclose
        and saved_replay_r2_delta <= SAVED_REPLAY_R2_ATOL
        and fold_r2_delta <= R2_ABSOLUTE_DELTA_LIMIT
        and cpu_gpu_r2_delta <= R2_ABSOLUTE_DELTA_LIMIT
    )

    arrays = {
        "all_valid_query_indices": query_indices,
        "truth_physical": truth,
        "gpu_adapted_prediction_physical": gpu_adapted_physical,
        "gpu_plain_prediction_physical": gpu_plain_physical,
        "cpu_plain_prediction_physical": cpu_physical,
    }
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    report = {
        "schema": "mamba3_selected_adaptation_cpu_official_gpu_full_query_parity_v1",
        "scope": "selected M1 candidate; complete frozen all-valid local development-query cohort",
        "selection_boundary": "checkpoint supplied by caller; this audit does not rank or select candidates",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "query_metrics": str(metrics_path.resolve()),
        "query_metrics_sha256": sha256_file(metrics_path),
        "saved_predictions_sha256": sha256_file(predictions_path),
        "method": fit_args.method,
        "seed": fit_args.seed,
        "lr": fit_args.lr,
        "width": base_cfg["width"],
        "layers": base_cfg["layers"],
        "support_trials_field": fit_args.support_trials,
        "query_cutoff_trials_field": fit_args.query_cutoff_trials,
        "policy": fit_args.policy,
        "target_session": target.session,
        "target_data_sha256": _file_hash(target.path),
        "plan": plan,
        "n_query_bounds": len(query_bounds),
        "n_all_valid_query_bins": int(len(query_indices)),
        "cohort_exact_to_saved_evaluation": cohort_exact,
        "truth_exact_to_saved_evaluation": truth_exact,
        "cohort_sha256": sha256_bytes(query_indices),
        "truth_sha256": sha256_bytes(truth),
        "gpu_adapted_prediction_sha256": sha256_bytes(gpu_adapted_physical),
        "gpu_plain_prediction_sha256": sha256_bytes(gpu_plain_physical),
        "cpu_plain_prediction_sha256": sha256_bytes(cpu_physical),
        "scores": {
            "saved_gpu_adapted_physical_r2": saved_r2,
            "replayed_gpu_adapted_physical_r2": adapted_r2,
            "merged_gpu_plain_physical_r2": plain_r2,
            "merged_cpu_plain_physical_r2": cpu_r2,
            "saved_replay_r2_absolute_delta": saved_replay_r2_delta,
            "adapted_to_merged_gpu_r2_absolute_delta": fold_r2_delta,
            "merged_gpu_to_cpu_r2_absolute_delta": cpu_gpu_r2_delta,
            "gpu_adapted_sse": adapted_sse,
            "gpu_plain_sse": plain_sse,
            "cpu_plain_sse": cpu_sse,
            "sst": sst,
        },
        "pointwise": {
            "saved_to_replayed_gpu_adapted_physical_max_absolute": saved_prediction_max_delta,
            "adapted_gpu_to_merged_gpu_normalized": describe_delta(
                gpu_adapted_normalized, gpu_plain_normalized,
            ),
            "merged_gpu_to_cpu_normalized": describe_delta(gpu_plain_normalized, cpu_normalized),
            "adapted_gpu_to_merged_gpu_physical": describe_delta(
                gpu_adapted_physical, gpu_plain_physical,
            ),
            "merged_gpu_to_cpu_physical": describe_delta(gpu_plain_physical, cpu_physical),
            "fold_allclose_atol": FOLD_ATOL,
            "fold_allclose_rtol": FOLD_RTOL,
            "fold_allclose": fold_allclose,
        },
        "all_values_finite": finite,
        "causality": causal,
        "timing": {
            "cpu_threads": torch.get_num_threads(),
            "batch_size": combined_batch,
            "gpu_seconds_including_transfers_for_two_models": gpu_seconds,
            "cpu_seconds": cpu_seconds,
        },
        "runtime_identity": _runtime_identity(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "script_sha256": sha256_file(Path(__file__)),
        "arrays": str(args.arrays.resolve()),
        "arrays_sha256": sha256_file(args.arrays),
        "acceptance": {
            "saved_replay_r2_absolute_delta_max": SAVED_REPLAY_R2_ATOL,
            "fold_r2_absolute_delta_max": R2_ABSOLUTE_DELTA_LIMIT,
            "cpu_gpu_r2_absolute_delta_max": R2_ABSOLUTE_DELTA_LIMIT,
            "finite_predictions_required": True,
            "causal_prefix_required": True,
            "fold_pointwise_allclose_required": True,
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
