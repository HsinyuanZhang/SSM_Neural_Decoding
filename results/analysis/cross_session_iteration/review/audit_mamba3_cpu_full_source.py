"""Audit plain CPU/GPU parity on the complete Stage B M1 source-validation cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.data import load_recording
from ssm_decode.mamba3_cpu import CPUDecoder
from ssm_decode.mamba3_official import build_mamba3_official
from ssm_decode.session_pretraining import continuous_source_windows, source_statistics


R2_ABSOLUTE_DELTA_LIMIT = 1e-3


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    sse = float(np.square(truth - prediction).sum())
    sst = float(np.square(truth - truth.mean(0, keepdims=True)).sum())
    return float(1.0 - sse / sst), sse, sst


def error_summary(cpu: np.ndarray, gpu: np.ndarray, y_std: np.ndarray) -> dict:
    delta = cpu.astype(np.float64) - gpu.astype(np.float64)
    physical = delta * np.asarray(y_std, dtype=np.float64)

    def describe(value: np.ndarray) -> dict:
        return {
            "mean_signed": float(value.mean()),
            "mean_absolute": float(np.abs(value).mean()),
            "rms": float(np.sqrt(np.square(value).mean())),
            "max_absolute": float(np.abs(value).max()),
        }

    return {"normalized": describe(delta), "physical": describe(physical)}


def actual_causality(cpu: CPUDecoder, gpu, inputs: torch.Tensor, device: torch.device) -> dict:
    original = inputs[:1].clone()
    changed = original.clone()
    generator = torch.Generator().manual_seed(20261003)
    changed[:, 96:] += 3 * torch.randn(changed[:, 96:].shape, generator=generator)
    cpu_a, cpu_b = cpu.forward(original).numpy(), cpu.forward(changed).numpy()
    with torch.inference_mode():
        gpu_a = gpu(original.to(device)).float().cpu().numpy()
        gpu_b = gpu(changed.to(device)).float().cpu().numpy()
    cpu_delta = np.abs(cpu_a[:, :96].astype(np.float64) - cpu_b[:, :96]).max()
    gpu_delta = np.abs(gpu_a[:, :96].astype(np.float64) - gpu_b[:, :96]).max()
    return {
        "perturbed_input_start": 96,
        "checked_output_stop_exclusive": 96,
        "cpu_prefix_bitwise_equal": bool(np.array_equal(cpu_a[:, :96], cpu_b[:, :96])),
        "gpu_prefix_bitwise_equal": bool(np.array_equal(gpu_a[:, :96], gpu_b[:, :96])),
        "cpu_prefix_max_absolute_delta": float(cpu_delta),
        "gpu_prefix_max_absolute_delta": float(gpu_delta),
    }


def compare(path: Path, device: torch.device, cpu_batch: int) -> tuple[dict, dict[str, np.ndarray]]:
    payload = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    bank_payload = torch.load(path / "session_bank_best.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    state = payload["state_dict"]
    records = [load_recording(args.task, "held_in", session, root=Path(args.data_root))
               for session in bank_payload["source_sessions"]]
    splits = [debug._trial_split(record) for record in records]
    statistics = source_statistics(records, splits)
    windows = continuous_source_windows(records, [split[1] for split in splits], statistics,
                                         torch.device("cpu"))
    selections = [debug._validation_selection([window], args.context, args.max_val_endpoints)[0]
                  for window in windows]

    cpu = CPUDecoder.from_state_dict(state)
    gpu = build_mamba3_official(
        state["in_proj.weight"].shape[1], state["out_proj.weight"].shape[0],
        args.width, args.layers, args.state_size, args.dropout,
    ).to(device).eval()
    gpu.load_state_dict(state, strict=True)

    cpu_predictions, gpu_predictions, truths, cohort = [], [], [], []
    cpu_seconds = 0.0
    gpu_seconds = 0.0
    first_inputs = None
    for session, selection in enumerate(selections):
        for lo in range(0, len(selection), cpu_batch):
            batch_selection = selection[lo:lo + cpu_batch]
            inputs, labels, mask = debug._right_aligned([windows[session]], batch_selection, args.context)
            if first_inputs is None:
                first_inputs = inputs
            started = time.perf_counter()
            cpu_output = cpu.forward(inputs)[:, -1]
            cpu_seconds += time.perf_counter() - started
            started = time.perf_counter()
            with torch.inference_mode():
                gpu_output = gpu(inputs.to(device))[:, -1].float().cpu()
            torch.cuda.synchronize(device)
            gpu_seconds += time.perf_counter() - started
            valid = mask[:, -1] & torch.isfinite(labels[:, -1]).all(-1)
            if not valid.all():
                raise RuntimeError("a selected validation endpoint is invalid")
            if not torch.isfinite(cpu_output).all() or not torch.isfinite(gpu_output).all():
                raise FloatingPointError("CPU or GPU plain prediction is not finite")
            cpu_predictions.append(cpu_output[valid].numpy())
            gpu_predictions.append(gpu_output[valid].numpy())
            truths.append(labels[:, -1][valid].numpy())
            cohort.extend((session, int(endpoint)) for _, endpoint in batch_selection)
        print(f"{path.name}: session {session + 1}/{len(selections)} complete", flush=True)

    cpu_normalized = np.concatenate(cpu_predictions).astype(np.float32, copy=False)
    gpu_normalized = np.concatenate(gpu_predictions).astype(np.float32, copy=False)
    truth_normalized = np.concatenate(truths).astype(np.float32, copy=False)
    cohort = np.asarray(cohort, dtype=np.int64)
    y_mean = statistics[0][2].astype(np.float64)
    y_std = statistics[0][3].astype(np.float64)
    truth = truth_normalized.astype(np.float64) * y_std + y_mean
    cpu_physical = cpu_normalized.astype(np.float64) * y_std + y_mean
    gpu_physical = gpu_normalized.astype(np.float64) * y_std + y_mean
    cpu_r2, cpu_sse, sst = r2(truth, cpu_physical)
    gpu_r2, gpu_sse, gpu_sst = r2(truth, gpu_physical)
    assert sst == gpu_sst
    delta = abs(cpu_r2 - gpu_r2)
    causal = actual_causality(cpu, gpu, first_inputs, device)
    finite = bool(np.isfinite(cpu_physical).all() and np.isfinite(gpu_physical).all() and
                  np.isfinite(truth).all())
    accepted = bool(finite and causal["cpu_prefix_bitwise_equal"] and
                    delta <= R2_ABSOLUTE_DELTA_LIMIT)
    result = {
        "path": str(path.resolve()),
        "plain_checkpoint_sha256": sha256_file(path / "best.pt"),
        "bank_checkpoint_sha256": sha256_file(path / "session_bank_best.pt"),
        "width": args.width,
        "layers": args.layers,
        "cpu_threads": torch.get_num_threads(),
        "cpu_batch_size": cpu_batch,
        "n_sessions": len(records),
        "n_endpoints": int(len(truth)),
        "cohort_sha256": sha256_bytes(cohort),
        "truth_sha256": sha256_bytes(truth),
        "cpu_prediction_sha256": sha256_bytes(cpu_physical),
        "gpu_prediction_sha256": sha256_bytes(gpu_physical),
        "all_values_finite": finite,
        "causality": causal,
        "errors": error_summary(cpu_normalized, gpu_normalized, statistics[0][3]),
        "gpu_plain_physical_r2": gpu_r2,
        "cpu_plain_physical_r2": cpu_r2,
        "physical_r2_absolute_delta": delta,
        "gpu_plain_sse": gpu_sse,
        "cpu_plain_sse": cpu_sse,
        "sst": sst,
        "saved_bank_physical_r2": float(bank_payload["source_val_r2"]),
        "gpu_plain_minus_saved_bank_r2": gpu_r2 - float(bank_payload["source_val_r2"]),
        "cpu_seconds": cpu_seconds,
        "gpu_seconds_including_transfers": gpu_seconds,
        "acceptance": {
            "r2_absolute_delta_limit": R2_ABSOLUTE_DELTA_LIMIT,
            "r2_within_limit": bool(delta <= R2_ABSOLUTE_DELTA_LIMIT),
            "finite_required": True,
            "causal_prefix_required": True,
            "passed": accepted,
        },
    }
    arrays = {
        "cohort": cohort,
        "truth_physical": truth,
        "cpu_prediction_physical": cpu_physical,
        "gpu_prediction_physical": gpu_physical,
    }
    return result, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    results, arrays = [], {}
    for path in args.source:
        row, values = compare(path, device, args.cpu_batch_size)
        results.append(row)
        for key, value in values.items():
            arrays[f"{path.name}_{key}"] = value
    report = {
        "schema": "mamba3_plain_cpu_official_gpu_full_source_parity_v1",
        "scope": "complete frozen Stage B M1 source-validation cohorts; plain mean-fold export on both runtimes",
        "acceptance_predeclared": {"direct_physical_r2_absolute_delta_max": R2_ABSOLUTE_DELTA_LIMIT,
                                    "finite_predictions": True, "causal_prefix": True},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "script_sha256": sha256_file(Path(__file__)),
        "results": results,
        "all_passed": all(row["acceptance"]["passed"] for row in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    report["arrays"] = str(args.arrays.resolve())
    report["arrays_sha256"] = sha256_file(args.arrays)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
