"""Compare the plain CPU Mamba-3 runtime with the pinned official GPU kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ssm_decode.data import load_recording
from ssm_decode.mamba3_cpu import CPUDecoder
from ssm_decode.mamba3_official import build_mamba3_official


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r2(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(1.0 - np.square(truth - prediction).sum() /
                 np.square(truth - truth.mean(0, keepdims=True)).sum())


def differences(cpu, gpu, y_std):
    delta = cpu.astype(np.float64) - gpu.astype(np.float64)
    physical = delta * np.asarray(y_std, dtype=np.float64)
    return {
        "normalized_max_abs": float(np.abs(delta).max()),
        "normalized_rms": float(np.sqrt(np.square(delta).mean())),
        "physical_max_abs": float(np.abs(physical).max()),
        "physical_rms": float(np.sqrt(np.square(physical).mean())),
    }


def compare_checkpoint(path, device):
    path = Path(path)
    payload = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["args"])
    state = payload["state_dict"]
    cpu = CPUDecoder.from_state_dict(state)
    gpu = build_mamba3_official(
        state["in_proj.weight"].shape[1], state["out_proj.weight"].shape[0],
        args.width, args.layers, args.state_size, args.dropout,
    ).to(device).eval()
    gpu.load_state_dict(state, strict=True)
    with np.load(path / "normalizer.npz") as saved:
        stats = {key: saved[key].copy() for key in ("x_mean", "x_std", "y_mean", "y_std")}
    generator = torch.Generator().manual_seed(20261003)
    cases = {
        "random_b1_t128": torch.randn(1, 128, cpu.input_size, generator=generator),
        "random_b4_t128": torch.randn(4, 128, cpu.input_size, generator=generator),
    }
    sessions = payload["session_pretraining_receipt"]["source_sessions"]
    record = load_recording(args.task, "held_in", sessions[0], root=Path(args.data_root))
    normalized = ((record.neural - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    cases["real_b1_t128"] = torch.from_numpy(normalized[:128][None])
    cases["real_b4_t128"] = torch.from_numpy(np.stack([normalized[start:start + 128]
                                                       for start in (0, 128, 256, 384)]))
    rows = {}
    for name, inputs in cases.items():
        with torch.inference_mode():
            gpu_output = gpu(inputs.to(device)).float().cpu().numpy()
        cpu_output = cpu.forward(inputs).numpy()
        row = differences(cpu_output, gpu_output, stats["y_std"])
        if name == "real_b4_t128":
            indices = np.concatenate([np.arange(start, start + 128) for start in (0, 128, 256, 384)])
            valid = record.eval_mask[indices]
            truth = record.behavior[indices][valid]
            gpu_physical = (gpu_output.reshape(-1, gpu_output.shape[-1])[valid] * stats["y_std"] +
                            stats["y_mean"])
            cpu_physical = (cpu_output.reshape(-1, cpu_output.shape[-1])[valid] * stats["y_std"] +
                            stats["y_mean"])
            row.update(
                valid_bins=int(valid.sum()),
                gpu_physical_r2=r2(truth, gpu_physical),
                cpu_physical_r2=r2(truth, cpu_physical),
                physical_r2_delta=abs(r2(truth, cpu_physical) - r2(truth, gpu_physical)),
            )
        rows[name] = row
    return {
        "checkpoint": str((path / "best.pt").resolve()),
        "checkpoint_sha256": file_sha(path / "best.pt"),
        "width": args.width,
        "layers": args.layers,
        "cases": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    results = [compare_checkpoint(path, torch.device(args.device)) for path in args.source]
    report = {
        "schema": "mamba3_plain_cpu_official_gpu_parity_v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
