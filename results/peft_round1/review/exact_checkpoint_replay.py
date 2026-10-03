#!/usr/bin/env python3
"""Replay selected saved PEFT best checkpoints at the formal evaluation batch size."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.data import load_recording, source_target_plan
from ssm_decode.peft_experiment import load_saved_model


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare(name: str, replayed: np.ndarray, saved: np.ndarray) -> dict[str, Any]:
    replayed = np.asarray(replayed)
    saved = np.asarray(saved)
    if replayed.shape != saved.shape or replayed.dtype != saved.dtype:
        raise AssertionError(
            f"{name}: shape/dtype differs: {replayed.shape}/{replayed.dtype} "
            f"vs {saved.shape}/{saved.dtype}"
        )
    if np.issubdtype(saved.dtype, np.number):
        difference = np.abs(replayed.astype(np.float64) - saved.astype(np.float64))
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean()) if difference.size else 0.0
    else:
        max_abs = mean_abs = None
    exact = bool(np.array_equal(replayed, saved))
    if not exact:
        raise AssertionError(f"{name}: replay is not bit exact; max_abs={max_abs}")
    return {
        "shape": list(saved.shape),
        "dtype": str(saved.dtype),
        "bit_exact": exact,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "saved_sha256": digest(saved),
        "replayed_sha256": digest(replayed),
    }


def replay(fit: Path, device: str) -> dict[str, Any]:
    args = json.loads((fit / "replay_args.json").read_text())
    metrics = json.loads((fit / "metrics.json").read_text())
    normalizer = np.load(fit / "normalizer.npz", allow_pickle=False)
    stats = tuple(normalizer[key].astype("float32") for key in ("x_mean", "x_std", "y_mean", "y_std"))
    plan = source_target_plan(args["task"], root=Path(args["data_root"]))
    target = load_recording(
        args["task"], "held_in", plan["cross_session_local_dev"]["target_session"],
        root=Path(args["data_root"]),
    )
    model, checkpoint = load_saved_model(fit / "best.pt", device=device)
    require_step = int(metrics["best_step"])
    if int(checkpoint["best_step"]) != require_step:
        raise AssertionError("best checkpoint step differs from metrics")
    cache = debug._target_predictions(model, target, stats, torch.device(device), args["context"])
    zero = debug._score_target(cache, stats, "zero")
    ridge = debug._score_target(cache, stats, "ridge")
    saved = np.load(fit / "predictions.npz", allow_pickle=False)
    arrays = {
        "legacy_indices": cache["legacy_indices"],
        "allvalid_indices": cache["all_valid_indices"],
        "truth_legacy": cache["legacy_truth_physical"],
        "truth_allvalid": cache["all_valid_truth_physical"],
        "support_indices": cache["support_indices"],
        "support_truth_normalized": cache["support_truth_normalized"],
        "support_prediction": cache["support_prediction"],
        "zero_legacy": zero["legacy"]["prediction_physical"],
        "ridge_legacy": ridge["legacy"]["prediction_physical"],
        "zero_allvalid": zero["all_valid"]["prediction_physical"],
        "ridge_allvalid": ridge["all_valid"]["prediction_physical"],
    }
    checks = {name: compare(name, value, saved[name]) for name, value in arrays.items()}
    score_checks: dict[str, Any] = {}
    for adaptation, result in (("zero", zero), ("ridge", ridge)):
        for scope in ("legacy", "all_valid"):
            actual = float(result[scope]["r2_variance_weighted"])
            expected = float(metrics["final"][adaptation][scope]["r2_variance_weighted"])
            if actual != expected:
                raise AssertionError(
                    f"{adaptation}/{scope}: replay score differs: {actual} vs {expected}"
                )
            score_checks[f"{adaptation}_{scope}"] = {
                "bit_exact_python_float": True,
                "value": actual,
            }
    return {
        "fit": str(fit),
        "task": args["task"],
        "method": args["method"],
        "seed": args["seed"],
        "best_step": require_step,
        "device": device,
        "formal_eval_batch_size": args["eval_batch_size"],
        "arrays": checks,
        "scores": score_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    # The formal runner pins CPU reductions to one thread before scoring.
    torch.set_num_threads(1)
    results = [replay(fit, options.device) for fit in options.fit]
    report = {
        "status": "verified",
        "scope": "selected best-checkpoint forward replay at the recorded formal eval batch size",
        "fits": results,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "fits": [item["fit"] for item in results],
        "arrays_verified": sum(len(item["arrays"]) for item in results),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
