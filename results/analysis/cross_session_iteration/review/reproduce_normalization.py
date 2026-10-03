"""Independently replay source-only normalization baselines on one task."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.data import load_recording, sha256_file, source_target_plan
from ssm_decode.mamba3_official import COMMIT, OFFICIAL, build_mamba3_official
from ssm_decode.session_normalization import (
    fit_support_statistics,
    strict_past_ema_normalize,
)


ROOT = Path(__file__).resolve().parents[4]
REVIEW = Path(__file__).resolve().parent
SOURCE = {
    "m1": ROOT / "results/debug_round2/latest/m1/cross_session_mamba3_official_w256_l4_n32_ctx128_seed0",
    "m2": ROOT / "results/debug_round2/latest/m2/cross_session_mamba3_official_w256_l4_n32_ctx128_seed0",
}
OLD_AUDIT = {
    "m1": {"source_norm": 0.3862094283103943, "support_zscore": 0.6735676527023315,
           "ema_half3000": 0.6992366313934326},
    "m2": {"source_norm": -0.005921721458435059, "support_zscore": 0.23380786180496216,
           "ema_half3000": 0.2658812403678894},
}


def file_hash(path):
    return sha256_file(Path(path))


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def array_receipt(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": array_hash(array)}


def physical_r2(truth, prediction):
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    sse = float(np.square(truth64 - prediction64).sum(dtype=np.float64))
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    sst = float(np.square(centered).sum(dtype=np.float64))
    return {"r2": float(1.0 - sse / max(sst, 1e-12)), "sse": sse, "sst": sst,
            "n_bins": int(len(truth64))}


def code_hashes():
    paths = [
        Path(__file__),
        ROOT / "ssm_decode/debug_experiment.py",
        ROOT / "ssm_decode/data.py",
        ROOT / "ssm_decode/mamba3_official.py",
        ROOT / "ssm_decode/session_normalization.py",
        OFFICIAL / "mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py",
        OFFICIAL / "mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py",
    ]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def score_protocol(model, record, stats, device, context):
    cache = debug._target_predictions(model, record, stats, device, context)
    zero = debug._score_target(cache, stats, "zero")
    ridge = debug._score_target(cache, stats, "ridge")
    return cache, zero, ridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("m1", "m2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--half-life", type=float, default=3000.0)
    args = parser.parse_args(argv)

    output = args.output or REVIEW / f"normalization_replay_{args.task}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    source_dir = SOURCE[args.task]
    checkpoint_path = source_dir / "best.pt"
    normalizer_path = source_dir / "normalizer.npz"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["args"]
    saved = np.load(normalizer_path)
    source_stats = tuple(saved[name].astype(np.float32) for name in
                         ("x_mean", "x_std", "y_mean", "y_std"))

    plan = source_target_plan(args.task)
    record = load_recording(args.task, "held_in", plan["cross_session_local_dev"]["target_session"])
    usable = [bound for bound in record.trial_bounds if bound[1] - bound[0] >= 50]
    support_bounds = usable[:33]
    query_bounds = usable[33:]
    if len(support_bounds) != 33 or not query_bounds:
        raise ValueError("the fixed support and query cohorts are unavailable")
    support_end = max(end for _, end in support_bounds)
    query_start = min(start for start, _ in query_bounds)
    if support_end > query_start:
        raise AssertionError("support ends after the query starts")

    support_stats = fit_support_statistics(record.neural, support_bounds, source_stats)
    ema_neural, ema_receipt = strict_past_ema_normalize(
        record.neural, support_bounds, half_life=args.half_life,
    )
    ema_record = replace(record, neural=ema_neural)
    ema_stats = (np.zeros(record.neural.shape[1], dtype=np.float32),
                 np.ones(record.neural.shape[1], dtype=np.float32),
                 source_stats[2], source_stats[3])

    model = build_mamba3_official(
        record.neural.shape[1], record.behavior.shape[1], width=cfg["width"],
        layers=cfg["layers"], state_size=cfg["state_size"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    protocols = {
        "source_norm": (record, source_stats),
        "support_zscore": (record, support_stats),
        "ema_half3000": (ema_record, ema_stats),
    }
    results = {}
    predictions = {}
    reference_cache = None
    with torch.inference_mode():
        for name, (protocol_record, stats) in protocols.items():
            cache, zero, ridge = score_protocol(model, protocol_record, stats, device, args.context)
            if reference_cache is None:
                reference_cache = cache
            else:
                for cohort in ("support", "legacy", "all_valid"):
                    if not np.array_equal(cache[f"{cohort}_indices"], reference_cache[f"{cohort}_indices"]):
                        raise AssertionError(f"{name} changed the {cohort} cohort")
            direct_prediction = zero["all_valid"]["prediction_physical"]
            ridge_prediction = ridge["all_valid"]["prediction_physical"]
            truth = zero["all_valid"]["truth_physical"]
            direct_oracle = physical_r2(truth, direct_prediction)
            ridge_oracle = physical_r2(truth, ridge_prediction)
            runner_direct = float(zero["all_valid_query_trial_r2_variance_weighted"])
            runner_ridge = float(ridge["all_valid_query_trial_r2_variance_weighted"])
            results[name] = {
                "runner_float32_direct_r2": runner_direct,
                "runner_float32_ridge_r2": runner_ridge,
                "float64_direct_oracle": direct_oracle,
                "float64_ridge_oracle": ridge_oracle,
                "old_audit_direct_r2": OLD_AUDIT[args.task][name],
                "direct_delta_from_old_audit": runner_direct - OLD_AUDIT[args.task][name],
                "direct_prediction": array_receipt(direct_prediction),
                "ridge_prediction": array_receipt(ridge_prediction),
            }
            predictions[f"{name}_direct"] = direct_prediction
            predictions[f"{name}_ridge"] = ridge_prediction

    predictions.update(
        support_indices=reference_cache["support_indices"],
        legacy_indices=reference_cache["legacy_indices"],
        allvalid_indices=reference_cache["all_valid_indices"],
        legacy_truth_physical=reference_cache["legacy_truth_physical"],
        allvalid_truth_physical=reference_cache["all_valid_truth_physical"],
    )
    np.savez_compressed(output / "predictions.npz", **predictions)

    actual_commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True,
    ).strip()
    metrics = {
        "schema": "independent_normalization_replay_v1",
        "status": "completed",
        "task": args.task,
        "protocol": "eval_only_fixed_source_checkpoint",
        "trainable_parameters": 0,
        "query_labels_used_for_selection": False,
        "query_labels_used_for_final_scoring_only": True,
        "context": args.context,
        "eval_batch_size": 256,
        "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "source_checkpoint_sha256": file_hash(checkpoint_path),
        "source_normalizer": str(normalizer_path.relative_to(ROOT)),
        "source_normalizer_sha256": file_hash(normalizer_path),
        "target_file": str(record.path),
        "target_file_sha256": file_hash(record.path),
        "support_bounds": support_bounds,
        "support_end": support_end,
        "query_start": query_start,
        "support_before_query": support_end <= query_start,
        "support_input_mean": array_receipt(support_stats[0]),
        "support_input_std": array_receipt(support_stats[1]),
        "ema_receipt": ema_receipt,
        "cohorts": {
            "support_indices": array_receipt(reference_cache["support_indices"]),
            "legacy_indices": array_receipt(reference_cache["legacy_indices"]),
            "allvalid_indices": array_receipt(reference_cache["all_valid_indices"]),
            "legacy_truth_physical": array_receipt(reference_cache["legacy_truth_physical"]),
            "allvalid_truth_physical": array_receipt(reference_cache["all_valid_truth_physical"]),
        },
        "results": results,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device),
            "official_expected_commit": COMMIT,
            "official_actual_commit": actual_commit,
        },
        "code_sha256": code_hashes(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.cuda.synchronize(device)
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"task": args.task, "output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
