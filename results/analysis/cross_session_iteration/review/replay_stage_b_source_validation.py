"""Replay Stage B source validation checkpoints with a float64 score oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.data import load_recording
from ssm_decode.mamba3_official import build_mamba3_official
from ssm_decode.session_frontend import SessionFrontendBank
from ssm_decode.session_pretraining import (SessionAwareDecoder, continuous_source_windows,
                                             source_statistics, validate_source)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(path, device):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    payload = torch.load(path / "session_bank_best.pt", map_location=device, weights_only=False)
    args = argparse.Namespace(**payload["args"])
    records = [load_recording(args.task, "held_in", session, root=Path(args.data_root))
               for session in payload["source_sessions"]]
    splits = [debug._trial_split(record) for record in records]
    statistics = source_statistics(records, splits)
    windows = continuous_source_windows(records, [split[1] for split in splits], statistics, device)
    selections = [debug._validation_selection([window], args.context, args.max_val_endpoints)[0]
                  for window in windows]
    base = build_mamba3_official(
        records[0].neural.shape[1], records[0].behavior.shape[1], args.width,
        args.layers, args.state_size, args.dropout,
    ).to(device)
    model = SessionAwareDecoder(
        base,
        SessionFrontendBank(len(records), records[0].neural.shape[1], args.width, device=device),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    replay_score = validate_source(
        model, windows, selections, args.context, args.eval_batch_size, statistics[0],
    )
    predictions = []
    truths = []
    model.eval()
    with torch.inference_mode():
        for session, selection in enumerate(selections):
            for start in range(0, len(selection), args.eval_batch_size):
                x, y, mask = debug._right_aligned(
                    [windows[session]], selection[start:start + args.eval_batch_size], args.context,
                )
                prediction = model(x, session)[:, -1]
                valid = mask[:, -1] & torch.isfinite(y[:, -1]).all(-1)
                predictions.append(prediction[valid].float().cpu().numpy())
                truths.append(y[:, -1][valid].float().cpu().numpy())
    scale = statistics[0][3].astype(np.float64)
    offset = statistics[0][2].astype(np.float64)
    prediction = np.concatenate(predictions).astype(np.float64) * scale + offset
    truth = np.concatenate(truths).astype(np.float64) * scale + offset
    sse = float(np.square(truth - prediction).sum())
    sst = float(np.square(truth - truth.mean(0, keepdims=True)).sum())
    oracle = float(1.0 - sse / sst)
    return {
        "path": str(path.resolve()),
        "checkpoint_sha256": file_sha(path / "session_bank_best.pt"),
        "saved_best_step": manifest["best_step"],
        "saved_source_val_r2": manifest["best_source_val_r2"],
        "runner_replay_r2": replay_score,
        "float64_physical_r2": oracle,
        "runner_saved_absolute_delta": abs(replay_score - manifest["best_source_val_r2"]),
        "oracle_saved_absolute_delta": abs(oracle - manifest["best_source_val_r2"]),
        "sse": sse,
        "sst": sst,
        "n_endpoints": int(len(truth)),
        "truth_sha256": hashlib.sha256(np.ascontiguousarray(truth).tobytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    rows = [replay(path, torch.device(args.device)) for path in args.source]
    result = {"schema": "independent_stage_b_source_validation_replay_v1", "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
