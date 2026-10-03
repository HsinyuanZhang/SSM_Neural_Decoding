"""Fit and export public calibration banks without private query access."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import debug_experiment as d
from .cross_session_iteration import (_base_model, _check_frozen, _frozen,
    _groups, _optimizer_receipt, _parameter_hashes, _runtime_identity, _unfreeze, partition_windows)
from .input_adaptation import configure_adaptation, merged_adapted_state_dict
from .official_calibration import load_public_calibration
from .session_normalization import fit_support_statistics

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("x_mean", "x_std", "y_mean", "y_std")
CODE = ("official_payload.py", "official_calibration.py", "input_adaptation.py",
        "cross_session_iteration.py", "session_normalization.py", "debug_experiment.py",
        "mamba3_official.py", "peft.py", "falcon_decoder.py", "mamba3_cpu.py")


def reader_identity():
    modules = ("apst.data.load", "apst.data.catalog", "falcon_challenge.config",
               "falcon_challenge.dataloaders", "falcon_challenge.interface")
    return {"falcon_challenge_version": importlib.metadata.version("falcon-challenge"),
            "modules": {name: {"path": str(Path(inspect.getfile(importlib.import_module(name))).resolve()),
                                "sha256": sha(inspect.getfile(importlib.import_module(name)))}
                        for name in modules}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def partitions(record):
    """Use the same segment-based interleaved validation rule as the probe."""
    segments = list(record.trial_bounds)
    if segments[0][0] > 0:
        segments.insert(0, (0, segments[0][0]))
    count = math.ceil(len(segments) / 5)
    held = set(range(4, len(segments), 5))
    if len(held) < count:
        held.add(len(segments) - 1)
    return ([bounds for index, bounds in enumerate(segments) if index not in held],
            [bounds for index, bounds in enumerate(segments) if index in held])


def fit_bank(record, source, source_stats, args, output, identity):
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    d._seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    stats = fit_support_statistics(record.neural, [(0, len(record.neural))], source_stats)
    np.savez(output / "normalizer.npz", **dict(zip(KEYS, stats)))
    train_bounds, val_bounds = partitions(record)
    model = _base_model(source["args"], record.neural.shape[1], record.behavior.shape[1], device)
    model.load_state_dict(source["state_dict"], strict=True)
    scope = configure_adaptation(model, args.method, rank=4, alpha=4, lora_scope="all", train_input_bias=True)
    root = getattr(model.in_proj, "base", model.in_proj)
    scope["root_input_bias_trainable"] = root.bias is not None and root.bias.requires_grad
    scope["root_bias_behavior"] = "trainable" if scope["root_input_bias_trainable"] else "frozen"
    train = partition_windows(record, train_bounds, stats, device, "recording_causal_fixed_window")
    valid = partition_windows(record, val_bounds, stats, device, "recording_causal_fixed_window")
    eligible = d._eligible(train)
    selection, val_index_sha = d._validation_selection(valid, 128, 10**9)
    frozen, initial = _frozen(model), _parameter_hashes(model)
    optimizer = torch.optim.AdamW(_groups([p for p in model.parameters() if p.requires_grad], args.lr, 1e-4)) if args.method != "none" else None
    optimizer_receipt = _optimizer_receipt(model, optimizer, 0) if optimizer else None
    optimizer_stages = [optimizer_receipt] if optimizer else []
    freeze_audits = []
    started = time.perf_counter()
    best = d._val(model, valid, 128, selection, 256, stats)
    if not np.isfinite(best):
        raise FloatingPointError("Nonfinite initial public-prefix validation")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_step, step, budget = 0, 0, args.steps if optimizer else 0
    logs, extensions = [{"step": 0, "prefix_val_r2": best}], []
    d._seed(args.seed)
    while step < budget:
        step += 1
        if args.method == "full" and step in (200, 400):
            freeze_audits.append(_check_frozen(model, frozen, step))
            if _unfreeze(model, optimizer, step, args.lr, 1e-4):
                optimizer_stages.append(_optimizer_receipt(model, optimizer, step))
                frozen = _frozen(model)
        model.train()
        x, y, mask = d._sample_batch(train, eligible, 128, 32)
        prediction, truth = d._forward(model, x)[:, 64:], y[:, 64:]
        good = mask[:, 64:] & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(prediction).all():
            raise FloatingPointError("Invalid public calibration batch")
        optimizer.zero_grad(set_to_none=True)
        loss = (prediction[good] - truth[good]).square().mean()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite public calibration optimization")
        optimizer.step()
        row = {"step": step, "train_loss": float(loss.detach()), "grad_norm": float(grad)}
        if step % 100 == 0 or step == budget:
            score = d._val(model, valid, 128, selection, 256, stats)
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite public calibration validation")
            row["prefix_val_r2"] = score
            if score > best:
                best, best_step = score, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        logs.append(row)
        if step == budget and best_step >= .8 * budget and budget < args.max_steps:
            updated = min(args.max_steps, 2 * budget)
            extensions.append({"from": budget, "to": updated, "best_step": best_step})
            budget = updated
    freeze_audit = _check_frozen(model, frozen, step)
    model.load_state_dict(best_state, strict=True)
    torch.save({"state_dict": best_state, "args": vars(args), "best_step": best_step}, output / "unmerged_best.pt")
    merged = {key: value.detach().cpu().clone() for key, value in merged_adapted_state_dict(model).items()}
    plain = _base_model(source["args"], record.neural.shape[1], record.behavior.shape[1], device).eval()
    plain.load_state_dict(merged, strict=True)
    # Folding is checked on all legal prefix-val endpoints using the GPU kernel.
    model.eval()
    fold_error = 0.
    fold_close = True
    with torch.inference_mode():
        for lo in range(0, len(selection), 256):
            x, _, _ = d._right_aligned(valid, selection[lo:lo+256], 128)
            original, folded = model(x), plain(x)
            error = (original - folded).abs().max()
            fold_error = max(fold_error, float(error))
            fold_close = fold_close and torch.allclose(original, folded, atol=1e-2, rtol=1e-2)
    converged = optimizer is None or best_step < .8 * budget
    receipt = dict(identity, public_calibration=record.receipt,
        train_bounds=train_bounds, val_bounds=val_bounds, prefix_val_indices_sha256=val_index_sha,
        best_step=best_step, best_prefix_val_r2=best, completed_steps=step, final_budget=budget,
        convergence_satisfied=converged, extensions=extensions, adaptation=scope,
        optimizer=optimizer_receipt, optimizer_stages=optimizer_stages,
        frozen_audit=freeze_audit, stage_freeze_audits=freeze_audits,
        initial_parameter_hashes=initial, best_parameter_hashes=_parameter_hashes(model),
        fold_output_max_abs=fold_error, fold_atol=1e-2, fold_rtol=1e-2, fold_allclose=fold_close,
        fit_seconds=time.perf_counter()-started, normalizer_sha256=sha(output / "normalizer.npz"),
        query_labels_used=False, query_used_for_selection=False,
        output_space="official physical units from Falcon load_nwb; no task-specific rescaling or smoothing")
    write(output / "train_log.json", logs)
    receipt["train_log_sha256"] = sha(output / "train_log.json")
    # A merged GPU source-val score catches errors that are small in weights
    # but amplified by BF16 boundaries downstream.
    merged_score = d._val(plain, valid, 128, selection, 256, stats)
    receipt["merged_prefix_val_r2"] = merged_score
    receipt["fold_r2_abs_delta"] = abs(merged_score - best)
    write(output / "receipt.json", receipt)
    if sha(record.path) != record.receipt["rawfile_sha256"]:
        raise RuntimeError("Public calibration file changed during fitting")
    if not converged or not fold_close or receipt["fold_r2_abs_delta"] > 1e-3:
        raise RuntimeError("Public bank fails convergence or merged-score acceptance")
    torch.save({"state_dict": merged}, output / "model.pt")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("m1", "m2"), required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("none", "io", "lora", "affine_lora", "full"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/work_host/SPINT/SPINT-main/data"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if (args.method == "none" and args.lr != 0) or (args.method != "none" and args.lr <= 0) or not 0 < args.steps <= args.max_steps:
        raise ValueError("Invalid predeclared public-bank training contract")
    args.source, args.output, args.data_root = (str(path.resolve()) for path in (args.source, args.output, args.data_root))
    from apst.data.load import list_sessions
    from falcon_challenge.config import FalconConfig, FalconTask
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    source_normalizer = Path(args.source).parent / "normalizer.npz"
    with np.load(source_normalizer) as archive:
        source_stats = tuple(archive[key].astype(np.float32) for key in KEYS)
    identity = dict(args=vars(args), source_checkpoint_sha256=sha(args.source),
        source_normalizer_sha256=sha(source_normalizer), runtime_identity=_runtime_identity(),
        code_sha256={name: sha(ROOT / "ssm_decode" / name) for name in CODE},
        reader_identity=reader_identity())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "export_contract.json", identity)
    task_config = FalconConfig(task=getattr(FalconTask, args.task))
    rows = []
    for split in ("held_in", "held_out"):
        for item in list_sessions(args.task, split, root=Path(args.data_root)):
            record = load_public_calibration(args.task, split, item["session"], args.data_root)
            tag = task_config.hash_dataset(record.path.stem)
            destination = output / "banks" / tag
            receipt = fit_bank(record, source, source_stats, args, destination, identity)
            rows.append(dict(tag=tag, checkpoint_file=f"banks/{tag}/model.pt",
                             normalizer_file=f"banks/{tag}/normalizer.npz",
                             calibration_receipt_sha256=sha(destination / "receipt.json")))
    if (identity["code_sha256"] != {name: sha(ROOT / "ssm_decode" / name) for name in CODE}
            or identity["source_checkpoint_sha256"] != sha(args.source)
            or identity["source_normalizer_sha256"] != sha(source_normalizer)
            or identity["reader_identity"] != reader_identity()):
        raise RuntimeError("Frozen export code/source changed during calibration")
    # Keep the deployment payload separate from fit diagnostics and raw data.
    payload = output / "payload"
    payload.mkdir()
    package = payload / "ssm_decode"
    package.mkdir()
    (package / "__init__.py").write_text('"""Packaged CPU SSM inference only."""\n')
    for name in ("mamba3_cpu.py", "falcon_decoder.py"):
        shutil.copyfile(ROOT / "ssm_decode" / name, package / name)
    shutil.copyfile(ROOT / "third_party/mamba/LICENSE", payload / "MAMBA_LICENSE")
    for row in rows:
        for key in ("checkpoint_file", "normalizer_file"):
            destination = payload / row[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / row[key], destination)
    files = {str(path.relative_to(payload)): sha(path) for path in payload.rglob("*") if path.is_file()}
    write(payload / "payload_manifest.json", dict(schema="ssm_falcon_cpu_payload_v1",
        task=args.task, input_size=64 if args.task == "m1" else 96,
        output_size=16 if args.task == "m1" else 2, context=128,
        normalization="fixed_calibration_zscore", output_space="official_physical",
        output_postprocess="none", calibration_trials=10 if args.task == "m1" else 33,
        method=args.method, models=rows, files=files,
        public_calibration_only=True, query_labels_used=False, test_time_parameter_updates=False,
        export_contract=identity))


if __name__ == "__main__":
    main()
