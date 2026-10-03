"""Runnable real-data SSM pilot; deliberately separate from official scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import socket
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .calibration import fit_delta, fit_ridge, fit_rls
from .data import (DEFAULT_ROOT, Recording, load_recording, manifest_json,
                   source_normalizer, source_target_plan, split_support_query)
from .models import ModelConfig, build_model


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _norm(record: Recording, stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return ((record.neural - stats["x_mean"]) / stats["x_std"]).astype(np.float32), ((record.behavior - stats["y_mean"]) / stats["y_std"]).astype(np.float32)


def _segments(records: list[Recording], stats: dict[str, np.ndarray], sequence: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    xs, ys, masks = [], [], []
    for r in records:
        x, y = _norm(r, stats)
        # Source uses full legal trials; all segments remain inside a trial.
        for a, b in r.trial_bounds:
            if b - a < sequence: continue
            for start in range(a, b - sequence + 1, sequence):
                valid = r.eval_mask[start:start + sequence] & np.isfinite(y[start:start + sequence]).all(axis=1)
                if not bool(valid.any()): continue
                xs.append(x[start:start + sequence]); ys.append(y[start:start + sequence]); masks.append(valid)
    if not xs: raise ValueError("no source segments")
    return xs, ys, masks


def _predict(model, x: np.ndarray, device: torch.device, *, chunk: int = 2048) -> np.ndarray:
    model.eval(); out = []
    # Trial reset semantics are handled by caller; each input is a single trial.
    with torch.inference_mode():
        state = None
        for a in range(0, len(x), chunk):
            # State flows across chunks but the caller resets it at trial edges.
            tx = torch.from_numpy(x[a:a + chunk]).unsqueeze(0).to(device)
            value, state = model(tx, state, return_state=True)
            out.append(value.squeeze(0).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _predict_trials(model, record: Recording, stats: dict[str, np.ndarray], device: torch.device, *, batch_trials: int = 16) -> np.ndarray:
    """Run independent trial states in padded batches; padded suffixes are discarded."""
    x, _ = _norm(record, stats); p = np.zeros((len(x), stats["y_mean"].size), np.float32)
    bounds = sorted(record.trial_bounds, key=lambda z: z[1]-z[0])
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(bounds), batch_trials):
            group = bounds[offset:offset+batch_trials]; lengths=[b-a for a,b in group]; longest=max(lengths)
            packed=np.zeros((len(group),longest,x.shape[1]),np.float32)
            for row,(a,b) in enumerate(group): packed[row,:b-a]=x[a:b]
            value=model(torch.from_numpy(packed).to(device)).float().cpu().numpy()
            for row,(a,b) in enumerate(group): p[a:b]=value[row,:b-a]
    return p


def _r2s(y: np.ndarray, p: np.ndarray) -> tuple[float, float, list[float]]:
    ssr = np.sum((y-p)**2, axis=0); sst = np.sum((y-y.mean(0,keepdims=True))**2, axis=0)
    each = 1.0 - ssr / np.maximum(sst, 1e-12)
    return float(1.0-ssr.sum()/max(float(sst.sum()),1e-12)), float(each.mean()), each.astype(float).tolist()


def train_one(*, kind: str, seed: int, source: list[Recording], stats: dict[str, np.ndarray], device: torch.device,
              width: int, sequence: int, batch_size: int, steps: int, lr: float) -> tuple[torch.nn.Module, list[dict]]:
    _seed(seed); model = build_model(ModelConfig(source[0].neural.shape[1], source[0].behavior.shape[1], width=width, kind=kind)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xs, ys, masks = _segments(source, stats, sequence); curves=[]
    for step in range(steps):
        take = np.random.randint(0, len(xs), size=batch_size)
        x = torch.from_numpy(np.stack([xs[i] for i in take])).to(device)
        y = torch.from_numpy(np.stack([ys[i] for i in take])).to(device)
        valid = torch.from_numpy(np.stack([masks[i] for i in take])).to(device).unsqueeze(-1)
        opt.zero_grad(set_to_none=True); p = model(x); loss = ((p-y).square()*valid).sum()/valid.sum().clamp_min(1)/p.shape[-1]; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0: curves.append({"step": step + 1, "mse": float(loss.detach().cpu())})
    return model, curves


def evaluate_target(record: Recording, stats: dict[str, np.ndarray], pred: np.ndarray, *, support_trials: int, query_start_trials: int, sequence: int, adaptation: str) -> dict:
    support_starts, query_starts = split_support_query(record, support_trials=support_trials, query_start_trials=query_start_trials, sequence=sequence)
    _, y = _norm(record, stats)
    # Calibration rows are all bins in prefix trials, query rows are tail windows' final bins.
    support_end = np.unique(support_starts[:, None] + np.arange(sequence)[None, :]).reshape(-1)
    support_end = support_end[record.eval_mask[support_end] & np.isfinite(y[support_end]).all(axis=1)]
    if not len(support_end): raise ValueError(f"{record.session}: no valid support bins")
    query_end = np.unique(query_starts + sequence - 1)
    query_end = query_end[record.eval_mask[query_end] & np.isfinite(y[query_end]).all(axis=1)]
    if not len(query_end): raise ValueError(f"{record.session}: no valid query endpoints")
    if set(support_end.tolist()) & set(query_end.tolist()): raise RuntimeError("support/query overlap")
    adjusted = pred.copy()
    if adaptation != "zero":
        # Frozen base output gets a target-support-only residual readout.  Query
        # labels are never read, and no base parameter is updated.
        feature = torch.from_numpy(pred[support_end]).to(torch.float64)
        residual = torch.from_numpy(y[support_end] - pred[support_end]).to(torch.float64)
        mapper = {"ridge": fit_ridge, "rls": fit_rls, "delta": fit_delta}[adaptation](feature, residual)
        with torch.inference_mode(): adjusted += mapper(torch.from_numpy(pred).to(torch.float64)).float().numpy()
    y_phys = y * stats["y_std"] + stats["y_mean"]; p_phys = adjusted * stats["y_std"] + stats["y_mean"]
    variance_weighted, uniform_average, per_dimension = _r2s(y_phys[query_end], p_phys[query_end])
    return {"session": record.session, "surface": record.split, "support_trials": support_trials, "query_start_trials":query_start_trials, "adaptation": adaptation,
            "n_support_bins": int(len(support_end)), "n_query_bins": int(len(query_end)), "r2_variance_weighted": variance_weighted, "r2_uniform_average":uniform_average, "r2_per_dimension":per_dimension,
            "support_query_overlap": False, "predictions": p_phys[query_end], "truth": y_phys[query_end], "query_indices": query_end}


def run(args: argparse.Namespace) -> Path:
    torch.set_num_threads(1)
    root, out = Path(args.data_root).resolve(), Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    plan = source_target_plan(args.task, root=root)
    source = [load_recording(args.task, "held_in", s, root=root) for s in plan["source_held_in_sessions"]]
    targets = [load_recording(args.task, "held_in", plan["cross_session_local_dev"]["target_session"], root=root)]
    targets += [load_recording(args.task, "minival", s, root=root) for s in plan["minival_targets"]]
    stats = source_normalizer(source); np.savez_compressed(out / "source_normalizer.npz", **stats)
    device=torch.device(args.device)
    config_payload={k:v for k,v in vars(args).items() if k not in {"output"}}
    config_payload["data_root"]=str(root)
    package=Path(__file__).resolve().parent
    code_hashes={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in ("models.py","calibration.py","data.py","experiment.py")}
    (out / "manifest.json").write_text(manifest_json({**plan, "protocol": {"sequence":args.sequence, "trial_reset":True, "support_prefix_nonoverlap":True, "common_query_tail":True, "checkpoint":"fixed_steps"}, "args":config_payload, "model_config":{"width":args.width,"kinds":args.kinds}, "code_sha256":code_hashes, "device":str(args.device), "cuda_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None, "host":socket.gethostname(), "torch":torch.__version__, "torch_num_threads":torch.get_num_threads()}))
    rows=[]; started=time.time()
    for kind in args.kinds:
        for seed in args.seeds:
            print(f"start kind={kind} seed={seed}", flush=True)
            model_started=time.time()
            model, curves=train_one(kind=kind, seed=seed, source=source, stats=stats, device=device, width=args.width, sequence=args.sequence, batch_size=args.batch_size, steps=args.steps, lr=args.learning_rate)
            tag=f"{kind}_s{seed}"; torch.save({"state_dict":model.state_dict(),"kind":kind,"seed":seed,"model_config":asdict(model.config),"code_sha256":code_hashes},out/f"{tag}.pt")
            (out/f"{tag}.curve.json").write_text(json.dumps(curves,indent=2)+"\n")
            for target in targets:
                feasible=[]
                for count in args.support_trials:
                    try: split_support_query(target,support_trials=count,sequence=args.sequence); feasible.append(count)
                    except ValueError: pass
                common_cutoff=max(feasible) if feasible else None
                availability={}
                for count in args.support_trials:
                    try:
                        if common_cutoff is None: raise ValueError(f"{target.session}: no requested support budget leaves a query tail")
                        support, query=split_support_query(target,support_trials=count,query_start_trials=common_cutoff,sequence=args.sequence)
                        availability[count]={"available":True,"support_starts":int(len(support)),"query_starts":int(len(query)),"query_start_trials":common_cutoff}
                    except ValueError as exc:
                        availability[count]={"available":False,"reason":str(exc)}
                pred = _predict_trials(model, target, stats, device)
                for count in args.support_trials:
                    for adaptation in args.adaptations:
                        if not availability[count]["available"]:
                            skip={"status":"skipped","kind":kind,"seed":seed,"session":target.session,"surface":target.split,"support_trials":count,"adaptation":adaptation,"common_query_cutoff_trials":common_cutoff,"reason":availability[count]["reason"],"held_out_accessed":False}
                            with (out/"skips.jsonl").open("a") as f: f.write(json.dumps(skip,sort_keys=True)+"\n")
                            continue
                        row=evaluate_target(target,stats,pred,support_trials=count,query_start_trials=common_cutoff,sequence=args.sequence,adaptation=adaptation)
                        arrays={k:row.pop(k) for k in ("predictions","truth","query_indices")}; np.savez_compressed(out/f"{tag}.{target.split}.{target.session}.k{count}.{adaptation}.npz",**arrays)
                        row.update({"kind":kind,"seed":seed,"checkpoint":f"{tag}.pt","official_evaluation":False}); rows.append(row)
                        with (out/"per_session_r2.jsonl").open("a") as f: f.write(json.dumps(row,sort_keys=True)+"\n")
            with (out/"models.jsonl").open("a") as f: f.write(json.dumps({"kind":kind,"seed":seed,"elapsed_seconds":time.time()-model_started,"status":"completed"})+"\n")
            print(f"done kind={kind} seed={seed} elapsed={time.time()-model_started:.1f}s", flush=True)
    with (out/"per_session_r2.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(rows[0]) if rows else ["empty"]); writer.writeheader(); writer.writerows(rows)
    (out/"run.json").write_text(json.dumps({"status":"completed","elapsed_seconds":time.time()-started,"rows":len(rows),"held_out_accessed":False},indent=2)+"\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=("m1","m2"),required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--output",type=Path,required=True); p.add_argument("--data-root",default=str(DEFAULT_ROOT)); p.add_argument("--steps",type=int,default=1200); p.add_argument("--width",type=int,default=64); p.add_argument("--sequence",type=int,default=50); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]); p.add_argument("--kinds",nargs="+",default=["diag","osc","bank","selective","gru"]); p.add_argument("--support-trials",nargs="+",type=int,default=[5,10,33]); p.add_argument("--adaptations",nargs="+",default=["zero","ridge","rls","delta"]); args=p.parse_args(argv); print(run(args))

if __name__ == "__main__": main()
