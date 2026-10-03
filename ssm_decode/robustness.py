"""Support-only fake-quantization and persistent-weight-drift experiment."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from .models import ModelConfig, build_model
from .hardware import calibrate_scale, fake_quantize, fake_quantize_per_row
from .calibration import fit_ridge
from .data import load_recording, source_normalizer, source_target_plan, split_support_query


def apply_persistent_weight_noise(model, fraction: float, seed: int):
    """Copy model and apply one deterministic multiplicative weight perturbation."""
    noisy=copy.deepcopy(model); gen=torch.Generator(device="cpu"); gen.manual_seed(seed)
    with torch.no_grad():
        for _, p in noisy.named_parameters():
            if p.ndim >= 2:
                n=torch.randn(p.shape,generator=gen,dtype=p.dtype) * fraction
                p.mul_(1+n.to(p.device))
    return noisy


def quantize_weights(model):
    q=copy.deepcopy(model)
    with torch.no_grad():
        for _, p in q.named_parameters():
            if p.ndim >= 2: p.copy_(fake_quantize_per_row(p)[0])
    return q


def _norm(record, stats):
    return ((record.neural-stats["x_mean"])/stats["x_std"]).astype(np.float32), ((record.behavior-stats["y_mean"])/stats["y_std"]).astype(np.float32)


def _predict(model, x, trials, device, state_scale=None, bits=8, collect=False):
    """Trial-reset prediction; state fake quantization uses a frozen supplied scale."""
    model.eval(); out=[]; seen=[]; clips=[]
    with torch.inference_mode():
        for a,b in trials:
            state=None
            for row in x[a:b]:
                tx=torch.from_numpy(row).unsqueeze(0).to(device); y,state=model.step(tx,state)
                if state_scale is not None:
                    qmax=(1<<(bits-1))-1
                    # A symmetric round-and-clamp quantizer clips only beyond
                    # qmax+0.5 scale units; count clipped *components*, not bins.
                    clips.append((state.abs() > (qmax + 0.5) * state_scale).sum().cpu())
                    state,_=fake_quantize(state,bits=bits,scale=state_scale)
                    latent=state[...,0] if state.ndim==3 else state
                    y=model.readout(latent)
                out.append(y.squeeze(0).cpu().numpy())
                if collect: seen.append(state.detach().cpu())
                elif state_scale is None: clips.append(torch.tensor(0, dtype=torch.int64))
    return np.asarray(out), (torch.cat(seen,0) if seen else torch.empty(0)), torch.stack(clips) if clips else torch.empty(0,dtype=torch.int64)


def _r2(y,p):
    ssr=((y-p)**2).sum(0); sst=((y-y.mean(0))**2).sum(0); each=1-ssr/sst.clamp_min(1e-12)
    return float(1-ssr.sum()/sst.sum().clamp_min(1e-12)),float(each.mean())


def _load_checkpoint(path, device):
    d=torch.load(path,map_location="cpu",weights_only=False); sd=d["state_dict"]
    if "model_config" in d:
        raw=d["model_config"]; m=build_model(ModelConfig(**raw) if isinstance(raw,dict) else raw)
    else:
        inp=sd["frontend.linear.weight"].shape[1]; out=sd["readout.weight"].shape[0]
        m=build_model(ModelConfig(inp,out,width=int(d.get("width",sd["readout.weight"].shape[1])),kind=d.get("kind","diag")))
    m=m.to(device); m.load_state_dict(sd); return m,d


def score_model(model, record, stats, support_trials, sequence, device, state_bits=None):
    x,y=_norm(record,stats); starts,qs=split_support_query(record,support_trials=support_trials,sequence=sequence)
    support=np.unique(starts[:,None]+np.arange(sequence)[None,:]).reshape(-1); support=support[record.eval_mask[support]&np.isfinite(y[support]).all(1)]
    query=np.unique(qs+sequence-1); query=query[record.eval_mask[query]&np.isfinite(y[query]).all(1)]
    # Scales may inspect only states generated from support input bins.
    scale=None; sat=0
    if state_bits:
        _, support_state,_=_predict(model,x,record.trial_bounds,device,collect=True)
        # Collect output state is indexed in global trial-concatenated order: trial bounds
        # in supplied recordings are contiguous, so support indices address it directly.
        scale=calibrate_scale(support_state[support],state_bits)
    pred,_,clips=_predict(model,x,record.trial_bounds,device,state_scale=scale,bits=state_bits or 8)
    sat=int(clips[query].sum()) if scale is not None else 0
    target=y[query]*stats["y_std"]+stats["y_mean"]; physical=pred*stats["y_std"]+stats["y_mean"]; zero=_r2(torch.from_numpy(target),torch.from_numpy(physical[query]))
    mapper=fit_ridge(torch.from_numpy(pred[support]).double(),torch.from_numpy(y[support]-pred[support]).double(),1.0)
    adjusted=pred+mapper(torch.from_numpy(pred).double()).float().numpy(); adjusted=adjusted*stats["y_std"]+stats["y_mean"]; ridge=_r2(torch.from_numpy(target),torch.from_numpy(adjusted[query]))
    return {"n_support_bins":int(len(support)),"n_query_bins":int(len(query)),"query_indices":query.tolist(),"state_scale":None if scale is None else float(scale),"query_state_clipped_components":sat,"zero":{"variance_weighted_r2":zero[0],"uniform_r2":zero[1]},"ridge":{"variance_weighted_r2":ridge[0],"uniform_r2":ridge[1]}}


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--runs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");args=p.parse_args(argv)
    device=torch.device(args.device); rows=[]
    for taskdir in sorted(x for x in args.runs.iterdir() if x.is_dir() and x.name in {"m1","m2"}):
        task=taskdir.name; base_plan=source_target_plan(task); manifest=taskdir/"manifest.json"; meta=json.loads(manifest.read_text()) if manifest.exists() else {}
        plan=meta.get("cross_session_local_dev",base_plan["cross_session_local_dev"])
        normalizer=taskdir/"source_normalizer.npz"
        if normalizer.exists():
            data=np.load(normalizer); stats={k:data[k] for k in data.files}
        else:
            base=source_target_plan(task); source=[load_recording(task,"held_in",s) for s in base["source_held_in_sessions"]]; stats=source_normalizer(source)
        target=load_recording(task,"held_in",plan["target_session"])
        sequence=int(meta.get("protocol",{}).get("sequence",50)); requested=meta.get("support_trials",meta.get("args",{}).get("support_trials",[33])); budget=max(int(x) for x in requested)
        pts=[z for z in taskdir.glob("*.pt") if any(z.name.startswith(k+"_s0") for k in ("diag","osc","bank","selective","gru"))]
        for path in sorted(pts):
            model,d=_load_checkpoint(path,device); base={"task":task,"checkpoint":str(path),"kind":d.get("kind"),"seed":d.get("seed",0),"support_trials":budget,"protocol":{"query_labels_used":False,"weight_noise":"persistent multiplicative, fixed seed 20261003","state_scales":"target support only; frozen for query"}}
            conditions={"clean":model,"w8":quantize_weights(model),"noise_2pct":apply_persistent_weight_noise(model,.02,20261003),"noise_5pct":apply_persistent_weight_noise(model,.05,20261003)}
            for name,m in conditions.items():
                for bits,label in ((None,"float_state"),(8,"s8"),(16,"s16")) if name=="w8" else ((None,"float_state"),):
                    row=dict(base);row.update({"condition":name,"state":label});row.update(score_model(m,target,stats,budget,sequence,device,bits));rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({"rows":rows,"integer_chip_claim":False},indent=2)+"\n");print(args.output)

if __name__=="__main__": main()
