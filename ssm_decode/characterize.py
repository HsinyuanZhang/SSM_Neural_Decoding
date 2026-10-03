"""Checkpoint-only runtime, resource, and fake-quantization characterization."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from statistics import median
import torch
from .models import ModelConfig, build_model
from .hardware import resource_accounting


def _config(payload: dict) -> ModelConfig:
    sd = payload["state_dict"]
    kind, width = payload.get("kind", "diag"), int(payload.get("width", sd["readout.weight"].shape[1]))
    output = sd["readout.weight"].shape[0]
    if "frontend.linear.weight" in sd: input_size, frontend = sd["frontend.linear.weight"].shape[1], "linear"
    elif "frontend.first.weight" in sd: input_size, frontend = sd["frontend.first.weight"].shape[1], "linear"
    else: raise ValueError("checkpoint has no supported raw-channel frontend")
    return ModelConfig(input_size=input_size, output_size=output, width=width, kind=kind, frontend=frontend)


def _latency(model, device: torch.device, steps: int = 220) -> dict:
    model.eval(); torch.set_num_threads(1); x=torch.zeros(1,model.config.input_size,device=device); state=model.initial_state(1,device)
    with torch.inference_mode():
        for _ in range(20): _,state=model.step(x,state)
        if device.type == "cuda": torch.cuda.synchronize(device)
        values=[]
        for _ in range(max(200,steps)):
            start=time.perf_counter_ns(); _,state=model.step(x,state)
            if device.type == "cuda": torch.cuda.synchronize(device)
            values.append((time.perf_counter_ns()-start)/1e6)
    values.sort(); return {"samples":len(values),"p50_ms":median(values),"p95_ms":values[round(.95*(len(values)-1))],"synchronized":device.type=="cuda"}


def _bytes(model) -> dict:
    def count(items): return sum(t.numel()*t.element_size() for _,t in items)
    matrices=[p for _,p in model.named_parameters() if p.ndim >= 2]
    w8=sum(p.numel() for p in matrices)+4*sum(p.shape[0] for p in matrices)
    return {"parameter_float_bytes":count(model.named_parameters()),"buffer_float_bytes":count(model.named_buffers()),"weight8_per_row_scale_bytes":w8,"weight8_scheme":"int8 values + float32 scale per matrix output row"}


def characterize_checkpoint(path: Path, gpu: bool) -> dict:
    payload=torch.load(path,map_location="cpu",weights_only=False); model=build_model(_config(payload)); model.load_state_dict(payload["state_dict"]); cfg=model.config
    result={"checkpoint":str(path),"kind":cfg.kind,"config":cfg.__dict__,"cpu_step":_latency(model,torch.device("cpu")),"resources":resource_accounting(model),"storage":_bytes(model),"calibration_cost":{"residual_output_d":cfg.output_size,"future_latent_d":cfg.width,"formula":"centered sufficient stats: d*d + d*outputs + d + outputs + 1","residual_state_scalars":cfg.output_size*cfg.output_size+cfg.output_size*cfg.output_size+2*cfg.output_size+1,"future_latent_state_scalars":cfg.width*cfg.width+cfg.width*cfg.output_size+cfg.width+cfg.output_size+1},"quantization":{"weights":"per-row calibrated symmetric W8; scales frozen before query","state":"S8/S16 scale from source or target support only; no per-query dynamic scale"}}
    if cfg.frontend == "profile":
        result["profile_folding_memory"]={"immutable_basis_float_bytes":cfg.profile_size*cfg.width*4,"per_session_folded_matrix_float_bytes":cfg.input_size*cfg.width*4,"folded_shape":[cfg.input_size,cfg.width],"note":"profile-dependent folded matrix is static during a session, not a checkpoint buffer"}
    if gpu and torch.cuda.is_available(): result["cuda_step"]=_latency(model.to("cuda"),torch.device("cuda"))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--runs",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--gpu",action="store_true"); args=p.parse_args(argv)
    paths=sorted(args.runs.rglob("*.pt")); args.output.parent.mkdir(parents=True,exist_ok=True); rows=[]
    for path in paths:
        try: rows.append(characterize_checkpoint(path,args.gpu))
        except Exception as exc: rows.append({"checkpoint":str(path),"error":str(exc)})
    args.output.write_text(json.dumps({"checkpoints_found":len(paths),"models":rows},indent=2)+"\n"); print(args.output)


if __name__ == "__main__": main()
