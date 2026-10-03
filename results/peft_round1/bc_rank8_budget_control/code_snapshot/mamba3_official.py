"""Explicit optional wrapper around the pinned official Mamba-3 reference."""
from __future__ import annotations
import importlib, json, subprocess, sys, types, traceback
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

OFFICIAL=Path(__file__).resolve().parents[1]/'.tools'/'mamba_official'
DEPS=Path(__file__).resolve().parents[1]/'.tools'/'mamba_deps'
COMMIT='e9594ce1c732d97440f0332fdc43170a2294dbfa'

def _require_isolated_triton():
    """Refuse mixed Triton module graphs; the launcher must set PYTHONPATH."""
    try: triton=importlib.import_module('triton')
    except Exception as exc: raise RuntimeError('Triton is unavailable; launch with isolated mamba_deps PYTHONPATH') from exc
    location=str(getattr(triton,'__file__',''))
    if not location.startswith(str(DEPS)):
        raise RuntimeError(f'non-isolated Triton already loaded from {location}; restart with PYTHONPATH={DEPS}:<SSM>')
    return triton

def _import_mamba3():
    if not OFFICIAL.is_dir(): raise RuntimeError(f'official Mamba-3 clone missing: {OFFICIAL}')
    _require_isolated_triton()
    # Deliberately create only the namespace package.  Importing the official
    # top-level __init__ pulls unrelated Mamba1/transformers dependencies.
    pkg=sys.modules.get('mamba_ssm')
    if pkg is None:
        pkg=types.ModuleType('mamba_ssm');pkg.__path__=[str(OFFICIAL/'mamba_ssm')];sys.modules['mamba_ssm']=pkg
    try:
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3
    except Exception as exc:
        raise RuntimeError('official Mamba-3 namespace import failed; no fallback is available') from exc

@dataclass(frozen=True)
class OfficialMamba3Config:
    input_size:int; output_size:int; width:int=64; layers:int=1; state_size:int=128; dropout:float=0.

class OfficialMamba3Decoder(nn.Module):
    def __init__(self,cfg:OfficialMamba3Config):
        super().__init__();Mamba3=_import_mamba3()
        if cfg.width<=0 or cfg.layers<1:raise ValueError('positive width/layers required')
        self.config=cfg;self.in_proj=nn.Linear(cfg.input_size,cfg.width);self.blocks=nn.ModuleList([_PreNormMamba3Block(Mamba3,cfg) for _ in range(cfg.layers)]);self.final_norm=nn.LayerNorm(cfg.width);self.out_proj=nn.Linear(cfg.width,cfg.output_size)
    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.config.input_size:raise ValueError('x must be [B,T,input_size]')
        z=self.in_proj(x)
        for b in self.blocks:z=b(z)
        return self.out_proj(self.final_norm(z))

class _PreNormMamba3Block(nn.Module):
    """Wrapper architecture only; the official Mamba3 core is unmodified."""
    def __init__(self,Mamba3,cfg):
        super().__init__();self.norm1=nn.LayerNorm(cfg.width);self.ssm=Mamba3(d_model=cfg.width,d_state=cfg.state_size,headdim=64,chunk_size=64,dropout=cfg.dropout,is_mimo=False);self.norm2=nn.LayerNorm(cfg.width);self.ffn=nn.Sequential(nn.Linear(cfg.width,2*cfg.width),nn.SiLU(),nn.Linear(2*cfg.width,cfg.width));self.dropout=nn.Dropout(cfg.dropout)
    def forward(self,x):
        x=x+self.dropout(self.ssm(self.norm1(x)))
        return x+self.dropout(self.ffn(self.norm2(x)))

def build_mamba3_official(input_size,output_size,width=64,layers=1,state_size=128,dropout=0.):
    return OfficialMamba3Decoder(OfficialMamba3Config(input_size,output_size,width,layers,state_size,dropout))

def compatibility_attempt(device='cuda:0'):
    result={'schema':'official_mamba3_siso_compatibility_v1','official_path':str(OFFICIAL),'expected_commit':COMMIT,'device':device,'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
    try:
        result['actual_commit']=subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()
        import time
        model=build_mamba3_official(8,2,128,2,16).to(device).train();x=torch.randn(32,128,8,device=device,requires_grad=True);y=model(x);y.square().mean().backward()
        model.eval();base=x.detach();changed=base.clone();changed[:,64:]+=torch.randn_like(changed[:,64:])
        with torch.no_grad(): a=model(base)[:,:64];b=model(changed)[:,:64]
        for _ in range(3): model(base)
        if torch.cuda.is_available():torch.cuda.synchronize()
        began=time.perf_counter()
        for _ in range(5): model(base)
        if torch.cuda.is_available():torch.cuda.synchronize()
        elapsed=time.perf_counter()-began
        triton=_require_isolated_triton();kernel=importlib.import_module('mamba_ssm.ops.triton.mamba3.mamba3_siso_combined');kernel_triton=getattr(kernel,'triton',None)
        result.update(status='supported',forward_shape=list(y.shape),backward_finite=bool(torch.isfinite(x.grad).all()),causal_future_perturb_max_abs=float((a-b).abs().max()),parameters=sum(p.numel() for p in model.parameters()),architecture='2x pre-LayerNorm residual official-Mamba3 SISO + LayerNorm/SiLU FFN(2x width), final LayerNorm',triton_version=triton.__version__,triton_file=str(triton.__file__),kernel_module_file=str(kernel.__file__),kernel_triton_file=str(getattr(kernel_triton,'__file__',None)),kernel_triton_matches_runtime=kernel_triton is triton,timing={'warmup':3,'iterations':5,'elapsed_seconds':elapsed,'milliseconds_per_forward':elapsed*200})
    except Exception as exc:
        chain=[];cur=exc
        while cur is not None:
            chain.append({'type':type(cur).__name__,'message':str(cur)[:1200]});cur=cur.__cause__ or cur.__context__
        result.update(status='unsupported_or_unavailable',exception_type=type(exc).__name__,exception_message=str(exc)[:1200],exception_chain=chain,traceback_tail=traceback.format_exc().splitlines()[-12:])
    return result

def write_compatibility(path=None,device='cuda:0'):
    p=Path(path or (Path(__file__).resolve().parents[1]/'results'/'debug_audit'/'mamba3_compatibility.json'));p.parent.mkdir(parents=True,exist_ok=True);r=compatibility_attempt(device);p.write_text(json.dumps(r,indent=2)+'\n');return r
