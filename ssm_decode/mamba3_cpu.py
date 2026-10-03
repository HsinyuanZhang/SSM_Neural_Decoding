"""CPU-only inference for the project's plain exported OfficialMamba3Decoder.

This is a narrow, inference-only port of the SISO forward path in the pinned
Mamba-3 source (Apache-2.0; Copyright (c) Dao AI Lab, Goombalab), commit
e9594ce1c732d97440f0332fdc43170a2294dbfa. This port changes the original
Triton implementation to CPU PyTorch; see third_party/mamba/LICENSE.
It intentionally has no Triton, CUDA, APST,
or adapter dependency.  It accepts only fully merged/plain state dictionaries.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_FORBIDDEN = ("input_adaptation", "state_offset", "research_adapter", "membrane")


def _linear(x, weight, bias=None):
    return F.linear(x.float(), weight.float(), None if bias is None else bias.float())


def _ln(x, weight, bias):
    return F.layer_norm(x.float(), (weight.numel(),), weight.float(), bias.float(), 1e-5)


def _bf(x):
    """Match materialized BF16 boundaries in the fused SISO kernel."""
    return x.to(torch.bfloat16).float()


def _rms(x, weight):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight.float()


@dataclass(frozen=True)
class _Block:
    norm1_w: torch.Tensor; norm1_b: torch.Tensor; ip: torch.Tensor
    dt_bias: torch.Tensor; b_bias: torch.Tensor; c_bias: torch.Tensor; d: torch.Tensor
    b_norm: torch.Tensor; c_norm: torch.Tensor; op: torch.Tensor
    norm2_w: torch.Tensor; norm2_b: torch.Tensor
    ff1_w: torch.Tensor; ff1_b: torch.Tensor; ff2_w: torch.Tensor; ff2_b: torch.Tensor


class CPUDecoder:
    """Fixed-window, causal CPU executor for plain OfficialMamba3Decoder exports."""

    context = 128
    chunk_size = 64
    state_size = 32
    expand = 2
    headdim = 64
    rope_fraction = .5

    def __init__(self, *, in_w, in_b, blocks, final_w, final_b, out_w, out_b):
        self.in_w, self.in_b = in_w, in_b
        self.blocks = tuple(blocks)
        self.final_w, self.final_b, self.out_w, self.out_b = final_w, final_b, out_w, out_b
        self.input_size, self.width, self.output_size = in_w.shape[1], in_w.shape[0], out_w.shape[0]
        if self.width % self.headdim:
            raise ValueError("plain Mamba3 CPU runtime needs width divisible by 64")
        self.nheads = self.width * self.expand // self.headdim

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> "CPUDecoder":
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        keys = set(state_dict)
        if any(any(part in key for part in _FORBIDDEN) or key.endswith((".A", ".B")) for key in keys):
            raise ValueError("offset or unmerged adapter/LoRA keys are unsupported")
        if any(key.endswith((".base.weight", ".base.bias")) for key in keys):
            raise ValueError("unmerged LoRA wrapper keys are unsupported")
        indices = sorted({int(m.group(1)) for key in keys if (m := _BLOCK.match(key))})
        if not indices or indices != list(range(len(indices))):
            raise ValueError("blocks must be contiguous from zero")
        def get(k, shape=None):
            if k not in state_dict: raise ValueError("missing key " + k)
            v = state_dict[k]
            if not isinstance(v, torch.Tensor) or v.device.type != "cpu": raise ValueError("CPU tensor required for " + k)
            if shape is not None and tuple(v.shape) != tuple(shape): raise ValueError(f"shape drift {k}: {tuple(v.shape)} != {tuple(shape)}")
            return v.detach().float().contiguous()
        in_w = get("in_proj.weight"); width, inputs = in_w.shape
        in_b = get("in_proj.bias", (width,)); final_w=get("final_norm.weight",(width,)); final_b=get("final_norm.bias",(width,))
        out_w=get("out_proj.weight"); out_b=get("out_proj.bias",(out_w.shape[0],))
        if tuple(out_w.shape[1:]) != (width,): raise ValueError("root output shape drift")
        blocks=[]; expected=set(("in_proj.weight","in_proj.bias","final_norm.weight","final_norm.bias","out_proj.weight","out_proj.bias"))
        for i in indices:
            p=f"blocks.{i}."; h=cls.expand*width; heads=h//cls.headdim; ip_rows=2*h+2*cls.state_size+3*heads+cls.state_size//4
            names={
                "norm1.weight":(width,), "norm1.bias":(width,), "ssm.in_proj.weight":(ip_rows,width),
                "ssm.dt_bias":(heads,), "ssm.B_bias":(heads,1,cls.state_size), "ssm.C_bias":(heads,1,cls.state_size), "ssm.D":(heads,),
                "ssm.B_norm.weight":(cls.state_size,), "ssm.C_norm.weight":(cls.state_size,), "ssm.out_proj.weight":(width,h),
                "norm2.weight":(width,), "norm2.bias":(width,), "ffn.0.weight":(2*width,width), "ffn.0.bias":(2*width,), "ffn.2.weight":(width,2*width), "ffn.2.bias":(width,)}
            expected.update(p+k for k in names)
            val={k:get(p+k,s) for k,s in names.items()}
            blocks.append(_Block(val['norm1.weight'],val['norm1.bias'],val['ssm.in_proj.weight'],val['ssm.dt_bias'],val['ssm.B_bias'],val['ssm.C_bias'],val['ssm.D'],val['ssm.B_norm.weight'],val['ssm.C_norm.weight'],val['ssm.out_proj.weight'],val['norm2.weight'],val['norm2.bias'],val['ffn.0.weight'],val['ffn.0.bias'],val['ffn.2.weight'],val['ffn.2.bias']))
        extra=keys-expected
        if extra: raise ValueError("unsupported non-plain keys: " + ", ".join(sorted(extra)[:4]))
        return cls(in_w=in_w,in_b=in_b,blocks=blocks,final_w=final_w,final_b=final_b,out_w=out_w,out_b=out_b)

    def _core(self, u, b: _Block):
        B,T,_=u.shape; H=self.nheads; P=self.headdim; S=self.state_size
        z,x,bb,cc,rawdt,rawa,trap,angles=torch.split(_linear(u,b.ip),[H*P,H*P,S,S,H,H,H,S//4],dim=-1)
        z,x=_bf(z).view(B,T,H,P),_bf(x).view(B,T,H,P)
        # The native module normalizes one BC head, then expands it to all heads.
        bb=_rms(bb,b.b_norm).unsqueeze(2).expand(-1,-1,H,-1); cc=_rms(cc,b.c_norm).unsqueeze(2).expand(-1,-1,H,-1)
        # Kernel phase 1 loads normalized BC in its storage dtype, then adds
        # FP32 bias.  Quantizing after the add is a different computation.
        qpre=_bf(cc)+b.c_bias.squeeze(1)[None,None].float(); kpre=_bf(bb)+b.b_bias.squeeze(1)[None,None].float()
        dt=F.softplus(rawdt.float()+b.dt_bias.float())
        activation=rawa.float().clamp_min(0)+torch.reciprocal(1-rawa.float().clamp_max(0))
        a=(-activation).clamp_max(-1e-4); adt=a*dt
        trap=_bf(trap); angles=_bf(angles).unsqueeze(2).expand(-1,-1,H,-1)
        theta=torch.remainder(torch.cumsum(torch.tanh(angles.float())*math.pi*dt[...,None],1),2*math.pi)
        def rot(v, *, materialize):
            rotary = theta.shape[-1] * 2
            pair=v[...,:rotary].view(B,T,H,rotary//2,2); co,si=torch.cos(theta),torch.sin(theta)
            r=torch.stack((pair[...,0]*co-pair[...,1]*si,pair[...,0]*si+pair[...,1]*co),-1).flatten(-2)
            result=torch.cat((r,v[...,rotary:]),-1)
            return _bf(result) if materialize else result
        q=rot(qpre,materialize=True); k=rot(kpre,materialize=False)
        gamma=dt*torch.sigmoid(trap.float()); next_gamma=torch.zeros_like(gamma); next_gamma[:,:-1]=dt[:,1:]*(1-torch.sigmoid(trap[:,1:].float())); scale=gamma+next_gamma
        ks=_bf(k*scale[...,None]); out=torch.empty_like(x)
        carry=torch.zeros(B,H,P,S)
        for lo in range(0,T,self.chunk_size):
            hi=min(T,lo+self.chunk_size); n=hi-lo; ac=adt[:,lo:hi]; cs=torch.cumsum(ac,1); rev=cs[:,-1:]-cs
            qb, kb, vb=q[:,lo:hi],ks[:,lo:hi],x[:,lo:hi]
            previous=torch.einsum('bths,bhps->bthp',qb,_bf(carry))*torch.exp(cs)[...,None]
            dot=torch.einsum('bihs,bjhs->bhij',qb,kb)
            decay=torch.exp(torch.minimum(cs.transpose(1,2)[:,:,:,None]-cs.transpose(1,2)[:,:,None,:],torch.zeros((),device=u.device)))
            lower=torch.tril(torch.ones(n,n,device=u.device,dtype=torch.bool),diagonal=-1)
            off=_bf(dot*decay*lower[None,None])
            current=torch.einsum('bhij,bjhp->bihp',off,vb).permute(0,1,2,3)
            diag=gamma[:,lo:hi]*torch.sum(qpre[:,lo:hi]*kpre[:,lo:hi],-1)
            raw=previous+current+(b.d[None,None,:,None]+diag[...,None])*vb
            out[:,lo:hi]=_bf(raw*F.silu(z[:,lo:hi]))
            vr=_bf(vb*torch.exp(rev)[...,None]); carry=carry*torch.exp(cs[:,-1])[...,None,None]+torch.einsum('bthp,bths->bhps',vr,kb)
        return _linear(out.flatten(2),b.op)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.ndim!=3 or x.shape[-1]!=self.input_size: raise ValueError(f'x must be CPU [B,T,{self.input_size}]')
        if not 1<=x.shape[1]<=self.context: raise ValueError('CPU runtime accepts fixed causal windows of length 1..128')
        h=_linear(x,self.in_w,self.in_b)
        for b in self.blocks:
            h=h+self._core(_ln(h,b.norm1_w,b.norm1_b),b)
            h=h+_linear(F.silu(_linear(_ln(h,b.norm2_w,b.norm2_b),b.ff1_w,b.ff1_b)),b.ff2_w,b.ff2_b)
        return _linear(_ln(h,self.final_w,self.final_b),self.out_w,self.out_b)
