"""CPU-only inference for plain or canonical unmerged-LoRA OfficialMamba3 state.

The unmerged format is the literal ``state_dict`` format emitted by
:class:`ssm_decode.peft.LoRALinear`: ``base.weight``, optional ``base.bias``,
``A``, ``B``, and ``rows`` under each wrapped linear path.  ``LoRALinear``
does not serialize its Python ``alpha`` attribute, so this runtime accepts the
only portable deployment contract: ``alpha / rank == 1``.  It validates rank
from ``A``/``B`` geometry and evaluates the base and LoRA projections
separately; it never materializes ``base.weight + delta`` for execution.
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
_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))


@dataclass(frozen=True)
class _Linear:
    """A plain linear projection or an unmerged canonical LoRA projection."""
    weight: torch.Tensor
    bias: torch.Tensor | None
    delta: torch.Tensor | None = None
    rank: int | None = None


def _linear(x: torch.Tensor, linear: _Linear) -> torch.Tensor:
    """Evaluate base and LoRA separately so the unmerged state stays unmerged."""
    base = F.linear(x.float(), linear.weight.float(), None if linear.bias is None else linear.bias.float())
    if linear.delta is None:
        return base
    return base + F.linear(x.float(), linear.delta.float())


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
    norm1_w: torch.Tensor; norm1_b: torch.Tensor; ip: _Linear
    dt_bias: torch.Tensor; b_bias: torch.Tensor; c_bias: torch.Tensor; d: torch.Tensor
    b_norm: torch.Tensor; c_norm: torch.Tensor; op: _Linear
    norm2_w: torch.Tensor; norm2_b: torch.Tensor
    ff1: _Linear; ff2: _Linear


class CPUDecoder:
    """Fixed-window, causal CPU executor for plain or canonical unmerged LoRA state."""

    context = 128
    chunk_size = 64
    state_size = 32
    expand = 2
    headdim = 64
    rope_fraction = .5

    def __init__(self, *, in_proj, blocks, final_w, final_b, out_proj):
        self.in_proj = in_proj
        self.blocks = tuple(blocks)
        self.final_w, self.final_b, self.out_proj = final_w, final_b, out_proj
        self.input_size, self.width, self.output_size = in_proj.weight.shape[1], in_proj.weight.shape[0], out_proj.weight.shape[0]
        if self.width % self.headdim:
            raise ValueError("Mamba3 CPU runtime needs width divisible by 64")
        self.nheads = self.width * self.expand // self.headdim

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor], *, lora_alpha_over_rank: float = 1.0) -> "CPUDecoder":
        """Build a decoder from exact plain or canonical unmerged-LoRA state.

        ``rank`` is taken from ``A.shape[1] == B.shape[0]``.  Native
        ``LoRALinear.state_dict()`` has no serialised alpha/scale field, so
        this format fixes its scale to one and rejects any caller-declared
        alternative.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        if not isinstance(lora_alpha_over_rank, (int, float)) or isinstance(lora_alpha_over_rank, bool) or lora_alpha_over_rank != 1.0:
            raise ValueError("canonical unmerged LoRA requires alpha/rank == 1")
        if not all(isinstance(key, str) for key in state_dict):
            raise ValueError("state_dict keys must be strings")
        keys = set(state_dict)
        if any(any(part in key for part in _FORBIDDEN) for key in keys):
            raise ValueError("offset or unsupported adapter keys are unsupported")
        indices = sorted({int(m.group(1)) for key in keys if (m := _BLOCK.match(key))})
        if not indices or indices != list(range(len(indices))):
            raise ValueError("blocks must be contiguous from zero")

        def get_float(key: str, shape=None) -> torch.Tensor:
            if key not in state_dict:
                raise ValueError("missing key " + key)
            value = state_dict[key]
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError("CPU tensor required for " + key)
            if value.dtype not in _FLOAT_DTYPES:
                raise ValueError("real floating tensor required for " + key)
            if shape is not None and tuple(value.shape) != tuple(shape):
                raise ValueError(f"shape drift {key}: {tuple(value.shape)} != {tuple(shape)}")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite tensor " + key)
            return value.detach().float().contiguous()

        def get_rows(key: str, out_features: int) -> torch.Tensor:
            if key not in state_dict:
                raise ValueError("missing key " + key)
            value = state_dict[key]
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError("CPU tensor required for " + key)
            if value.dtype != torch.int64 or value.ndim != 1:
                raise ValueError("LoRA rows must be a 1-D int64 CPU tensor for " + key)
            rows = value.detach().contiguous()
            if rows.numel() == 0 or rows.min().item() < 0 or rows.max().item() >= out_features or torch.unique(rows).numel() != rows.numel():
                raise ValueError("LoRA rows must be unique and in range for " + key)
            return rows

        expected: set[str] = set()

        def linear(path: str, shape=None, *, has_bias: bool) -> _Linear:
            plain_weight, plain_bias = path + ".weight", path + ".bias"
            base_weight, base_bias = path + ".base.weight", path + ".base.bias"
            a_key, b_key, rows_key = path + ".A", path + ".B", path + ".rows"
            wrapper_keys = {base_weight, base_bias, a_key, b_key, rows_key}
            has_plain = plain_weight in keys or plain_bias in keys
            has_wrapper = bool(wrapper_keys & keys)
            if has_plain and has_wrapper:
                raise ValueError("mixed plain/unmerged LoRA keys for " + path)
            if not has_plain and not has_wrapper:
                raise ValueError("missing linear state for " + path)
            if has_plain:
                expected.add(plain_weight)
                weight = get_float(plain_weight, shape)
                if weight.ndim != 2:
                    raise ValueError("linear weight must be 2-D for " + plain_weight)
                if not has_bias:
                    if plain_bias in keys:
                        raise ValueError("unexpected bias for " + path)
                    bias = None
                else:
                    expected.add(plain_bias)
                    bias = get_float(plain_bias, (weight.shape[0],))
                return _Linear(weight, bias)

            if plain_bias in keys or plain_weight in keys:
                raise ValueError("mixed plain/unmerged LoRA keys for " + path)
            required = {base_weight, a_key, b_key, rows_key}
            if has_bias:
                required.add(base_bias)
            missing = required - keys
            if missing:
                raise ValueError("incomplete unmerged LoRA state for " + path + ": " + ", ".join(sorted(missing)))
            if not has_bias and base_bias in keys:
                raise ValueError("unexpected base bias for " + path)
            expected.update(required)
            weight = get_float(base_weight, shape)
            if weight.ndim != 2:
                raise ValueError("linear base weight must be 2-D for " + base_weight)
            bias = None if not has_bias else get_float(base_bias, (weight.shape[0],))
            rows = get_rows(rows_key, weight.shape[0])
            a, b = get_float(a_key), get_float(b_key)
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rows.numel() or a.shape[1] < 1 or b.shape != (a.shape[1], weight.shape[1]):
                raise ValueError("LoRA rank/geometry drift for " + path)
            delta = torch.zeros_like(weight)
            delta.index_copy_(0, rows, a @ b)  # alpha / rank is fixed and verified as one above.
            return _Linear(weight, bias, delta.contiguous(), int(a.shape[1]))

        # Root in_proj has a required base bias.  Unmerged states preserve it
        # under ``in_proj.base.bias`` even when it was trainable during PEFT.
        in_proj = linear("in_proj", has_bias=True)
        width, inputs = in_proj.weight.shape
        final_w = get_float("final_norm.weight", (width,)); final_b = get_float("final_norm.bias", (width,))
        expected.update(("final_norm.weight", "final_norm.bias"))
        # The output width is not known before reading its 2-D matrix.
        out_proj = linear("out_proj", has_bias=True)
        if out_proj.weight.ndim != 2 or out_proj.weight.shape[1] != width:
            raise ValueError("root output shape drift")
        blocks = []
        for i in indices:
            p = f"blocks.{i}."; h = cls.expand * width; heads = h // cls.headdim
            ip_rows = 2*h + 2*cls.state_size + 3*heads + cls.state_size//4
            norm1_w = get_float(p+"norm1.weight", (width,)); norm1_b = get_float(p+"norm1.bias", (width,))
            dt_bias = get_float(p+"ssm.dt_bias", (heads,)); b_bias = get_float(p+"ssm.B_bias", (heads,1,cls.state_size)); c_bias = get_float(p+"ssm.C_bias", (heads,1,cls.state_size)); d = get_float(p+"ssm.D", (heads,))
            b_norm = get_float(p+"ssm.B_norm.weight", (cls.state_size,)); c_norm = get_float(p+"ssm.C_norm.weight", (cls.state_size,))
            norm2_w = get_float(p+"norm2.weight", (width,)); norm2_b = get_float(p+"norm2.bias", (width,))
            expected.update((p+"norm1.weight",p+"norm1.bias",p+"ssm.dt_bias",p+"ssm.B_bias",p+"ssm.C_bias",p+"ssm.D",p+"ssm.B_norm.weight",p+"ssm.C_norm.weight",p+"norm2.weight",p+"norm2.bias"))
            ip = linear(p+"ssm.in_proj", (ip_rows,width), has_bias=False)
            op = linear(p+"ssm.out_proj", (width,h), has_bias=False)
            ff1 = linear(p+"ffn.0", (2*width,width), has_bias=True)
            ff2 = linear(p+"ffn.2", (width,2*width), has_bias=True)
            blocks.append(_Block(norm1_w,norm1_b,ip,dt_bias,b_bias,c_bias,d,b_norm,c_norm,op,norm2_w,norm2_b,ff1,ff2))
        extra = keys - expected
        if extra:
            raise ValueError("unsupported state keys: " + ", ".join(sorted(extra)[:4]))
        return cls(in_proj=in_proj, blocks=blocks, final_w=final_w, final_b=final_b, out_proj=out_proj)

    def _core(self, u, b: _Block):
        B,T,_=u.shape; H=self.nheads; P=self.headdim; S=self.state_size
        z,x,bb,cc,rawdt,rawa,trap,angles=torch.split(_linear(u,b.ip),[H*P,H*P,S,S,H,H,H,S//4],dim=-1)
        z,x=_bf(z).view(B,T,H,P),_bf(x).view(B,T,H,P)
        bb=_rms(bb,b.b_norm).unsqueeze(2).expand(-1,-1,H,-1); cc=_rms(cc,b.c_norm).unsqueeze(2).expand(-1,-1,H,-1)
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
        h=_linear(x,self.in_proj)
        for b in self.blocks:
            h=h+self._core(_ln(h,b.norm1_w,b.norm1_b),b)
            h=h+_linear(F.silu(_linear(_ln(h,b.norm2_w,b.norm2_b),b.ff1)),b.ff2)
        return _linear(_ln(h,self.final_w,self.final_b),self.out_proj)
