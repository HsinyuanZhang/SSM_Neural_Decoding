"""Functional Mamba-3 adapter ports. These ports are custom experiments."""
from __future__ import annotations

import json
import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from .peft import LoRALinear


def _siso_kernel(**kwargs):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    return mamba3_siso_combined(**kwargs)


class Membrane(nn.Module):
    """Integrate each batch item separately. Reset at every window."""

    def __init__(self, width, rank, *, device=None, dtype=None):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.down = nn.Linear(width, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, width, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)
        self.tau = 2.
        self.threshold = 1.

    def forward(self, z, previous_layer=None):
        drive = self.down(z)
        if previous_layer is not None:
            if previous_layer.shape != drive.shape:
                raise ValueError("cross-layer membrane shape does not match")
            drive = drive + previous_layer
        membrane = torch.zeros_like(drive[:, 0])
        trace = []
        for t in range(z.shape[1]):
            membrane = membrane + (drive[:, t] - membrane) / self.tau
            membrane = torch.where(membrane >= self.threshold, torch.zeros_like(membrane), membrane)
            trace.append(membrane)
        trace = torch.stack(trace, dim=1)
        return self.up(trace), trace


class StateOffset(nn.Module):
    """Map a fixed state offset through the current rotated readout."""

    def __init__(self, heads, channels, state_size, rank, *, device=None, dtype=None):
        super().__init__()
        if min(heads, channels, state_size, rank) < 1:
            raise ValueError("offset dimensions must be positive")
        self.heads, self.channels, self.state_size = heads, channels, state_size
        self.U = nn.Parameter(torch.zeros(heads * channels, rank, device=device, dtype=dtype))
        self.V = nn.Parameter(torch.empty(rank, state_size, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, rotated_c):
        offset = (self.U @ self.V).reshape(self.heads, self.channels, self.state_size)
        return torch.einsum("bthn,hpn->bthp", rotated_c, offset)


def effective_readout(c, bias, angles, dt):
    """Approximate the official BF16 readout coordinates with PyTorch operations."""
    heads = bias.shape[0]
    c = c.to(torch.bfloat16).float().repeat_interleave(heads // c.shape[2], dim=2)
    q = c + bias.float()[None, None]
    rates = torch.tanh(angles.to(torch.bfloat16).float()) * math.pi
    theta = torch.cumsum(rates * dt.float()[..., None], dim=1)
    theta = torch.remainder(theta, 2 * math.pi).to(torch.bfloat16).float()
    count = theta.shape[-1]
    pairs = q[..., :2 * count].reshape(*q.shape[:-1], count, 2)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    first = pairs[..., 0] * cosine - pairs[..., 1] * sine
    second = pairs[..., 0] * sine + pairs[..., 1] * cosine
    rotated = torch.stack((first, second), dim=-1).flatten(-2)
    return torch.cat((rotated, q[..., 2 * count:]), dim=-1).to(torch.bfloat16).float()


def _core_forward(core, u, previous_layer=None):
    batch, length, _ = u.shape
    sizes = [core.d_inner, core.d_inner,
             core.d_state * core.num_bc_heads, core.d_state * core.num_bc_heads,
             core.nheads, core.nheads, core.nheads, core.num_rope_angles]
    z, x, b, c, raw_dt, raw_a, trap, angles = torch.split(core.in_proj(u), sizes, dim=-1)
    z = z.reshape(batch, length, core.nheads, core.headdim)
    x = x.reshape(batch, length, core.nheads, core.headdim)
    b = core.B_norm(b.reshape(batch, length, 1, core.num_bc_heads, core.d_state))
    c = core.C_norm(c.reshape(batch, length, 1, core.num_bc_heads, core.d_state))
    raw_a = raw_a.float()
    a = -(raw_a.clamp_min(0) + torch.reciprocal(1 - raw_a.clamp_max(0)))
    a = a.clamp_max(-core.A_floor)
    dt = F.softplus(raw_dt + core.dt_bias)
    angles = angles.unsqueeze(-2).expand(-1, -1, core.nheads, -1).float()
    trace = None
    if isinstance(core.research_adapter, Membrane):
        correction, trace = core.research_adapter(z.flatten(-2), previous_layer)
        z = z + correction.reshape_as(z)
    y = _siso_kernel(Q=c.squeeze(2), K=b.squeeze(2), V=x, ADT=(a * dt).transpose(1, 2),
                     DT=dt.transpose(1, 2), Trap=trap.transpose(1, 2),
                     Q_bias=core.C_bias.squeeze(1), K_bias=core.B_bias.squeeze(1),
                     Angles=angles, D=core.D, Z=z, chunk_size=core.chunk_size,
                     Input_States=None, return_final_states=False, cu_seqlens=None)
    if isinstance(core.research_adapter, StateOffset):
        q = effective_readout(c.squeeze(2), core.C_bias.squeeze(1), angles, dt)
        offset_readout = core.research_adapter(q.to(core.research_adapter.U.dtype))
        # The kernel gates the base scan. Gate the new offset once.
        y = y.float() + F.silu(z.to(torch.bfloat16).float()) * offset_readout.float()
    return core.out_proj(y.flatten(-2).to(x.dtype)), trace


def _patched_core_forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
    if seq_idx is not None or cu_seqlens is not None or inference_params is not None:
        raise ValueError("research ports support independent full windows only")
    return _core_forward(self, u)[0]


def _membrane_decoder_forward(self, x):
    if x.ndim != 3 or x.shape[-1] != self.config.input_size:
        raise ValueError("x must have shape [B,T,input_size]")
    hidden = self.in_proj(x)
    previous_layer = None
    for block in self.blocks:
        output, previous_layer = _core_forward(block.ssm, block.norm1(hidden), previous_layer)
        hidden = hidden + block.dropout(output)
        hidden = hidden + block.dropout(block.ffn(block.norm2(hidden)))
    return self.out_proj(self.final_norm(hidden))


def install_research_adapters(model, method, rank=4):
    if method not in {"state_offset", "memba_causal"} or not isinstance(rank, int) or rank < 1:
        raise ValueError("invalid research method or rank")
    for block in model.blocks:
        core = block.ssm
        if core.is_mimo or core.is_outproj_norm or core.mimo_rank != 1:
            raise ValueError("research ports require SISO without output normalization")
        if hasattr(core, "research_adapter"):
            raise ValueError("research adapter is already installed")
    for p in model.parameters():
        p.requires_grad_(False)
    for path in ("in_proj", "out_proj"):
        base = getattr(model, path)
        if not isinstance(base, nn.Linear):
            raise ValueError("root projections must be Linear")
        setattr(model, path, LoRALinear(base, rank))
    for block in model.blocks:
        core = block.ssm
        kwargs = {"device": core.in_proj.weight.device, "dtype": core.in_proj.weight.dtype}
        if method == "state_offset":
            core.research_adapter = StateOffset(core.nheads, core.headdim, core.d_state, rank, **kwargs)
        else:
            core.research_adapter = Membrane(core.d_inner, rank, **kwargs)
        core.forward = types.MethodType(_patched_core_forward, core)
    if method == "memba_causal":
        model.forward = types.MethodType(_membrane_decoder_forward, model)
    receipt = {"schema": "mamba3_research_adapter_v2", "method": method, "rank": rank,
               "status": "functional_custom_port_requires_validation", "functional": True,
               "port": "state_offset_m3_port" if method == "state_offset" else "memba_causal_custom_port",
               "trainable_paths": [n for n, p in model.named_parameters() if p.requires_grad],
               "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "window_reset": True, "upstream_reproduction": False,
               "deviations": "M3 normalized, biased, rotated C with BF16 coordinates and PyTorch trigonometry" if method == "state_offset" else "per-token causal integration, fixed tau=2, same-time cross-layer transfer, explicit window reset",
               "upstream_commit": "6a0a7247cc8905d01c70089f620d46d565c259d2" if method == "state_offset" else "99f8401cf8891affaba57be85f9a27c4f728d31c"}
    json.dumps(receipt)
    return receipt
