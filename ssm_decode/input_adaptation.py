"""Input affine, projection LoRA, and state-offset adapter configurations."""
from __future__ import annotations

import json
import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from .peft import LoRALinear, configure_peft


_METHODS = {
    "none", "io", "lora", "affine", "affine_lora", "offset_rotated",
    "offset_original", "full",
}


class ChannelAffine(nn.Module):
    """Apply one gain and one bias to each decoder input channel."""

    def __init__(self, channels, *, device=None, dtype=None):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.gain = nn.Parameter(torch.ones(channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(channels, device=device, dtype=dtype))

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.gain.numel():
            raise ValueError("input channel count does not match ChannelAffine")
        return x * self.gain + self.bias


class StateOffset(nn.Module):
    """Map a low-rank state offset through a readout coordinate tensor."""

    def __init__(self, heads, channels, state_size, rank, *, device=None, dtype=None):
        super().__init__()
        if not all(isinstance(value, int) and value > 0
                   for value in (heads, channels, state_size, rank)):
            raise ValueError("offset dimensions and rank must be positive integers")
        self.heads = heads
        self.channels = channels
        self.state_size = state_size
        self.U = nn.Parameter(torch.zeros(heads * channels, rank, device=device, dtype=dtype))
        self.V = nn.Parameter(torch.empty(rank, state_size, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, readout):
        offset = (self.U @ self.V).reshape(self.heads, self.channels, self.state_size)
        return torch.einsum("bthn,hpn->bthp", readout, offset)


def _siso_kernel(**kwargs):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    return mamba3_siso_combined(**kwargs)


def normalized_readout(c, bias):
    """Return normalized C plus C_bias in per-head original coordinates."""
    heads = bias.shape[0]
    if heads % c.shape[2]:
        raise ValueError("C head count does not divide the Mamba-3 head count")
    return c.to(torch.bfloat16).float().repeat_interleave(heads // c.shape[2], dim=2) + bias.float()[None, None]


def original_readout(c, bias):
    """Return the original readout coordinates at the same BF16 boundary as RoPE."""
    return normalized_readout(c, bias).to(torch.bfloat16).float()


def rotated_readout(c, bias, angles, dt):
    """Return the prior effective Mamba-3 rotated readout coordinates."""
    q = normalized_readout(c, bias)
    rates = torch.tanh(angles.to(torch.bfloat16).float()) * math.pi
    theta = torch.cumsum(rates * dt.float()[..., None], dim=1)
    theta = torch.remainder(theta, 2 * math.pi).to(torch.bfloat16).float()
    count = theta.shape[-1]
    if 2 * count > q.shape[-1]:
        raise ValueError("RoPE angle count exceeds C state size")
    pairs = q[..., :2 * count].reshape(*q.shape[:-1], count, 2)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    first = pairs[..., 0] * cosine - pairs[..., 1] * sine
    second = pairs[..., 0] * sine + pairs[..., 1] * cosine
    rotated = torch.stack((first, second), dim=-1).flatten(-2)
    return torch.cat((rotated, q[..., 2 * count:]), dim=-1).to(torch.bfloat16).float()


def _offset_core_forward(core, u):
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
    y = _siso_kernel(Q=c.squeeze(2), K=b.squeeze(2), V=x, ADT=(a * dt).transpose(1, 2),
                     DT=dt.transpose(1, 2), Trap=trap.transpose(1, 2),
                     Q_bias=core.C_bias.squeeze(1), K_bias=core.B_bias.squeeze(1),
                     Angles=angles, D=core.D, Z=z, chunk_size=core.chunk_size,
                     Input_States=None, return_final_states=False, cu_seqlens=None)
    if core.state_offset_coordinate == "rotated":
        readout = rotated_readout(c.squeeze(2), core.C_bias.squeeze(1), angles, dt)
    else:
        readout = original_readout(c.squeeze(2), core.C_bias.squeeze(1))
    offset = core.state_offset(readout.to(core.state_offset.U.dtype))
    # The fused scan gates its base output. Apply the new offset gate one time.
    y = y.float() + F.silu(z.to(torch.bfloat16).float()) * offset.float()
    return core.out_proj(y.flatten(-2).to(x.dtype))


def _patched_offset_forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
    if seq_idx is not None or cu_seqlens is not None or inference_params is not None:
        raise ValueError("state-offset ports support independent full windows only")
    return _offset_core_forward(self, u)


def _root_linear(model):
    root = getattr(model, "in_proj", None)
    return root.base if isinstance(root, LoRALinear) else root


def _root_input_hook(model, inputs):
    if len(inputs) != 1:
        raise ValueError("decoder input hook requires one input tensor")
    return (model.input_adaptation(inputs[0]),)


def attach_input_adaptation_hook(model):
    """Attach the serializable input affine module to the decoder root input."""
    adapter = getattr(model, "input_adaptation", None)
    if not isinstance(adapter, ChannelAffine):
        raise ValueError("model has no ChannelAffine input adaptation module")
    prior = getattr(model, "_input_adaptation_hook", None)
    if prior is not None:
        prior.remove()
    model._input_adaptation_hook = model.register_forward_pre_hook(_root_input_hook)
    return model._input_adaptation_hook


def _remove_lora(model, paths):
    for path in paths:
        parent = model
        bits = path.split(".")
        for bit in bits[:-1]:
            parent = parent[int(bit)] if bit.isdigit() else getattr(parent, bit)
        module = getattr(parent, bits[-1])
        if not isinstance(module, LoRALinear):
            raise ValueError("expected LoRA wrapper is absent")
        setattr(parent, bits[-1], module.base)


def _configure_lora(model, rank, alpha, scope, train_input_bias):
    receipt = configure_peft(model, "lora", rank=rank, alpha=alpha)
    root_paths = ["in_proj", "out_proj"]
    input_paths = ["in_proj"]
    core_paths = [path for path in receipt["lora_paths"] if path not in root_paths]
    if scope == "root":
        _remove_lora(model, core_paths)
    elif scope == "core":
        _remove_lora(model, root_paths)
    elif scope == "input":
        _remove_lora(model, [path for path in receipt["lora_paths"] if path not in input_paths])
    selected = [path for path in receipt["lora_paths"] if
                (scope == "all" or (scope == "root" and path in root_paths) or
                 (scope == "core" and path in core_paths) or
                 (scope == "input" and path in input_paths))]
    if scope == "core" or not train_input_bias:
        return selected, False
    root = model.in_proj
    if root.base.bias is None:
        return selected, False
    root.base.bias.requires_grad_(True)
    return selected, True


def _validate_offset_core(core):
    if (getattr(core, "is_mimo", None) is not False or
            getattr(core, "is_outproj_norm", None) is not False or
            getattr(core, "mimo_rank", None) != 1):
        raise ValueError("state-offset ports require SISO without output normalization")
    if not isinstance(getattr(core, "in_proj", None), nn.Linear) or not isinstance(getattr(core, "out_proj", None), nn.Linear):
        raise ValueError("state-offset ports require native Linear core projections")
    if hasattr(core, "state_offset"):
        raise ValueError("state-offset adapter is already installed")


def _freeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _root_io(model):
    for name, parameter in model.named_parameters():
        if name.rsplit(".", 1)[0] in {"in_proj", "out_proj"} or name.startswith("final_norm."):
            parameter.requires_grad_(True)


def configure_adaptation(model, method, rank=4, alpha=4, lora_scope="all", train_input_bias=True):
    """Configure one auditable input-adaptation method on an official decoder."""
    if method not in _METHODS:
        raise ValueError("unknown adaptation method")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("alpha must be positive")
    if lora_scope not in {"all", "root", "core", "input"}:
        raise ValueError("lora_scope must be all, root, core, or input")
    if not isinstance(train_input_bias, bool):
        raise ValueError("train_input_bias must be a bool")
    if hasattr(model, "input_adaptation") or any(hasattr(block.ssm, "state_offset") for block in model.blocks):
        raise ValueError("an input adaptation is already installed")
    root = _root_linear(model)
    if not isinstance(root, nn.Linear):
        raise ValueError("decoder root in_proj must be Linear")
    if method in {"offset_rotated", "offset_original"}:
        for block in model.blocks:
            _validate_offset_core(block.ssm)
    _freeze_all(model)
    coordinate = None
    lora_paths = []
    root_bias_trainable = False
    if method in {"io", "full"}:
        _root_io(model)
    elif method in {"lora", "affine_lora"}:
        lora_paths, root_bias_trainable = _configure_lora(model, rank, alpha, lora_scope, train_input_bias)
    if method in {"affine", "affine_lora", "offset_rotated", "offset_original"}:
        root = _root_linear(model)
        model.input_adaptation = ChannelAffine(model.config.input_size, device=root.weight.device, dtype=root.weight.dtype)
        attach_input_adaptation_hook(model)
    if method in {"offset_rotated", "offset_original"}:
        coordinate = "rotated" if method == "offset_rotated" else "original"
        for block in model.blocks:
            core = block.ssm
            _validate_offset_core(core)
            core.state_offset = StateOffset(core.nheads, core.headdim, core.d_state, rank,
                                            device=core.in_proj.weight.device,
                                            dtype=core.in_proj.weight.dtype)
            core.state_offset_coordinate = coordinate
            core.forward = types.MethodType(_patched_offset_forward, core)
    named = list(model.named_parameters(remove_duplicate=False))
    trainable = [name for name, parameter in named if parameter.requires_grad]
    receipt = {
        "schema": "mamba3_input_adaptation_v1",
        "method": method,
        "rank": rank,
        "alpha": alpha,
        "lora_scope": lora_scope,
        "coordinate": coordinate,
        "lora_paths": lora_paths,
        "train_input_bias": train_input_bias,
        "root_input_bias_trainable": root_bias_trainable,
        "root_bias_behavior": "root in_proj base bias is trainable" if root_bias_trainable else "root in_proj base bias is frozen",
        "custom_port_boundary": ("independent full SISO windows only; no seq_idx, cu_seqlens, or inference_params"
                                 if coordinate else None),
        "trainable_paths": trainable,
        "trainable_count": sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
    }
    json.dumps(receipt)
    return receipt


def merged_adapted_state_dict(model):
    """Export a source-key state dictionary with affine and LoRA folded into projections."""
    if any(hasattr(block.ssm, "state_offset") for block in model.blocks):
        raise ValueError("state-offset adapters cannot be folded into source projection keys")
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "." if name else ""
            state[prefix + "weight"] = (module.base.weight + module.delta()).detach().clone()
            if module.base.bias is not None:
                state[prefix + "bias"] = module.base.bias.detach().clone()
    ignored = (".base.", "input_adaptation.")
    for key, value in model.state_dict().items():
        if (not any(token in key for token in ignored) and
                not key.endswith((".A", ".B", ".rows"))):
            state.setdefault(key, value.detach().clone())
    affine = getattr(model, "input_adaptation", None)
    if isinstance(affine, ChannelAffine):
        weight = state["in_proj.weight"]
        bias = state.get("in_proj.bias")
        state["in_proj.weight"] = weight * affine.gain.detach().to(weight).unsqueeze(0)
        folded_bias = weight @ affine.bias.detach().to(weight)
        state["in_proj.bias"] = folded_bias if bias is None else bias + folded_bias
    return state


merged_state_dict = merged_adapted_state_dict
