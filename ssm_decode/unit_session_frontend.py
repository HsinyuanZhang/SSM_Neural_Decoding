"""Foldable, explicit unit-direction front ends for session-conditioned decoders."""
from __future__ import annotations

import copy

import torch
from torch import nn


class UnitSessionBank(nn.Module):
    """Store per-session channel maps, latent offsets, and optional residual directions."""

    def __init__(
        self,
        num_sessions: int,
        input_size: int,
        width: int,
        output_size: int,
        *,
        unit_residual: bool = True,
        session_readout: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                   for v in (num_sessions, input_size, width, output_size)):
            raise ValueError("num_sessions, input_size, width, and output_size must be positive integers")
        if not isinstance(unit_residual, bool) or not isinstance(session_readout, bool):
            raise ValueError("unit_residual and session_readout must be bool")
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be floating point")
        self.num_sessions = num_sessions
        self.input_size = input_size
        self.width = width
        self.output_size = output_size
        self.unit_residual = unit_residual
        self.session_readout = session_readout
        self.gain = nn.Parameter(torch.ones(num_sessions, input_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_sessions, input_size, device=device, dtype=dtype))
        self.embedding = nn.Parameter(torch.zeros(num_sessions, width, device=device, dtype=dtype))
        if unit_residual:
            self.unit_delta = nn.Parameter(torch.zeros(num_sessions, width, input_size, device=device, dtype=dtype))
        else:
            self.register_parameter("unit_delta", None)
        if session_readout:
            self.readout_log_gain = nn.Parameter(torch.zeros(num_sessions, output_size, device=device, dtype=dtype))
            self.readout_bias = nn.Parameter(torch.zeros(num_sessions, output_size, device=device, dtype=dtype))
        else:
            self.register_parameter("readout_log_gain", None)
            self.register_parameter("readout_bias", None)

    def _indices(self, session_index, batch: int, device: torch.device) -> torch.Tensor:
        index = torch.as_tensor(session_index, device=device)
        if index.ndim == 0:
            index = index.expand(batch)
        elif index.ndim != 1 or index.numel() != batch:
            raise ValueError("session_index must be scalar or have shape [B]")
        if index.dtype == torch.bool:
            raise ValueError("session_index must contain finite integers")
        if index.is_floating_point():
            if not torch.isfinite(index).all().item() or not torch.equal(index, index.round()):
                raise ValueError("session_index must contain finite integers")
        elif index.is_complex():
            raise ValueError("session_index must contain finite integers")
        index = index.to(torch.long)
        if (index < 0).any().item() or (index >= self.num_sessions).any().item():
            raise ValueError("session_index is out of range")
        return index

    def target_bank(self) -> "UnitSessionBank":
        """Return an independent one-session bank containing source parameter means."""
        target = UnitSessionBank(
            1, self.input_size, self.width, self.output_size,
            unit_residual=self.unit_residual, session_readout=self.session_readout,
            device=self.gain.device, dtype=self.gain.dtype,
        )
        with torch.no_grad():
            target.gain.copy_(self.gain.mean(0, keepdim=True))
            target.bias.copy_(self.bias.mean(0, keepdim=True))
            target.embedding.copy_(self.embedding.mean(0, keepdim=True))
            if self.unit_residual:
                target.unit_delta.copy_(self.unit_delta.mean(0, keepdim=True))
            if self.session_readout:
                # Average the effective positive gains, not their log coordinates.
                target.readout_log_gain.copy_(self.readout_log_gain.exp().mean(0, keepdim=True).log())
                target.readout_bias.copy_(self.readout_bias.mean(0, keepdim=True))
        return target


def _validate_x(x: torch.Tensor, bank: UnitSessionBank) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[-1] != bank.input_size:
        raise ValueError("x must have shape [B,T,input_size]")
    if not x.is_floating_point() or not torch.isfinite(x).all().item():
        raise ValueError("x must be finite floating point")


def _linear_with_bias(linear: nn.Linear) -> nn.Linear:
    return nn.Linear(linear.in_features, linear.out_features, bias=True,
                     device=linear.weight.device, dtype=linear.weight.dtype)


def _session_values(bank: UnitSessionBank, session_index):
    if not isinstance(bank, UnitSessionBank):
        raise TypeError("bank must be a UnitSessionBank")
    if session_index is None:
        gain, bias, embedding = bank.gain.mean(0), bank.bias.mean(0), bank.embedding.mean(0)
        delta = bank.unit_delta.mean(0) if bank.unit_residual else None
        log_gain = bank.readout_log_gain.exp().mean(0).log() if bank.session_readout else None
        readout_bias = bank.readout_bias.mean(0) if bank.session_readout else None
        return gain, bias, embedding, delta, log_gain, readout_bias
    index = bank._indices(session_index, 1, bank.gain.device)[0]
    delta = bank.unit_delta[index] if bank.unit_residual else None
    log_gain = bank.readout_log_gain[index] if bank.session_readout else None
    readout_bias = bank.readout_bias[index] if bank.session_readout else None
    return bank.gain[index], bank.bias[index], bank.embedding[index], delta, log_gain, readout_bias


class UnitSessionDecoder(nn.Module):
    """Run an existing bin-level decoder after an explicit session-specific front end."""

    def __init__(self, base: nn.Module, bank: UnitSessionBank):
        super().__init__()
        if not isinstance(bank, UnitSessionBank):
            raise TypeError("bank must be a UnitSessionBank")
        _validate_base(base, bank)
        self.base = base
        self.bank = bank
        if hasattr(base, "config"):
            self.config = base.config

    def forward(self, x: torch.Tensor, session_index=0):
        _validate_x(x, self.bank)
        index = self.bank._indices(session_index, x.shape[0], x.device)
        gain = self.bank.gain[index].to(x).unsqueeze(1)
        bias = self.bank.bias[index].to(x).unsqueeze(1)
        z = self.base.in_proj(x * gain + bias)
        z = z + self.bank.embedding[index].to(z).unsqueeze(1)
        if self.bank.unit_residual:
            # This residual deliberately acts on the original normalized channels.
            z = z + torch.einsum("btc,bdc->btd", x, self.bank.unit_delta[index].to(x))
        for block in self.base.blocks:
            z = block(z)
        y = self.base.out_proj(self.base.final_norm(z))
        if self.bank.session_readout:
            scale = self.bank.readout_log_gain[index].to(y).exp().unsqueeze(1)
            y = y * scale + self.bank.readout_bias[index].to(y).unsqueeze(1)
        return y


def _validate_base(base: nn.Module, bank: UnitSessionBank) -> None:
    if not isinstance(base, nn.Module):
        raise TypeError("base must be an nn.Module")
    for name in ("in_proj", "blocks", "final_norm", "out_proj"):
        if not hasattr(base, name):
            raise ValueError("base must have in_proj, blocks, final_norm, and out_proj")
    if not isinstance(base.in_proj, nn.Linear) or not isinstance(base.out_proj, nn.Linear):
        raise ValueError("base in_proj and out_proj must be nn.Linear")
    if base.in_proj.in_features != bank.input_size or base.in_proj.out_features != bank.width:
        raise ValueError("base in_proj dimensions do not match bank")
    if base.out_proj.in_features != bank.width or base.out_proj.out_features != bank.output_size:
        raise ValueError("base out_proj dimensions do not match bank")


def fold_unit_session(base: nn.Module, bank: UnitSessionBank, session_index=None) -> nn.Module:
    """Deep-copy ``base`` and fold one session (or source means) into its projections."""
    _validate_base(base, bank)
    gain, offset, embedding, delta, log_gain, readout_bias = _session_values(bank, session_index)
    folded = copy.deepcopy(base)
    old_in = folded.in_proj
    old_out = folded.out_proj
    new_in = _linear_with_bias(old_in)
    new_out = _linear_with_bias(old_out)
    with torch.no_grad():
        weight = old_in.weight.detach()
        folded_weight = weight * gain.detach().to(weight).unsqueeze(0)
        if delta is not None:
            folded_weight = folded_weight + delta.detach().to(weight)
        new_in.weight.copy_(folded_weight)
        base_bias = torch.zeros(old_in.out_features, device=weight.device, dtype=weight.dtype)
        if old_in.bias is not None:
            base_bias = old_in.bias.detach()
        new_in.bias.copy_(weight @ offset.detach().to(weight) + base_bias + embedding.detach().to(weight))
        out_weight = old_out.weight.detach()
        out_bias = torch.zeros(old_out.out_features, device=out_weight.device, dtype=out_weight.dtype)
        if old_out.bias is not None:
            out_bias = old_out.bias.detach()
        if log_gain is not None:
            scale = log_gain.detach().to(out_weight).exp()
            new_out.weight.copy_(out_weight * scale.unsqueeze(1))
            new_out.bias.copy_(out_bias * scale + readout_bias.detach().to(out_weight))
        else:
            new_out.weight.copy_(out_weight)
            new_out.bias.copy_(out_bias)
    folded.in_proj = new_in
    folded.out_proj = new_out
    return folded


def unit_delta_penalty(bank: UnitSessionBank) -> torch.Tensor:
    """Mean-square regularizer for explicit unit-direction residuals."""
    if not isinstance(bank, UnitSessionBank):
        raise TypeError("bank must be a UnitSessionBank")
    if bank.unit_delta is None:
        return torch.zeros((), device=bank.gain.device, dtype=bank.gain.dtype)
    return bank.unit_delta.square().mean()
