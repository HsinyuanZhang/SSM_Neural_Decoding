"""Bounded, real-valued causal decoder models.

All models accept ``x`` shaped ``[batch, time, channels]``.  ``step`` consumes
one sample and returns ``(prediction, next_state)``; consequently a state can be
passed from one chunk to the next without looking at later samples.  State is
``[B, width]`` for diagonal/selective/GRU models and ``[B, width, 2]`` for
oscillator and bank models.  In the latter the first coordinate is the exposed
latent; a real bank entry simply keeps its second coordinate at zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_size: int
    output_size: int
    width: int = 64
    kind: Literal["diag", "osc", "bank", "selective", "gru"] = "diag"
    frontend: Literal["linear", "fixed", "profile"] = "linear"
    profile_size: int | None = None
    projection_rank: int | None = None
    group_size: int = 8
    power_of_two_decay: bool = False
    rate_gate: bool = True
    bias: bool = True


def _power_two_decay(value: Tensor) -> Tensor:
    """Use stable shift-subtract retention ``1 - 2**-k`` (k=1..16).

    This is a power-of-two *leak*, rather than a retention rounded to one; it
    remains strictly inside the unit disk and can be implemented as a shift and
    subtraction in fixed-point arithmetic.
    """
    leak = (1 - value).clamp(min=2.0 ** -16, max=0.5)
    k = torch.round(-torch.log2(leak)).clamp(1, 16)
    return 1 - torch.pow(torch.tensor(2.0, device=value.device, dtype=value.dtype), -k)


class _Frontend(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fixed = cfg.frontend == "fixed"
        self.rank = cfg.projection_rank
        if self.fixed:
            # Foldable deterministic projection: no data-dependent fitting, and
            # therefore usable with channel/profile association feature vectors.
            c = torch.arange(cfg.input_size, dtype=torch.float32)[:, None] + 1
            w = torch.arange(cfg.width, dtype=torch.float32)[None, :] + 1
            matrix = (torch.sin(c * w * 0.61803398875) + torch.cos(c * w * 0.41421356237))
            matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-6)
            self.register_buffer("matrix", matrix)
        elif self.rank is not None:
            if not 0 < self.rank <= min(cfg.input_size, cfg.width):
                raise ValueError("projection_rank must be in [1, min(input_size, width)]")
            self.first = nn.Linear(cfg.input_size, self.rank, bias=False)
            self.second = nn.Linear(self.rank, cfg.width, bias=cfg.bias)
        else:
            self.linear = nn.Linear(cfg.input_size, cfg.width, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.fixed:
            return x @ self.matrix.to(dtype=x.dtype)
        if self.rank is not None:
            return self.second(self.first(x))
        return self.linear(x)


class FoldedProfileFrontend(nn.Module):
    """Permutation-invariant unit-profile frontend for padded, variable unit sets.

    ``profiles`` is ``[B, C, profile_size]`` and must contain only information
    available from the support/session association profile.  It is turned into
    static per-unit weights before the temporal recurrence: ``x @ fold(profile)``.
    Reordering neural channels and their profiles together leaves the result
    unchanged.  ``unit_mask`` excludes padded/missing units exactly.
    """
    def __init__(self, profile_size: int, width: int):
        super().__init__()
        if profile_size < 1:
            raise ValueError("profile_size must be positive for frontend='profile'")
        p = torch.arange(profile_size, dtype=torch.float32)[:, None] + 1
        w = torch.arange(width, dtype=torch.float32)[None, :] + 1
        basis = torch.sin(p * w * 0.754877666) + torch.cos(p * w * 0.569840291)
        self.register_buffer("profile_basis", basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6))

    def fold(self, profiles: Tensor, unit_mask: Tensor | None = None) -> Tensor:
        """Return frozen static weights ``[B,C,width]`` for export/inspection."""
        if profiles.ndim != 3 or profiles.shape[-1] != self.profile_basis.shape[0]:
            raise ValueError("profiles must be [batch, channels, profile_size]")
        folded = profiles @ self.profile_basis.to(dtype=profiles.dtype)
        if unit_mask is not None:
            if unit_mask.shape != profiles.shape[:2]:
                raise ValueError("unit_mask must be [batch, channels]")
            folded = folded * unit_mask.to(dtype=folded.dtype).unsqueeze(-1)
        return folded

    def forward(self, x: Tensor, profiles: Tensor | None = None,
                unit_mask: Tensor | None = None) -> Tensor:
        if profiles is None:
            raise ValueError("profile frontend requires support-derived profiles")
        if x.ndim != 2 or profiles.shape[:2] != x.shape:
            raise ValueError("x must be [batch, channels] aligned with profiles")
        # Pool across the unordered unit axis before applying the temporal model.
        return torch.einsum("bc,bcw->bw", x, self.fold(profiles, unit_mask))


class CausalSSMDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.kind not in {"diag", "osc", "bank", "selective", "gru"}:
            raise ValueError(f"unknown decoder kind: {config.kind}")
        self.config = config
        if config.frontend == "profile":
            if config.profile_size is None:
                raise ValueError("profile_size is required for frontend='profile'")
            self.frontend = FoldedProfileFrontend(config.profile_size, config.width)
        else:
            self.frontend = _Frontend(config)
        self.readout = nn.Linear(config.width, config.output_size, bias=config.bias)
        w = config.width
        if config.kind in {"diag", "selective"}:
            self.log_rate = nn.Parameter(torch.full((w,), -2.25))
            if config.kind == "selective":
                self.rate_gate = nn.Linear(1, 1, bias=True) if config.rate_gate else None
        elif config.kind == "osc":
            self.log_rate = nn.Parameter(torch.full((w,), -2.7))
            self.angle = nn.Parameter(torch.linspace(0.10, 1.10, w))
        elif config.kind == "bank":
            self._init_bank()
        else:
            self.gru = nn.GRUCell(w, w, bias=config.bias)

    def _decay(self, log_rate: Tensor) -> Tensor:
        value = torch.exp(-F.softplus(log_rate))  # strictly inside the unit disk
        return _power_two_decay(value) if self.config.power_of_two_decay else value

    def _init_bank(self) -> None:
        """Frozen grouped codebook of stable real and oscillator coefficients."""
        w, gs = self.config.width, max(1, self.config.group_size)
        group = torch.arange(w) // gs
        code = group % 4
        is_osc = (group % 2 == 1)
        rates = torch.tensor([0.82, 0.88, 0.93, 0.97])
        if self.config.power_of_two_decay:
            rates = _power_two_decay(rates)
        angles = torch.tensor([0.18, 0.43, 0.76, 1.12])
        a = torch.zeros(w, 2, 2)
        r = rates[code]
        th = angles[code]
        a[:, 0, 0] = r
        a[is_osc, 0, 0] = r[is_osc] * torch.cos(th[is_osc])
        a[is_osc, 0, 1] = -r[is_osc] * torch.sin(th[is_osc])
        a[is_osc, 1, 0] = r[is_osc] * torch.sin(th[is_osc])
        a[is_osc, 1, 1] = r[is_osc] * torch.cos(th[is_osc])
        self.register_buffer("bank_A", a)
        self.register_buffer("bank_is_osc", is_osc)
        self.register_buffer("bank_code", code)

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> Tensor:
        device = device if device is not None else self.readout.weight.device
        shape = (batch, self.config.width, 2) if self.config.kind in {"osc", "bank"} else (batch, self.config.width)
        return torch.zeros(*shape, device=device, dtype=self.readout.weight.dtype)

    def recurrence_matrices(self) -> Tensor:
        """Return per-latent 2x2 real recurrence matrices for inspection/export."""
        w = self.config.width
        if self.config.kind == "bank":
            return self.bank_A
        if self.config.kind in {"diag", "selective"}:
            a = self._decay(self.log_rate)
            out = torch.zeros(w, 2, 2, device=a.device, dtype=a.dtype)
            out[:, 0, 0] = a
            return out
        if self.config.kind == "osc":
            r, t = self._decay(self.log_rate), self.angle
            out = torch.zeros(w, 2, 2, device=r.device, dtype=r.dtype)
            out[:, 0, 0], out[:, 0, 1] = r * torch.cos(t), -r * torch.sin(t)
            out[:, 1, 0], out[:, 1, 1] = r * torch.sin(t), r * torch.cos(t)
            return out
        return torch.empty(0, 2, 2, device=self.readout.weight.device)

    def _advance(self, u: Tensor, state: Tensor, A: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Advance a projected input once; ``A`` allows a batch forward to cache coefficients."""
        kind = self.config.kind
        if kind == "gru":
            nxt, latent = self.gru(u, state), None
            latent = nxt
        elif kind in {"diag", "selective"}:
            a = self._decay(self.log_rate)
            if kind == "selective" and self.rate_gate is not None:
                # One scalar input-only rate gate: light enough for small decoders.
                gate = torch.sigmoid(self.rate_gate(u.mean(-1, keepdim=True)))
                a = a.unsqueeze(0).pow(gate)
            nxt = a * state + u
            latent = nxt
        else:
            A = self.recurrence_matrices().to(dtype=u.dtype) if A is None else A
            nxt = torch.einsum("wij,bwj->bwi", A, state)
            nxt[..., 0] = nxt[..., 0] + u
            latent = nxt[..., 0]
        return self.readout(latent), nxt

    def step(self, x: Tensor, state: Tensor | None = None, *, profiles: Tensor | None = None,
             unit_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError("step expects x with shape [batch, channels]")
        u = self.frontend(x, profiles, unit_mask) if self.config.frontend == "profile" else self.frontend(x)
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        return self._advance(u, state)

    def forward(self, x: Tensor, state: Tensor | None = None, return_state: bool = False,
                return_latents: bool = False,
                *, profiles: Tensor | None = None, unit_mask: Tensor | None = None):
        if x.ndim != 3:
            raise ValueError("forward expects x with shape [batch, time, channels]")
        state = self.initial_state(x.shape[0], x.device) if state is None else state
        # Project the whole sequence in one GEMM; only the causal recurrence is looped.
        if self.config.frontend == "profile":
            if profiles is None:
                raise ValueError("profile frontend requires support-derived profiles")
            folded = self.frontend.fold(profiles, unit_mask)
            u_all = torch.einsum("btc,bcw->btw", x, folded)
        else:
            u_all = self.frontend(x)
        A = self.recurrence_matrices().to(dtype=x.dtype) if self.config.kind in {"osc", "bank"} else None
        ys, latents = [], []
        for t in range(x.shape[1]):
            y, state = self._advance(u_all[:, t], state, A)
            ys.append(y)
            latents.append(state[..., 0] if state.ndim == 3 else state)
        out = torch.stack(ys, dim=1) if ys else x.new_empty(x.shape[0], 0, self.config.output_size)
        latent_out = torch.stack(latents, dim=1) if latents else x.new_empty(x.shape[0], 0, self.config.width)
        if return_state and return_latents:
            return out, state, latent_out
        if return_state:
            return out, state
        return (out, latent_out) if return_latents else out


def build_model(config: ModelConfig | dict) -> CausalSSMDecoder:
    """Build a decoder from a dataclass or a compatible configuration mapping."""
    return CausalSSMDecoder(ModelConfig(**config) if isinstance(config, dict) else config)
