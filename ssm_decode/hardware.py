"""Transparent accounting and fake quantization helpers; no hardware claims."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from .models import CausalSSMDecoder, ModelConfig


def fake_quantize(x: Tensor, bits: int = 8, scale: float | Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Symmetric fake quantization with a calibrated (or supplied) floating scale."""
    if bits < 2: raise ValueError("bits must be >= 2")
    qmax = (1 << (bits - 1)) - 1
    if scale is None:
        scale = x.detach().abs().amax().clamp_min(torch.finfo(x.dtype).eps) / qmax
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    return (torch.round(x / scale).clamp(-qmax, qmax) * scale, scale)


def calibrate_scale(samples: Tensor, bits: int = 8) -> Tensor:
    """Calibrate one symmetric scale from source/target-support samples only.

    Persist the returned scale and pass it back to :func:`fake_quantize` during
    query inference; this helper intentionally has no dynamic query-scale mode.
    """
    _, scale = fake_quantize(samples, bits=bits)
    return scale.detach()


def fake_quantize_per_row(weight: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    """W8-style per-output-row fake quantization for a matrix weight."""
    if weight.ndim != 2:
        raise ValueError("per-row quantization expects a rank-2 matrix")
    qmax = (1 << (bits - 1)) - 1
    scales = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps) / qmax
    return fake_quantize(weight, bits=bits, scale=scales)


def resource_accounting(model: CausalSSMDecoder, dtype_bytes: int = 4) -> dict[str, Any]:
    c = model.config
    if c.frontend == "profile":
        frontend, projection_weights = c.input_size * c.width, (c.profile_size or 0) * c.width
    else:
        frontend = c.input_size * (c.projection_rank or c.width) + ((c.projection_rank or 0) * c.width)
        projection_weights = frontend
    recurrent = {"diag": c.width, "selective": 2 * c.width, "osc": 4 * c.width,
                 "bank": 4 * c.width, "gru": 6 * c.width * c.width}[c.kind]
    state_values = c.width * (2 if c.kind in {"osc", "bank"} else 1)
    return {"kind": c.kind, "macs_per_step": {"frontend_projection": frontend,
            "state_recurrence": recurrent, "readout": c.width * c.output_size,
            "total": frontend + recurrent + c.width * c.output_size},
            "state_bytes": state_values * dtype_bytes,
            "state_values_allocated": state_values,
            "bank_redundant_second_values": int((~model.bank_is_osc).sum()) if c.kind == "bank" else 0,
            "projection_weight_bytes": projection_weights * dtype_bytes,
            "dtype_bytes": dtype_bytes,
            "nonlinear_ops_per_step": {"selective_gate_sigmoid_and_pow": 1 if c.kind == "selective" and c.rate_gate else 0},
            "note": "MAC categories are arithmetic estimates; fake quantization is not an integer-chip claim."}


def export_recurrence_coefficients(model: CausalSSMDecoder) -> dict[str, Any]:
    """Return a JSON-friendly frozen/current recurrence-coefficient schema."""
    matrices = model.recurrence_matrices().detach().cpu().tolist()
    return {"schema_version": 1, "kind": model.config.kind, "real_valued": True,
            "state_layout": "[batch,width,2]" if model.config.kind in {"osc", "bank"} else "[batch,width]",
            "matrices_2x2": matrices,
            "frozen": model.config.kind == "bank",
            "power_of_two_decay": model.config.power_of_two_decay,
            "power_of_two_convention": "retention = 1 - 2**(-k), k in [1,16]" if model.config.power_of_two_decay else None}
