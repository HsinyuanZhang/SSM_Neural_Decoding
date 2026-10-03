"""Bit-exact fixed-point reference for a real 2x2 damped SSM recurrence.

The integer core uses signed Q1.14 coefficients and signed integer state/drive
values.  There is deliberately no floating-point arithmetic in ``step``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, floor
from pathlib import Path
from typing import Iterable, Sequence

Q_FRAC = 14
Q_SCALE = 1 << Q_FRAC
COEFF_MIN = -(1 << 15)
COEFF_MAX = (1 << 15) - 1


def _round_ties_away_div_q14(value: int) -> int:
    """Return value / 2**14 rounded to nearest with half cases away from zero."""
    magnitude = abs(int(value))
    quotient = (magnitude + (Q_SCALE // 2)) >> Q_FRAC
    return quotient if value >= 0 else -quotient


def _saturate(value: int, bits: int) -> tuple[int, bool]:
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def spectral_radius_2x2_q14(a: Sequence[int]) -> float:
    """Spectral radius of the de-quantized real 2x2 matrix (validation only)."""
    if len(a) != 4:
        raise ValueError("A must contain (a00, a01, a10, a11)")
    a00, a01, a10, a11 = (x / Q_SCALE for x in a)
    trace = a00 + a11
    determinant = a00 * a11 - a01 * a10
    discriminant = trace * trace - 4.0 * determinant
    if discriminant >= 0:
        root = discriminant**0.5
        return max(abs((trace + root) / 2.0), abs((trace - root) / 2.0))
    # Complex conjugate roots have squared magnitude equal to determinant.
    return max(0.0, determinant) ** 0.5


def quantize_q14(value: float) -> int:
    """Quantize a coefficient with the same nearest/ties-away rule as RTL."""
    scaled = value * Q_SCALE
    quantized = floor(scaled + 0.5) if scaled >= 0 else ceil(scaled - 0.5)
    if not COEFF_MIN <= quantized <= COEFF_MAX:
        raise ValueError(f"coefficient {value} is outside signed Q1.14 range")
    return int(quantized)


def quantize_stable_matrix(a_float: Sequence[float], *, margin: float = 1e-4) -> tuple[int, int, int, int]:
    """Quantize A and reject it unless its post-quantization radius is strictly < 1."""
    if len(a_float) != 4:
        raise ValueError("A must contain four coefficients")
    result = tuple(quantize_q14(x) for x in a_float)
    radius = spectral_radius_2x2_q14(result)
    if radius >= 1.0 - margin:
        raise ValueError(f"unsafe post-quantization spectral radius {radius:.9f}")
    return result  # type: ignore[return-value]


@dataclass
class Real2x2FixedPoint:
    """One recurrence mode: x_next = round_q14(A*x + B*u), then saturate."""

    a_q14: tuple[int, int, int, int]
    b_q14: tuple[int, int] = (0, Q_SCALE)
    state_bits: int = 16
    p: int = 0
    q: int = 0
    saturation_count: int = 0

    def __post_init__(self) -> None:
        if self.state_bits not in (16, 24, 32):
            raise ValueError("state_bits must be one of 16, 24, 32")
        if any(not COEFF_MIN <= x <= COEFF_MAX for x in (*self.a_q14, *self.b_q14)):
            raise ValueError("all A/B coefficients must be signed Q1.14 int16")
        if spectral_radius_2x2_q14(self.a_q14) >= 1.0:
            raise ValueError("post-quantization A is not strictly stable")
        self.p, p_sat = _saturate(self.p, self.state_bits)
        self.q, q_sat = _saturate(self.q, self.state_bits)
        self.saturation_count += int(p_sat) + int(q_sat)

    def step(self, u: int = 0) -> tuple[int, int]:
        """Advance exactly once. ``u`` is an integer in state-value units."""
        a00, a01, a10, a11 = self.a_q14
        b0, b1 = self.b_q14
        p_wide = a00 * self.p + a01 * self.q + b0 * int(u)
        q_wide = a10 * self.p + a11 * self.q + b1 * int(u)
        p_next, p_sat = _saturate(_round_ties_away_div_q14(p_wide), self.state_bits)
        q_next, q_sat = _saturate(_round_ties_away_div_q14(q_wide), self.state_bits)
        self.p, self.q = p_next, q_next
        self.saturation_count += int(p_sat) + int(q_sat)
        return self.p, self.q

    def export_layout(self) -> dict:
        """Export only integer hardware-visible quantities and register layout."""
        return {
            "format": {"coefficient": "signed Q1.14 int16", "state": f"signed int{self.state_bits}",
                       "rounding": "nearest, ties away from zero", "saturation": "symmetric signed clamp"},
            "a_q14": list(self.a_q14), "b_q14": list(self.b_q14),
            "spectral_radius": spectral_radius_2x2_q14(self.a_q14),
            "register_layout": [
                {"name": "p", "bits": self.state_bits, "signed": True},
                {"name": "q", "bits": self.state_bits, "signed": True},
                {"name": "saturation_count", "bits": 32, "signed": False},
            ],
            "state": {"p": self.p, "q": self.q, "saturation_count": self.saturation_count},
        }

    def export_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.export_layout(), indent=2, sort_keys=True) + "\n")


@dataclass
class Real2x2ModeBank:
    """Independent fixed-point modes; accepts the per-mode A/B matrix arrays used by a bank."""

    a_q14_modes: Sequence[Sequence[int]]
    b_q14_modes: Sequence[Sequence[int]] | None = None
    state_bits: int = 16

    def __post_init__(self) -> None:
        if self.b_q14_modes is not None and len(self.a_q14_modes) != len(self.b_q14_modes):
            raise ValueError("A and B mode arrays must have equal length")
        default_b = (0, Q_SCALE)
        self.modes = [Real2x2FixedPoint(tuple(a), tuple(self.b_q14_modes[i]) if self.b_q14_modes else default_b,
                                        self.state_bits)
                      for i, a in enumerate(self.a_q14_modes)]

    def step(self, drives: Iterable[int]) -> list[tuple[int, int]]:
        drives = list(drives)
        if len(drives) != len(self.modes):
            raise ValueError("one integer drive is required per mode")
        return [mode.step(u) for mode, u in zip(self.modes, drives)]

    def export_layout(self) -> dict:
        return {"mode_count": len(self.modes), "modes": [mode.export_layout() for mode in self.modes]}


def run_zero_input_characterization(mode: Real2x2FixedPoint, steps: int = 100_000) -> dict:
    """Empirical fixed-point check; it is a characterization, never a stability proof."""
    peak = max(abs(mode.p), abs(mode.q))
    seen: dict[tuple[int, int], int] = {}
    first_repeat: tuple[int, int] | None = None
    for index in range(steps):
        state = (mode.p, mode.q)
        if state in seen and first_repeat is None:
            first_repeat = (seen[state], index)
        else:
            seen.setdefault(state, index)
        mode.step(0)
        peak = max(peak, abs(mode.p), abs(mode.q))
    return {"steps_requested": steps, "steps_executed": steps, "unique_states": len(seen), "peak_abs_state": peak,
            "saturation_count": mode.saturation_count, "repeat": first_repeat,
            "final_state": [mode.p, mode.q],
            "interpretation": "finite-word empirical characterization, not a proof of absence of overflow or limit cycles"}
