"""Target-support-only linear calibration utilities.

These routines fit from supplied support ``(x, y)``.  Applying their returned
mapping never consumes query labels.  ``fit_delta`` is intentionally an
approximation; use ridge or RLS when an exact ridge solution is required.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class LinearMapping:
    weight: Tensor                 # [features, outputs]
    intercept: Tensor              # [outputs]
    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.intercept


@dataclass
class StreamingRidge:
    """One-pass centered sufficient statistics for exact ridge fitting.

    Storage is ``d*d + d*o + d + o`` scalar values. Call ``update`` only on
    target support samples, then ``solve(alpha)``; query labels are not read.
    """
    features: int
    outputs: int
    dtype: torch.dtype = torch.float64
    device: torch.device | str | None = None

    def __post_init__(self):
        self.count = 0
        self.mean_x = torch.zeros(self.features, dtype=self.dtype, device=self.device)
        self.mean_y = torch.zeros(self.outputs, dtype=self.dtype, device=self.device)
        self.cxx = torch.zeros(self.features, self.features, dtype=self.dtype, device=self.device)
        self.cxy = torch.zeros(self.features, self.outputs, dtype=self.dtype, device=self.device)

    @property
    def state_scalars(self) -> int:
        return self.features * self.features + self.features * self.outputs + self.features + self.outputs + 1

    def update(self, x: Tensor, y: Tensor) -> "StreamingRidge":
        x, y = _check(x, y)
        if x.shape[1] != self.features or y.shape[1] != self.outputs:
            raise ValueError("streaming dimensions do not match")
        for row, target in zip(x.to(self.mean_x), y.to(self.mean_y)):
            self.count += 1
            dx, dy = row - self.mean_x, target - self.mean_y
            self.mean_x = self.mean_x + dx / self.count
            self.mean_y = self.mean_y + dy / self.count
            self.cxx = self.cxx + torch.outer(dx, row - self.mean_x)
            self.cxy = self.cxy + torch.outer(dx, target - self.mean_y)
        return self

    def solve(self, alpha: float = 1.0) -> LinearMapping:
        if self.count == 0 or alpha < 0:
            raise ValueError("nonempty support and alpha >= 0 required")
        eye = torch.eye(self.features, dtype=self.dtype, device=self.mean_x.device)
        weight = torch.linalg.solve(self.cxx + alpha * eye, self.cxy)
        return LinearMapping(weight, self.mean_y - self.mean_x @ weight)


def _check(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    if x.ndim != 2: raise ValueError("x must be [samples, features]")
    if y.ndim == 1: y = y[:, None]
    if y.ndim != 2 or y.shape[0] != x.shape[0]: raise ValueError("y must align with x samples")
    return x, y


def fit_ridge(x: Tensor, y: Tensor, alpha: float = 1.0) -> LinearMapping:
    x, y = _check(x, y)
    if alpha < 0: raise ValueError("alpha must be nonnegative")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    weight = torch.linalg.solve(xc.T @ xc + alpha * eye, xc.T @ yc)
    return LinearMapping(weight, ym - xm @ weight)


def fit_rls(x: Tensor, y: Tensor, alpha: float = 1.0) -> LinearMapping:
    """Exact centered ridge via two-pass support-only recursive least squares.

    The first pass gets centering means. Use :class:`StreamingRidge` when a
    one-pass sufficient-statistics fit is needed; float64 is recommended here.
    """
    x, y = _check(x, y)
    if alpha <= 0: return fit_ridge(x, y, alpha)
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    d = x.shape[1]
    P = torch.eye(d, dtype=x.dtype, device=x.device) / alpha
    weight = torch.zeros(d, y.shape[1], dtype=x.dtype, device=x.device)
    for row, target in zip(xc, yc):
        p_row = P @ row
        gain = p_row / (1 + row @ p_row)
        weight = weight + gain[:, None] * (target - row @ weight)[None, :]
        P = P - gain[:, None] * p_row[None, :]
    return LinearMapping(weight, ym - xm @ weight)


def fit_delta(x: Tensor, y: Tensor, lr: float = 1e-2, steps: int = 1) -> LinearMapping:
    """Approximate online delta-rule fit; this is not a ridge-equivalent method."""
    x, y = _check(x, y)
    if lr <= 0 or steps < 1: raise ValueError("lr must be positive and steps >= 1")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    weight = torch.zeros(x.shape[1], y.shape[1], dtype=x.dtype, device=x.device)
    for _ in range(steps):
        for row, target in zip(xc, yc):
            weight = weight + lr * row[:, None] * (target - row @ weight)[None, :]
    return LinearMapping(weight, ym - xm @ weight)
