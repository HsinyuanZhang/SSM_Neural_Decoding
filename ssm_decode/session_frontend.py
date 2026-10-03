"""Small per-session input adapters for cross-session decoder experiments."""
from __future__ import annotations

import torch
from torch import nn


class SessionFrontendBank(nn.Module):
    """Store one channel affine map and one latent embedding per session."""

    def __init__(self, num_sessions, input_size, width, *, device=None, dtype=None):
        super().__init__()
        if not all(isinstance(value, int) and value > 0 for value in (num_sessions, input_size, width)):
            raise ValueError("num_sessions, input_size, and width must be positive integers")
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be floating point")
        self.num_sessions, self.input_size, self.width = num_sessions, input_size, width
        self.gain = nn.Parameter(torch.ones(num_sessions, input_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_sessions, input_size, device=device, dtype=dtype))
        self.embedding = nn.Parameter(torch.zeros(num_sessions, width, device=device, dtype=dtype))

    def _indices(self, session_index, batch, device):
        index = torch.as_tensor(session_index, device=device)
        if index.ndim == 0:
            index = index.expand(batch)
        elif index.ndim != 1 or index.numel() != batch:
            raise ValueError("session_index must be scalar or have shape [B]")
        if index.dtype.is_floating_point:
            if not torch.isfinite(index).all() or not torch.equal(index, index.round()):
                raise ValueError("session_index must contain finite integers")
        index = index.to(torch.long)
        if (index < 0).any() or (index >= self.num_sessions).any():
            raise ValueError("session_index is out of range")
        return index

    def forward_input(self, x, session_index):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError("x must have shape [B,T,input_size]")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("x must be finite floating point")
        index = self._indices(session_index, x.shape[0], x.device)
        return x * self.gain[index].to(x).unsqueeze(1) + self.bias[index].to(x).unsqueeze(1)

    def add_embedding(self, z, session_index):
        if z.ndim != 3 or z.shape[-1] != self.width:
            raise ValueError("z must have shape [B,T,width]")
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("z must be finite floating point")
        index = self._indices(session_index, z.shape[0], z.device)
        return z + self.embedding[index].to(z).unsqueeze(1)


def _session_values(bank, session_index):
    if not isinstance(bank, SessionFrontendBank):
        raise TypeError("bank must be a SessionFrontendBank")
    if session_index is None:
        return bank.gain.mean(0), bank.bias.mean(0), bank.embedding.mean(0)
    index = bank._indices(session_index, 1, bank.gain.device)
    return bank.gain[index[0]], bank.bias[index[0]], bank.embedding[index[0]]


def fold_session_input(linear, bank, session_index=None):
    """Fold a session input affine map and latent embedding into one Linear layer."""
    if not isinstance(linear, nn.Linear):
        raise TypeError("linear must be nn.Linear")
    if linear.in_features != bank.input_size or linear.out_features != bank.width:
        raise ValueError("linear dimensions do not match the frontend bank")
    gain, offset, embedding = _session_values(bank, session_index)
    output = nn.Linear(linear.in_features, linear.out_features, bias=True,
                       device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        weight = linear.weight.detach()
        output.weight.copy_(weight * gain.detach().to(weight).unsqueeze(0))
        base_bias = torch.zeros(linear.out_features, device=weight.device, dtype=weight.dtype) if linear.bias is None else linear.bias.detach()
        output.bias.copy_(weight @ offset.detach().to(weight) + base_bias + embedding.detach().to(weight))
    return output


def drift_augment(x, gain_sigma=.3, offset_sigma=.1, channel_dropout=.05, generator=None):
    """Apply a time-constant independent channel drift to a float32 input batch."""
    if x.ndim != 3 or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("x must be a finite float32 tensor with shape [B,T,C]")
    if gain_sigma < 0 or offset_sigma < 0 or not 0 <= channel_dropout <= 1:
        raise ValueError("invalid drift augmentation parameter")
    batch, _, channels = x.shape
    gains = torch.exp(torch.randn(batch, 1, channels, device=x.device, dtype=x.dtype, generator=generator) * gain_sigma)
    offsets = torch.randn(batch, 1, channels, device=x.device, dtype=x.dtype, generator=generator) * offset_sigma
    keep = torch.rand(batch, 1, channels, device=x.device, dtype=x.dtype, generator=generator) >= channel_dropout
    return (x * gains + offsets) * keep.to(x.dtype)
