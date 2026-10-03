"""Set-profile frontend for frozen-session causal SSM decoders."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class LearnedProfileFrontend(nn.Module):
    """Map per-unit support profiles to folded [channels,width] input weights.

    ``set_profile`` is called once during calibration.  Query ``forward`` only
    reads the materialized matrix, so it cannot consume query labels.
    """
    def __init__(self, profile_dim: int, width: int):
        super().__init__()
        self.profile_dim, self.width = int(profile_dim), int(width)
        self.encoder=nn.Sequential(nn.Linear(profile_dim,32),nn.ReLU(),nn.Linear(32,width))
        self.register_buffer("weights",torch.empty(0),persistent=False)

    def set_profile(self, profile: Tensor, unit_mask: Tensor | None = None) -> Tensor:
        if profile.ndim!=2 or profile.shape[1]!=self.profile_dim: raise ValueError("profile must be [channels, profile_dim]")
        if not torch.isfinite(profile).all(): raise ValueError("profile must be finite")
        mask=torch.ones(profile.shape[0],device=profile.device,dtype=profile.dtype) if unit_mask is None else unit_mask.to(profile.device,profile.dtype)
        if mask.shape != (profile.shape[0],): raise ValueError("unit_mask must be [channels]")
        # Keep folded activation scale comparable across recordings with
        # different observed channel counts; masking happens after MLP bias.
        scale=mask.sum().clamp_min(1).sqrt()
        self.weights=(self.encoder(profile)*mask[:,None]/scale).contiguous()
        return self.weights

    def forward(self,x: Tensor) -> Tensor:
        if self.weights.numel()==0: raise RuntimeError("set_profile must precede query/training forward")
        if x.shape[-1]!=self.weights.shape[0]: raise ValueError("input channel/profile mismatch")
        return x @ self.weights.to(x.device,x.dtype)

    def export_weights(self) -> Tensor:
        if self.weights.numel()==0: raise RuntimeError("no materialized profile weights")
        return self.weights.detach().cpu().contiguous()

def load_profile_checkpoint(path, *, profile: Tensor, device="cpu"):
    """Recreate a saved profile SSM and materialize its frozen query weights."""
    from .models import ModelConfig, build_model
    payload=torch.load(path,map_location=device); cfg=ModelConfig(**payload["model_config"])
    model=build_model(cfg).to(device); front=LearnedProfileFrontend(int(payload["profile_dim"]),cfg.width).to(device);model.frontend=front
    model.load_state_dict(payload["state_dict"]);front.set_profile(profile.to(device));return model
