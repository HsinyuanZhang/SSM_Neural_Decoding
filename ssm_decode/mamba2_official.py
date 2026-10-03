"""Optional wrapper around the pinned official Mamba-2 module.

This intentionally has no fallback implementation.  It uses Mamba2's own
log-uniform ``dt_bias`` initialization and its unfused PyTorch causal-conv +
official Triton SSD scan route (``use_mem_eff_path=False``).
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import traceback
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


OFFICIAL = Path(__file__).resolve().parents[1] / ".tools" / "mamba_official"
DEPS = Path(__file__).resolve().parents[1] / ".tools" / "mamba_deps"
COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def _require_isolated_triton():
    """Avoid a mixed graph of system and isolated Triton modules."""
    try:
        triton = importlib.import_module("triton")
    except Exception as exc:
        raise RuntimeError("Triton is unavailable; launch with isolated mamba_deps PYTHONPATH") from exc
    location = str(getattr(triton, "__file__", ""))
    if not location.startswith(str(DEPS)):
        raise RuntimeError(
            f"non-isolated Triton already loaded from {location}; restart with PYTHONPATH={DEPS}:<SSM>"
        )
    return triton


def _import_mamba2():
    """Import only the official namespace, avoiding its unrelated package init."""
    if not OFFICIAL.is_dir():
        raise RuntimeError(f"official Mamba clone missing: {OFFICIAL}")
    _require_isolated_triton()
    # mamba_ssm.__init__ imports Mamba1/transformers.  Namespace loading keeps
    # this optional comparator scoped to Mamba2 and surfaces real import errors.
    pkg = sys.modules.get("mamba_ssm")
    if pkg is None:
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [str(OFFICIAL / "mamba_ssm")]
        sys.modules["mamba_ssm"] = pkg
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        return Mamba2
    except Exception as exc:
        raise RuntimeError(
            "official Mamba-2 namespace import failed; no fallback is available"
        ) from exc


@dataclass(frozen=True)
class OfficialMamba2Config:
    input_size: int
    output_size: int
    width: int = 64
    layers: int = 1
    state_size: int = 128
    dropout: float = 0.0


class _PreNormMamba2Block(nn.Module):
    def __init__(self, mamba2_cls, cfg: OfficialMamba2Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width)
        # Preserve the reference module's dt/A/D initialization. The unfused
        # path deliberately avoids the unavailable causal-conv1d extension.
        self.ssm = mamba2_cls(
            d_model=cfg.width,
            d_state=cfg.state_size,
            headdim=64,
            chunk_size=64,
            use_mem_eff_path=False,
        )
        self.norm2 = nn.LayerNorm(cfg.width)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.width, 2 * cfg.width), nn.SiLU(), nn.Linear(2 * cfg.width, cfg.width)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.ssm(self.norm1(x)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class OfficialMamba2Decoder(nn.Module):
    """Deep pre-LN residual decoder using the official Mamba2 core unchanged."""
    def __init__(self, cfg: OfficialMamba2Config):
        super().__init__()
        if cfg.width <= 0 or cfg.layers < 1 or cfg.width % 32:
            raise ValueError("width must be positive, a multiple of 32, and layers >= 1")
        mamba2_cls = _import_mamba2()
        self.config = cfg
        self.in_proj = nn.Linear(cfg.input_size, cfg.width)
        self.blocks = nn.ModuleList([_PreNormMamba2Block(mamba2_cls, cfg) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.width)
        self.out_proj = nn.Linear(cfg.width, cfg.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.input_size:
            raise ValueError("x must be [B,T,input_size]")
        z = self.in_proj(x)
        for block in self.blocks:
            z = block(z)
        return self.out_proj(self.final_norm(z))


def build_mamba2_official(input_size, output_size, width=64, layers=1, state_size=128, dropout=0.0):
    return OfficialMamba2Decoder(
        OfficialMamba2Config(input_size, output_size, width, layers, state_size, dropout)
    )


def _exception_chain(exc: BaseException):
    chain = []
    cur = exc
    while cur is not None:
        chain.append({"type": type(cur).__name__, "message": str(cur)[:1200]})
        cur = cur.__cause__ or cur.__context__
    return chain


def cpu_import_constructor_attempt():
    """CPU-only readiness check; it never invokes an official Triton kernel."""
    result = {
        "schema": "official_mamba2_cpu_readiness_v1",
        "official_path": str(OFFICIAL),
        "expected_commit": COMMIT,
        "torch": torch.__version__,
        "check": "CPU namespace import and constructor only; no forward/kernel/GPU call",
        "architecture": "pre-LayerNorm residual official-Mamba2 + LayerNorm/SiLU FFN(2x width), final LayerNorm",
        "official_core": {"use_mem_eff_path": False, "headdim": 64, "state_size": 16, "chunk_size": 64},
    }
    try:
        result["actual_commit"] = subprocess.check_output(
            ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
        ).strip()
        model = build_mamba2_official(8, 2, 64, 2, 16)
        core = model.blocks[0].ssm
        triton = _require_isolated_triton()
        result.update(
            status="cpu_import_constructor_supported",
            parameter_count=sum(p.numel() for p in model.parameters()),
            config=asdict(model.config),
            official_dt_initialization="official Mamba2 torch.exp(uniform(log(dt_min), log(dt_max))) then inverse-softplus bias",
            observed_core={
                "use_mem_eff_path": bool(core.use_mem_eff_path),
                "headdim": int(core.headdim), "d_state": int(core.d_state), "chunk_size": int(core.chunk_size),
                "dt_bias_shape": list(core.dt_bias.shape), "dt_bias_finite": bool(torch.isfinite(core.dt_bias).all()),
            },
            triton_version=triton.__version__, triton_file=str(triton.__file__),
        )
    except Exception as exc:
        result.update(
            status="cpu_import_or_constructor_unavailable",
            exception_type=type(exc).__name__, exception_message=str(exc)[:1200],
            exception_chain=_exception_chain(exc), traceback_tail=traceback.format_exc().splitlines()[-12:],
        )
    return result


def compatibility_attempt(device="cuda:0"):
    """Actual GPU probe; callers must obtain explicit GPU-release authorization."""
    if not str(device).startswith("cuda"):
        raise ValueError("Mamba2 compatibility_attempt is a GPU-only probe; use cpu_import_constructor_attempt for CPU")
    result = {
        "schema": "official_mamba2_gpu_compatibility_v1", "official_path": str(OFFICIAL),
        "expected_commit": COMMIT, "device": device, "torch": torch.__version__,
        "architecture": "2x pre-LayerNorm residual official-Mamba2 + LayerNorm/SiLU FFN(2x width), final LayerNorm",
        "official_core": {"use_mem_eff_path": False, "headdim": 64, "state_size": 16, "chunk_size": 64},
    }
    try:
        import time
        result["actual_commit"] = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
        model = build_mamba2_official(8, 2, 128, 2, 16).to(device).train()
        x = torch.randn(32, 128, 8, device=device, requires_grad=True)
        y = model(x); y.square().mean().backward()
        model.eval(); base = x.detach(); changed = base.clone(); changed[:, 64:] += torch.randn_like(changed[:, 64:])
        with torch.no_grad():
            first = model(base)[:, :64]; altered = model(changed)[:, :64]
        for _ in range(3): model(base)
        torch.cuda.synchronize(); began = time.perf_counter()
        for _ in range(5): model(base)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - began
        triton = _require_isolated_triton()
        kernel = importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
        kernel_triton = getattr(kernel, "triton", None)
        result.update(status="supported", forward_shape=list(y.shape), backward_finite=bool(torch.isfinite(x.grad).all()), causal_future_perturb_max_abs=float((first-altered).abs().max()), parameters=sum(p.numel() for p in model.parameters()), triton_version=triton.__version__, triton_file=str(triton.__file__), kernel_module_file=str(kernel.__file__), kernel_triton_file=str(getattr(kernel_triton, "__file__", None)), kernel_triton_matches_runtime=kernel_triton is triton, timing={"warmup":3,"iterations":5,"elapsed_seconds":elapsed,"milliseconds_per_forward":elapsed*200})
    except Exception as exc:
        result.update(status="unsupported_or_unavailable", exception_type=type(exc).__name__, exception_message=str(exc)[:1200], exception_chain=_exception_chain(exc), traceback_tail=traceback.format_exc().splitlines()[-12:])
    return result


def write_gpu_compatibility(path=None, device="cuda:0"):
    path = Path(path or (Path(__file__).resolve().parents[1] / "results" / "debug_audit" / "mamba2_compatibility.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    result = compatibility_attempt(device)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_cpu_compatibility(path=None):
    path = Path(path or (Path(__file__).resolve().parents[1] / "results" / "debug_audit" / "mamba2_compatibility.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    result = cpu_import_constructor_attempt()
    # A failed pre-install receipt remains useful evidence after dependencies
    # are supplied; preserve it only when this is the canonical audit artifact.
    if path.exists():
        try:
            prior = json.loads(path.read_text())
            if prior.get("status") == "cpu_import_or_constructor_unavailable":
                result["dependency_history"] = [{
                    "status": prior.get("status"),
                    "exception_chain": prior.get("exception_chain", []),
                    "remediation": "installed huggingface_hub==0.27.1 --no-deps into isolated .tools/mamba_deps",
                }]
        except (OSError, json.JSONDecodeError):
            pass
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result
