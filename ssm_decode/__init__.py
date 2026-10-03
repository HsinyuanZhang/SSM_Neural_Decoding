"""Lightweight, explicitly non-official SSM motor-decoding pilot."""

from .models import ModelConfig, build_model
from .calibration import StreamingRidge, fit_delta, fit_ridge, fit_rls

__all__ = ["ModelConfig", "build_model", "StreamingRidge", "fit_ridge", "fit_rls", "fit_delta"]
