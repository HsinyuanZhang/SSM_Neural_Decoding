"""Fit support input statistics and apply causal test-time normalization."""
from __future__ import annotations

import numpy as np


def _validate_bounds(neural, bounds):
    x = np.asarray(neural)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('neural must be a finite [time, channels] array')
    checked = [(int(a), int(b)) for a, b in bounds]
    if not checked or any(a < 0 or b <= a or b > len(x) for a, b in checked):
        raise ValueError('support bounds must be nonempty valid intervals')
    ordered = sorted(checked)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('support intervals must not overlap')
    return x, checked


def fit_support_statistics(neural, bounds, source_stats):
    """Use support neural data only. Preserve source output coordinates."""
    x, checked = _validate_bounds(neural, bounds)
    if len(source_stats) != 4:
        raise ValueError('source_stats must contain x_mean, x_std, y_mean, y_std')
    support = np.concatenate([x[a:b] for a, b in checked]).astype(np.float64)
    mean, std = support.mean(0), support.std(0)
    std[std < 1e-6] = 1.0
    result = (mean.astype(np.float32), std.astype(np.float32),
              np.asarray(source_stats[2], dtype=np.float32).copy(),
              np.asarray(source_stats[3], dtype=np.float32).copy())
    if result[0].shape != (x.shape[1],) or not all(np.isfinite(z).all() for z in result):
        raise ValueError('invalid support or source statistics')
    if not (result[1] > 0).all() or not (result[3] > 0).all():
        raise ValueError('normalizer scales must be positive')
    return result


def normalize_neural(neural, stats):
    x = np.asarray(neural, dtype=np.float32)
    if x.ndim != 2 or np.shape(stats[0]) != (x.shape[1],):
        raise ValueError('neural and statistics shapes differ')
    result = (x - stats[0]) / stats[1]
    if not np.isfinite(result).all():
        raise FloatingPointError('normalized input is not finite')
    return result.astype(np.float32)


def strict_past_ema_normalize(neural, support_bounds, *, half_life=3000):
    """Freeze prefix statistics. Update after the support boundary only."""
    x, checked = _validate_bounds(neural, support_bounds)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('half_life must be finite and positive')
    support = np.concatenate([x[a:b] for a, b in checked]).astype(np.float64)
    mean, variance = support.mean(0), support.var(0)
    variance[variance < 1e-12] = 1.0
    boundary = max(b for a, b in checked)
    stats = (mean.astype(np.float32), np.sqrt(variance).astype(np.float32))
    mean, variance = stats[0].astype(np.float64), stats[1].astype(np.float64)**2
    result = np.empty_like(x, dtype=np.float32)
    # The offline support contract may use all support neural observations.
    # It is identical for fit, validation, and support calibration.
    result[:boundary] = (x[:boundary].astype(np.float32) - stats[0]) / stats[1]
    alpha = -np.expm1(-np.log(2.0) / float(half_life))
    for index in range(boundary, len(x)):
        observation = x[index].astype(np.float64)
        delta = observation - mean
        result[index] = delta / np.sqrt(np.maximum(variance, 1e-12))
        # This central-moment update uses the pre-update delta.
        variance = (1.0 - alpha) * (variance + alpha * delta**2)
        mean = mean + alpha * delta
    if not np.isfinite(result).all():
        raise FloatingPointError('EMA normalized input is not finite')
    receipt = {
        'protocol': 'support_initialized_strict_past_unlabeled_ema',
        'half_life_bins': float(half_life), 'update_start_bin': boundary,
        'support_bounds': checked, 'support_neural_count': len(support),
        'query_labels_used': False, 'output_before_update': True,
        'prefix_statistics_frozen': True,
        'support_mean': stats[0].tolist(), 'support_std': stats[1].tolist(),
    }
    return result, receipt
