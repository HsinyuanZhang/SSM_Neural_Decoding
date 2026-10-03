import numpy as np
import pytest

from ssm_decode.session_normalization import fit_support_statistics, normalize_neural, strict_past_ema_normalize


def test_support_statistics_ignore_query_neural_and_preserve_output_coordinates():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    stats = (np.zeros(2), np.ones(2), np.array([5.]), np.array([2.]))
    bounds = [(0, 4), (5, 8)]
    first = fit_support_statistics(x, bounds, stats)
    changed = x.copy()
    changed[8:] = 1e6
    second = fit_support_statistics(changed, bounds, stats)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[2], stats[2])
    np.testing.assert_array_equal(first[3], stats[3])


def test_ema_prefix_matches_fixed_support_normalization_exactly():
    x = np.random.default_rng(0).normal(size=(30, 3)).astype(np.float32)
    bounds = [(1, 7), (9, 14)]
    fixed = fit_support_statistics(x, bounds, (None, None, np.zeros(1), np.ones(1)))
    actual, receipt = strict_past_ema_normalize(x, bounds, half_life=3)
    np.testing.assert_array_equal(actual[:14], normalize_neural(x, fixed)[:14])
    assert receipt['update_start_bin'] == 14


def test_ema_future_perturbation_cannot_change_earlier_outputs():
    x = np.random.default_rng(3).normal(size=(40, 2)).astype(np.float32)
    a, _ = strict_past_ema_normalize(x, [(0, 5)], half_life=4)
    altered = x.copy()
    altered[22:] += 100
    b, _ = strict_past_ema_normalize(altered, [(0, 5)], half_life=4)
    np.testing.assert_array_equal(a[:22], b[:22])


def test_ema_uses_preupdate_mean_and_central_variance():
    x = np.array([[0.], [2.], [5.], [9.]], dtype=np.float32)
    actual, _ = strict_past_ema_normalize(x, [(0, 2)], half_life=1)
    assert actual[2, 0] == 4.0  # mean=1, variance=1 before the first query bin.
    # alpha=.5: mean=3, variance=.5*(1+.5*16)=4.5.
    np.testing.assert_allclose(actual[3, 0], 6 / np.sqrt(4.5), rtol=1e-6)


def test_constant_support_channel_remains_finite():
    x = np.ones((20, 2), dtype=np.float32)
    x[10:] *= 2
    actual, _ = strict_past_ema_normalize(x, [(0, 8)])
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:8], np.zeros((8, 2)))


def test_invalid_support_intervals_are_rejected():
    with pytest.raises(ValueError, match='overlap'):
        strict_past_ema_normalize(np.zeros((10, 2)), [(0, 5), (4, 7)])
