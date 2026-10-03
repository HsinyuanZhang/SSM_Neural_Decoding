import torch
import pytest

from ssm_decode import ModelConfig, build_model
from ssm_decode.calibration import StreamingRidge, fit_ridge, fit_rls


@pytest.mark.parametrize("kind", ["diag", "osc", "bank", "selective", "gru"])
def test_streaming_and_chunks_equal_batch(kind):
    torch.manual_seed(3)
    model = build_model(ModelConfig(5, 3, width=7, kind=kind, group_size=2)).eval()
    x = torch.randn(2, 11, 5)
    full, state = model(x, return_state=True)
    first, middle = model(x[:, :4], return_state=True)
    second, final_state = model(x[:, 4:], state=middle, return_state=True)
    assert torch.allclose(full, torch.cat([first, second], 1), atol=1e-6)
    assert torch.allclose(state, final_state, atol=1e-6)


@pytest.mark.parametrize("kind", ["diag", "osc", "bank", "selective", "gru"])
def test_no_future_causality(kind):
    torch.manual_seed(7)
    model = build_model(ModelConfig(4, 2, width=6, kind=kind)).eval()
    x = torch.randn(2, 9, 4)
    changed = x.clone()
    changed[:, 6:] += torch.randn_like(changed[:, 6:]) * 20
    assert torch.allclose(model(x)[:, :6], model(changed)[:, :6], atol=1e-6)


def test_bank_has_bounded_spectra_and_is_frozen():
    model = build_model(ModelConfig(3, 2, width=12, kind="bank", group_size=3))
    eig = torch.linalg.eigvals(model.recurrence_matrices())
    assert torch.all(eig.abs() < 1)
    assert "bank_A" in dict(model.named_buffers())
    assert "bank_A" not in dict(model.named_parameters())


def test_rls_equals_centered_batch_ridge_float64():
    torch.manual_seed(11)
    x = torch.randn(51, 8, dtype=torch.float64)
    y = torch.randn(51, 3, dtype=torch.float64)
    batch, rls = fit_ridge(x, y, 0.7), fit_rls(x, y, 0.7)
    assert torch.allclose(batch.weight, rls.weight, rtol=1e-10, atol=1e-11)
    assert torch.allclose(batch.intercept, rls.intercept, rtol=1e-10, atol=1e-11)


def test_one_pass_streaming_stats_equal_batch_ridge_float64():
    torch.manual_seed(12)
    x, y = torch.randn(31, 4, dtype=torch.float64), torch.randn(31, 2, dtype=torch.float64)
    stream = StreamingRidge(4, 2).update(x[:9], y[:9]).update(x[9:], y[9:]).solve(0.3)
    batch = fit_ridge(x, y, 0.3)
    assert torch.allclose(stream.weight, batch.weight, rtol=1e-11, atol=1e-12)
    assert torch.allclose(stream.intercept, batch.intercept, rtol=1e-11, atol=1e-12)


def test_profile_frontend_is_permutation_invariant_and_masks_missing_units():
    torch.manual_seed(15)
    model = build_model(ModelConfig(6, 2, width=5, kind="diag", frontend="profile", profile_size=3)).eval()
    x = torch.randn(2, 7, 6)
    profile, mask = torch.randn(2, 6, 3), torch.tensor([[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0]], dtype=torch.bool)
    order = torch.tensor([3, 1, 5, 0, 2, 4])
    base = model(x, profiles=profile, unit_mask=mask)
    reordered = model(x[:, :, order], profiles=profile[:, order], unit_mask=mask[:, order])
    assert torch.allclose(base, reordered, atol=1e-6)
    # Padded/masked channels cannot affect the pooled static projection.
    altered = x.clone(); altered[0, :, ~mask[0]] += 1000
    altered[1, :, ~mask[1]] += 1000
    assert torch.allclose(base, model(altered, profiles=profile, unit_mask=mask), atol=1e-5)


def test_profile_fold_equals_explicit_static_projection():
    torch.manual_seed(16)
    model = build_model(ModelConfig(4, 2, width=5, frontend="profile", profile_size=2))
    x, p = torch.randn(3, 4), torch.randn(3, 4, 2)
    folded = model.frontend.fold(p)
    assert torch.allclose(model.frontend(x, p), torch.einsum("bc,bcw->bw", x, folded))
