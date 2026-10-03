import torch
from torch import nn
import pytest

from ssm_decode.session_frontend import SessionFrontendBank, drift_augment, fold_session_input


def test_identity_startup_and_session_batch_indexing():
    bank = SessionFrontendBank(3, 4, 5)
    x, z = torch.randn(2, 6, 4), torch.randn(2, 6, 5)
    assert torch.equal(bank.forward_input(x, 1), x)
    assert torch.equal(bank.add_embedding(z, torch.tensor([0, 2])), z)
    bank.gain.data[2].fill_(2); bank.bias.data[2].fill_(3); bank.embedding.data[2].fill_(4)
    y = bank.forward_input(x, torch.tensor([0, 2])); q = bank.add_embedding(z, torch.tensor([0, 2]))
    assert torch.equal(y[0], x[0]) and torch.equal(y[1], 2*x[1]+3)
    assert torch.equal(q[0], z[0]) and torch.equal(q[1], z[1]+4)


def test_fold_matches_session_and_mean_with_bias_or_no_bias():
    torch.manual_seed(2); bank = SessionFrontendBank(2, 3, 4); x = torch.randn(3, 5, 3)
    bank.gain.data.copy_(torch.tensor([[1., 2., .5], [2., .5, 1.5]])); bank.bias.data.normal_(); bank.embedding.data.normal_()
    for linear in (nn.Linear(3, 4), nn.Linear(3, 4, bias=False)):
        session = 1; folded = fold_session_input(linear, bank, session)
        expected = bank.add_embedding(linear(bank.forward_input(x, session)), session)
        assert torch.allclose(folded(x), expected, atol=2e-6, rtol=2e-6)
        mean = fold_session_input(linear, bank)
        gain, bias, embedding = bank.gain.mean(0), bank.bias.mean(0), bank.embedding.mean(0)
        assert torch.allclose(mean(x), linear(x*gain+bias)+embedding, atol=2e-6, rtol=2e-6)


def test_drift_is_deterministic_time_constant_and_causal():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    a = drift_augment(x, generator=torch.Generator().manual_seed(8)); b = drift_augment(x, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and torch.equal(x, torch.arange(24, dtype=torch.float32).reshape(2, 4, 3))
    # A constant channel transform preserves one-step differences across time.
    assert torch.allclose((a[:, 2:]-a[:, 1:-1]) / (x[:, 2:]-x[:, 1:-1]), (a[:, 1:-1]-a[:, :-2]) / (x[:, 1:-1]-x[:, :-2]))
    changed = x.clone(); changed[:, 3:] += 100
    same = drift_augment(x, generator=torch.Generator().manual_seed(9))
    altered = drift_augment(changed, generator=torch.Generator().manual_seed(9))
    assert torch.equal(same[:, :3], altered[:, :3])


def test_dropout_bounds_and_invalid_inputs():
    x = torch.ones(4, 3, 5, dtype=torch.float32)
    assert torch.equal(drift_augment(x, gain_sigma=0, offset_sigma=0, channel_dropout=1), torch.zeros_like(x))
    assert torch.equal(drift_augment(x, gain_sigma=0, offset_sigma=0, channel_dropout=0), x)
    with pytest.raises(ValueError): SessionFrontendBank(0, 2, 3)
    with pytest.raises(ValueError): drift_augment(x.double())
    with pytest.raises(ValueError): SessionFrontendBank(2, 3, 4).forward_input(x[..., :3], torch.tensor([0]))
