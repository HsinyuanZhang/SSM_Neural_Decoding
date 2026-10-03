import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ssm_decode.unit_session_frontend import (
    UnitSessionBank, UnitSessionDecoder, fold_unit_session, unit_delta_penalty,
)


class ToyDecoder(nn.Module):
    def __init__(self, channels=3, width=4, outputs=2, *, bias=True):
        super().__init__()
        self.in_proj = nn.Linear(channels, width, bias=bias)
        self.blocks = nn.ModuleList([nn.Identity(), nn.Tanh()])
        self.final_norm = nn.Identity()
        self.out_proj = nn.Linear(width, outputs, bias=bias)

    def forward(self, x):
        z = self.in_proj(x)
        for block in self.blocks:
            z = block(z)
        return self.out_proj(self.final_norm(z))


def _configured_bank(residual=True, readout=True):
    bank = UnitSessionBank(2, 3, 4, 2, unit_residual=residual, session_readout=readout)
    with torch.no_grad():
        bank.gain.copy_(torch.tensor([[1., 2., .5], [2., .5, 1.5]]))
        bank.bias.normal_(); bank.embedding.normal_()
        if residual: bank.unit_delta.normal_()
        if readout: bank.readout_log_gain.normal_(); bank.readout_bias.normal_()
    return bank


def test_identity_and_multiple_session_routing():
    torch.manual_seed(1); base = ToyDecoder(); bank = UnitSessionBank(3, 3, 4, 2)
    model = UnitSessionDecoder(base, bank); x = torch.randn(2, 5, 3)
    assert torch.equal(model(x, 1), base(x))
    with torch.no_grad(): bank.bias[2].fill_(1); bank.embedding[2].fill_(2)
    assert torch.allclose(model(x, torch.tensor([0, 2]))[0], base(x[:1])[0])
    assert not torch.allclose(model(x, torch.tensor([0, 2]))[1], base(x[1:])[0])


def test_config_is_exposed_from_base_when_present():
    base = ToyDecoder()
    base.config = SimpleNamespace(output_size=2)
    model = UnitSessionDecoder(base, UnitSessionBank(1, 3, 4, 2))
    assert model.config is base.config
    assert model.config.output_size == 2


@pytest.mark.parametrize("bias", [True, False])
def test_fold_exact_for_session_and_source_means_including_bias_free(bias):
    torch.manual_seed(2); base = ToyDecoder(bias=bias); bank = _configured_bank(); model = UnitSessionDecoder(base, bank)
    x = torch.randn(3, 6, 3)
    for session in (1, None):
        folded = fold_unit_session(base, bank, session)
        compare_bank = bank if session is not None else bank.target_bank()
        compare = UnitSessionDecoder(copy.deepcopy(base), compare_bank)(x, 0 if session is None else session)
        assert torch.allclose(folded(x), compare, atol=3e-6, rtol=3e-6)
    assert fold_unit_session(base, bank, 0).in_proj.bias is not None
    assert fold_unit_session(base, bank, 0).out_proj.bias is not None


def test_readout_fold_and_penalty():
    torch.manual_seed(3); base = ToyDecoder(); bank = _configured_bank(residual=False, readout=True); x = torch.randn(2, 3, 3)
    assert torch.allclose(fold_unit_session(base, bank, 1)(x), UnitSessionDecoder(base, bank)(x, 1), atol=3e-6, rtol=3e-6)
    no_residual = UnitSessionBank(2, 3, 4, 2, unit_residual=False)
    assert unit_delta_penalty(no_residual).item() == 0 and unit_delta_penalty(no_residual).device == no_residual.gain.device
    residual_bank = UnitSessionBank(2, 3, 4, 2)
    residual_bank.unit_delta.data.fill_(2)
    assert unit_delta_penalty(residual_bank).item() == 4


def test_mean_fold_averages_effective_input_and_readout_maps():
    torch.manual_seed(31); base = ToyDecoder(); bank = _configured_bank()
    with torch.no_grad():
        bank.readout_log_gain.copy_(torch.tensor([[-2., 1.], [1.5, -1.]]))
    first, second = (fold_unit_session(base, bank, index) for index in (0, 1))
    mean_fold = fold_unit_session(base, bank)
    target_fold = fold_unit_session(base, bank.target_bank(), 0)
    for path in ('in_proj.weight', 'in_proj.bias', 'out_proj.weight', 'out_proj.bias'):
        module, attribute = path.split('.')
        expected = (getattr(getattr(first, module), attribute) + getattr(getattr(second, module), attribute)) / 2
        assert torch.allclose(getattr(getattr(mean_fold, module), attribute), expected, atol=2e-6, rtol=2e-6)
        assert torch.allclose(getattr(getattr(target_fold, module), attribute), expected, atol=2e-6, rtol=2e-6)


def test_residual_gradient_and_original_channel_direction():
    torch.manual_seed(4); base = ToyDecoder(); bank = UnitSessionBank(1, 3, 4, 2)
    # e1 is orthogonal to the first input-projection column e0, but remains useful.
    direction = torch.tensor([0., 1., 0., 0.])
    with torch.no_grad():
        base.in_proj.weight.zero_(); base.in_proj.bias.zero_(); base.in_proj.weight[0, 0] = 1
        base.out_proj.weight.zero_(); base.out_proj.bias.zero_(); base.out_proj.weight[0, 1] = 1
    assert torch.dot(base.in_proj.weight[:, 0], direction) == 0
    x = torch.zeros(2, 4, 3, requires_grad=True); x.data[:, :, 0] = 1
    model = UnitSessionDecoder(base, bank)
    model(x, 0)[..., 0].sum().backward()
    assert bank.unit_delta.grad is not None
    assert torch.dot(bank.unit_delta.grad[0, :, 0], direction).abs() > 0
    # A residual direction contributes independently of the input channel scaling.
    with torch.no_grad():
        bank.gain.fill_(7); bank.unit_delta.zero_(); bank.unit_delta[0, 0, 1] = 1
    z = base.in_proj(x.detach() * bank.gain[0]) + torch.einsum("btc,dc->btd", x.detach(), bank.unit_delta[0])
    got = model(x.detach(), 0)
    expected = base.out_proj(torch.tanh(z))
    assert torch.allclose(got, expected)


def test_target_bank_is_independent_and_mean_initialized():
    bank = _configured_bank(); target = bank.target_bank()
    for name in ("gain", "bias", "embedding", "unit_delta", "readout_bias"):
        assert getattr(target, name).data_ptr() != getattr(bank, name).data_ptr()
        assert torch.equal(getattr(target, name)[0], getattr(bank, name).mean(0))
    assert target.readout_log_gain.data_ptr() != bank.readout_log_gain.data_ptr()
    assert torch.allclose(target.readout_log_gain[0].exp(), bank.readout_log_gain.exp().mean(0))
    target.gain.data.add_(1)
    assert not torch.equal(target.gain[0], bank.gain.mean(0))


@pytest.mark.parametrize("index", [True, float("nan"), float("inf"), 1.5, -1, 2, torch.tensor([0, 1])])
def test_invalid_session_indices_rejected(index):
    model = UnitSessionDecoder(ToyDecoder(), UnitSessionBank(2, 3, 4, 2)); x = torch.randn(1, 2, 3)
    with pytest.raises(ValueError): model(x, index)


def test_causality_toy_base():
    base = ToyDecoder(); bank = _configured_bank(); model = UnitSessionDecoder(base, bank)
    x = torch.randn(1, 5, 3); altered = x.clone(); altered[:, 3:] += 30
    assert torch.equal(model(x, 1)[:, :3], model(altered, 1)[:, :3])


def test_channel_reordering_equivariance():
    torch.manual_seed(5); base = ToyDecoder(); bank = _configured_bank(); x = torch.randn(2, 4, 3); p = torch.tensor([2, 0, 1])
    permuted_base, permuted_bank = copy.deepcopy(base), copy.deepcopy(bank)
    with torch.no_grad():
        permuted_base.in_proj.weight.copy_(base.in_proj.weight[:, p])
        permuted_bank.gain.copy_(bank.gain[:, p]); permuted_bank.bias.copy_(bank.bias[:, p]); permuted_bank.unit_delta.copy_(bank.unit_delta[:, :, p])
    assert torch.allclose(UnitSessionDecoder(base, bank)(x, torch.tensor([0, 1])), UnitSessionDecoder(permuted_base, permuted_bank)(x[:, :, p], torch.tensor([0, 1])))
