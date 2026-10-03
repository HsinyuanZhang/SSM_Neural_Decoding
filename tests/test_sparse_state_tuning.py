import copy

import pytest
import torch
from torch import nn

from ssm_decode.peft import LoRALinear
from ssm_decode.sparse_state_tuning import assert_stage_ready, assert_unique_trainable_storage, begin_dense_selection, install_sparse_state_tuning, selection_from_warmup


class ToyCore(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.B_bias = nn.Parameter(torch.ones(2, 1, 4, dtype=dtype))
        self.C_bias = nn.Parameter(torch.ones(2, 1, 4, dtype=dtype))
        self.out_proj = nn.Linear(4, 4, dtype=dtype)

    def forward(self, value):
        # Both bias paths affect loss and gradients.
        return self.out_proj(value + self.B_bias.mean() + 2.0 * self.C_bias.mean())


class ToyBlock(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.ssm = ToyCore(dtype)


class ToyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.in_proj = nn.Linear(3, 4, dtype=dtype)
        self.blocks = nn.ModuleList([ToyBlock(dtype)])
        self.out_proj = nn.Linear(4, 2, dtype=dtype)

    def forward(self, value):
        return self.out_proj(self.blocks[0].ssm(self.in_proj(value)))


def _dense(core, name):
    return core.parametrizations[name][0].delta


def test_dense_warmup_selects_known_states_and_restores_exact_base():
    torch.manual_seed(2)
    model, value = ToyModel(), torch.randn(3, 3)
    baseline = model(value).detach().clone()
    original_b, original_c = (model.blocks[0].ssm.B_bias.detach().clone(), model.blocks[0].ssm.C_bias.detach().clone())
    warmup = begin_dense_selection(model)
    assert_stage_ready(model, "dense_warmup")
    assert warmup["trainable_count"] == original_b.numel() + original_c.numel()
    assert torch.equal(model(value), baseline)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(value).square().mean().backward()
    for name in ("B_bias", "C_bias"):
        grad = _dense(model.blocks[0].ssm, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    before_b = _dense(model.blocks[0].ssm, "B_bias").detach().clone()
    before_c = _dense(model.blocks[0].ssm, "C_bias").detach().clone()
    optimizer.step()
    assert not torch.equal(_dense(model.blocks[0].ssm, "B_bias"), before_b)
    assert not torch.equal(_dense(model.blocks[0].ssm, "C_bias"), before_c)
    with torch.no_grad():
        _dense(model.blocks[0].ssm, "B_bias").zero_()
        _dense(model.blocks[0].ssm, "C_bias").zero_()
        _dense(model.blocks[0].ssm, "B_bias")[..., 2] = 3
        _dense(model.blocks[0].ssm, "C_bias")[..., 1] = 4
    receipt = selection_from_warmup(model, 2)
    assert_stage_ready(model, "selected")
    assert receipt["selections"][0] == [1, 2]
    assert receipt["details"][0]["mask"] == [False, True, True, False]
    assert torch.equal(model.blocks[0].ssm.B_bias, original_b)
    assert torch.equal(model.blocks[0].ssm.C_bias, original_c)


def test_sparse_state_tuning_is_compact_and_changes_only_selected_rows():
    model, value = ToyModel(), torch.randn(2, 3)
    baseline = model(value).detach().clone()
    begin_dense_selection(model)
    selection_from_warmup(model, 2)
    receipt = install_sparse_state_tuning(model, {0: [1, 3]}, rank=2)
    assert_stage_ready(model, "sparse")
    core = model.blocks[0].ssm
    assert receipt["custom_not_original_sdlora"] is True
    assert receipt["warmup_dense_trainable_count"] == 16
    assert torch.equal(model(value), baseline)
    assert core.parametrizations.B_bias[0].values.numel() == 4
    assert core.parametrizations.C_bias[0].values.numel() == 4
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(value).square().mean().backward()
    assert core.parametrizations.B_bias[0].values.grad.abs().sum() > 0
    assert core.parametrizations.C_bias[0].values.grad.abs().sum() > 0
    assert model.in_proj.A.grad.abs().sum() > 0
    before_b = core.parametrizations.B_bias[0].values.detach().clone()
    before_c = core.parametrizations.C_bias[0].values.detach().clone()
    optimizer.step()
    assert not torch.equal(core.parametrizations.B_bias[0].values, before_b)
    assert not torch.equal(core.parametrizations.C_bias[0].values, before_c)
    for name in ("B_bias", "C_bias"):
        adapter = core.parametrizations[name][0]
        delta = getattr(core, name).detach() - core.parametrizations[name].original.detach()
        assert torch.equal(delta[..., [0, 2]], torch.zeros_like(delta[..., [0, 2]]))
        assert adapter.values.detach().abs().sum() > 0
    assert not torch.equal(model(value), baseline)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    assert_unique_trainable_storage(model)


def test_sparse_checkpoint_replays_when_same_selection_is_installed():
    torch.manual_seed(3)
    model, value = ToyModel(), torch.randn(2, 3)
    base_state = copy.deepcopy(model.state_dict())
    begin_dense_selection(model)
    selected = selection_from_warmup(model, 2)
    install_sparse_state_tuning(model, selected["selections"], 2)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(value).square().mean().backward()
    optimizer.step()
    saved = copy.deepcopy(model.state_dict())
    expected = model(value).detach()

    restored = ToyModel()
    restored.load_state_dict(base_state)
    install_sparse_state_tuning(restored, selected["selections"], 2)
    restored.load_state_dict(saved)
    assert torch.equal(restored(value), expected)


def test_selection_inputs_and_rank_are_validated():
    model = ToyModel()
    begin_dense_selection(model)
    with pytest.raises(ValueError, match="positive"):
        selection_from_warmup(model, 0)
    selection_from_warmup(model, 1)
    with pytest.raises(ValueError, match="positive"):
        install_sparse_state_tuning(model, {0: [1]}, 0)
    with pytest.raises(ValueError, match="unique"):
        install_sparse_state_tuning(model, {0: [1, 1]}, 2)
    with pytest.raises(ValueError, match="outside"):
        install_sparse_state_tuning(model, {0: [4]}, 2)


def test_dense_and_sparse_adapters_follow_base_dtype():
    model = ToyModel(dtype=torch.float64)
    begin_dense_selection(model)
    core = model.blocks[0].ssm
    assert _dense(core, "B_bias").dtype is torch.float64
    assert _dense(core, "B_bias").device == core.B_bias.device
    selection_from_warmup(model, 2)
    install_sparse_state_tuning(model, {0: [0, 2]}, 2)
    assert core.parametrizations.B_bias[0].values.dtype is torch.float64
    assert isinstance(model.in_proj, LoRALinear)
    assert model.in_proj.A.dtype is torch.float64
