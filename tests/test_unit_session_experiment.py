"""CPU contracts for target-prefix helpers in the unit-session experiment."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import ssm_decode.unit_session_experiment as experiment
from ssm_decode.unit_session_frontend import UnitSessionBank, UnitSessionDecoder, fold_unit_session

r2 = experiment.r2
raw_support_partition = experiment.raw_support_partition
target_statistics = experiment.target_statistics
target_windows = experiment.target_windows


SOURCE_STATS = (
    np.array([-3., 2.], np.float32), np.array([4., 5.], np.float32),
    np.array([11., -7.], np.float32), np.array([2., 9.], np.float32),
)


class TinyBase(nn.Module):
    """Projection-only base that exposes the same fold boundary as Mamba-3."""

    def __init__(self, inputs=3, width=4, outputs=2):
        super().__init__()
        self.in_proj = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([nn.Identity()])
        self.final_norm = nn.Identity()
        self.out_proj = nn.Linear(width, outputs)


def map_model(sessions=3, *, readout=True):
    torch.manual_seed(81)
    base = TinyBase()
    bank = UnitSessionBank(sessions, 3, 4, 2, unit_residual=True, session_readout=readout)
    with torch.no_grad():
        bank.gain.uniform_(.4, 1.8); bank.bias.normal_(); bank.embedding.normal_(); bank.unit_delta.normal_()
        if readout:
            bank.readout_log_gain.normal_(); bank.readout_bias.normal_()
    return UnitSessionDecoder(base, bank)


def calibration(n_trials, length=6, *, leading=0):
    """A raw-calibration fixture; all trials intentionally remain shorter than 50 bins."""
    total = leading + n_trials * length
    neural = np.arange(total * 2, dtype=np.float32).reshape(total, 2) / 10
    behavior = np.arange(total * 2, dtype=np.float32).reshape(total, 2) + 100
    bounds = tuple((leading + i * length, leading + (i + 1) * length) for i in range(n_trials))
    return SimpleNamespace(
        neural=neural,
        behavior=behavior,
        eval_mask=np.arange(total) % 2 == 0,
        trial_bounds=bounds,
        receipt={'raw_nwb_trial_ids_first_n': list(range(n_trials))},
    )


def test_raw_support_partition_m1_is_first_ten_with_trials_4_and_9_held_out():
    cal = calibration(10, leading=3)
    train, valid, ids = raw_support_partition(cal)
    assert ids == [4, 9]
    assert valid == [cal.trial_bounds[4], cal.trial_bounds[9]]
    assert train == [bound for i, bound in enumerate(cal.trial_bounds) if i not in ids]


def test_raw_support_partition_m2_retains_short_trials_and_reserves_required_ids():
    cal = calibration(33, length=6)
    train, valid, ids = raw_support_partition(cal)
    assert ids == [4, 9, 14, 19, 24, 29, 32]
    assert len(train) == 26 and len(valid) == 7
    assert all(b - a < 50 for a, b in train + valid)
    assert valid == [cal.trial_bounds[i] for i in ids]


@pytest.mark.parametrize('ids', [list(range(9)), list(range(1, 11)), [*range(9), 99]])
def test_raw_support_partition_rejects_wrong_raw_ids(ids):
    cal = calibration(10)
    cal.receipt['raw_nwb_trial_ids_first_n'] = ids
    with pytest.raises(ValueError, match='first 10 or 33 raw trials'):
        raw_support_partition(cal)


def test_target_statistics_uses_all_prefix_neural_and_fixed_source_outputs():
    cal = calibration(10, leading=4)
    before = target_statistics(cal, SOURCE_STATS)
    assert np.array_equal(before[0], cal.neural.mean(0).astype(np.float32))
    assert np.array_equal(before[1], cal.neural.astype(np.float64).std(0).astype(np.float32))
    assert np.array_equal(before[2], SOURCE_STATS[2]) and np.array_equal(before[3], SOURCE_STATS[3])
    changed = calibration(10, leading=4)
    changed.behavior[:] = -1e20
    after = target_statistics(changed, SOURCE_STATS)
    for first, second in zip(before, after):
        assert np.array_equal(first, second)


def test_target_windows_masks_other_partitions_and_behavior_poison_cannot_leak():
    cal = calibration(10, leading=3)
    stats = target_statistics(cal, SOURCE_STATS)
    train, valid, _ = raw_support_partition(cal)
    (x, y, mask), = target_windows(cal, train, stats, 'cpu')
    assert torch.equal(x, torch.from_numpy(((cal.neural - stats[0]) / stats[1]).astype(np.float32)))
    # The leading neural-only bins and the held-out raw trials remain label-free.
    assert torch.isnan(y[:3]).all() and not mask[:3].any()
    for a, b in valid:
        assert torch.isnan(y[a:b]).all() and not mask[a:b].any()
    for a, b in train:
        expected = torch.from_numpy(((cal.behavior[a:b] - stats[2]) / stats[3]).astype(np.float32))
        assert torch.equal(y[a:b], expected)
        assert torch.equal(mask[a:b], torch.from_numpy(cal.eval_mask[a:b]))
    # Poison future/outside-partition behavior: neither normalizer nor train window may change.
    changed = calibration(10, leading=3)
    outside = np.ones(len(changed.neural), dtype=bool)
    for a, b in train:
        outside[a:b] = False
    changed.behavior[outside] = np.nan
    changed_stats = target_statistics(changed, SOURCE_STATS)
    (changed_x, changed_y, changed_mask), = target_windows(changed, train, changed_stats, 'cpu')
    assert all(np.array_equal(a, b) for a, b in zip(stats, changed_stats))
    assert torch.equal(x, changed_x) and torch.allclose(y, changed_y, equal_nan=True) and torch.equal(mask, changed_mask)


def test_r2_is_float64_global_variance_weighted_for_multidimensional_outputs():
    truth = np.array([[0., 0.], [2., 100.], [4., 0.]], dtype=np.float32)
    prediction = np.array([[1., 0.], [1., 80.], [6., 0.]], dtype=np.float32)
    expected = 1 - np.square(truth.astype(np.float64) - prediction).sum() / np.square(
        truth.astype(np.float64) - truth.astype(np.float64).mean(0)).sum()
    assert r2(truth, prediction) == pytest.approx(expected)
    per_output_mean = np.mean([1 - np.square(truth[:, i] - prediction[:, i]).sum() /
                              np.square(truth[:, i] - truth[:, i].mean()).sum() for i in range(2)])
    assert r2(truth, prediction) != pytest.approx(per_output_mean)


@pytest.mark.parametrize('truth,prediction', [
    (np.ones((2, 1)), np.ones((3, 1))),
    (np.array([[np.nan]]), np.array([[0.]])),
    (np.array([[0.]]), np.array([[np.inf]])),
])
def test_r2_rejects_shape_and_nonfinite_inputs(truth, prediction):
    with pytest.raises(ValueError, match='equal shapes and finite values'):
        r2(truth, prediction)


def test_effective_maps_and_mapping_penalty_are_invariant_to_input_gauge_shift():
    model = map_model(2)
    before = experiment.effective_maps(model)
    before_penalty = experiment.mapping_penalty(before)
    alpha = torch.tensor([[.3, -.4, .2], [-.5, .1, .7]])
    shift = torch.tensor([[.2, -.1, .4], [-.3, .6, -.2]])
    with torch.no_grad():
        model.bank.gain.add_(alpha)
        model.bank.unit_delta.sub_(model.base.in_proj.weight.unsqueeze(0) * alpha.unsqueeze(1))
        model.bank.bias.add_(shift)
        model.bank.embedding.sub_(torch.einsum('dc,sc->sd', model.base.in_proj.weight, shift))
    after = experiment.effective_maps(model)
    for name in before:
        assert torch.allclose(after[name], before[name])
    assert torch.allclose(experiment.mapping_penalty(after), before_penalty)


def test_source_mapping_penalty_is_invariant_to_session_permutation():
    model = map_model(3)
    maps = experiment.effective_maps(model)
    permutation = torch.tensor([2, 0, 1])
    shuffled = {name: value[permutation] for name, value in maps.items()}
    assert torch.allclose(experiment.mapping_penalty(maps), experiment.mapping_penalty(shuffled))


def test_target_anchor_mapping_penalty_starts_zero_and_detects_directional_change():
    model = map_model(1)
    anchors = {name: value.detach().clone() for name, value in experiment.effective_maps(model).items()}
    assert experiment.mapping_penalty(experiment.effective_maps(model), anchors).item() == 0
    with torch.no_grad():
        model.bank.unit_delta[0, 2, 1].add_(.75)
    assert experiment.mapping_penalty(experiment.effective_maps(model), anchors).item() > 0


def test_effective_mean_maps_and_folded_readout_match_target_bank():
    model = map_model(2, readout=True)
    with torch.no_grad():
        model.bank.readout_log_gain.copy_(torch.tensor([[-2., 1.], [1.5, -1.]]))
    source_maps = experiment.effective_maps(model)
    target = UnitSessionDecoder(model.base, model.bank.target_bank())
    target_maps = experiment.effective_maps(target)
    for name, value in source_maps.items():
        assert torch.allclose(target_maps[name][0], value.mean(0), atol=2e-6, rtol=2e-6)
    source_fold = fold_unit_session(model.base, model.bank)
    target_fold = fold_unit_session(model.base, target.bank, 0)
    assert torch.allclose(source_fold.in_proj.weight, target_fold.in_proj.weight)
    assert torch.allclose(source_fold.in_proj.bias, target_fold.in_proj.bias)
    assert torch.allclose(source_fold.out_proj.weight, target_fold.out_proj.weight)
    assert torch.allclose(source_fold.out_proj.bias, target_fold.out_proj.bias)


@pytest.mark.parametrize('stage', ['source', 'target'])
def test_cpu_train_loop_keeps_frozen_base_and_does_not_extend_when_initial_checkpoint_wins(
        monkeypatch, tmp_path, stage):
    """Use actual TinyBase predictions; the synthetic validation sequence has no query input."""
    model = map_model(1)
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    base_before = {name: parameter.detach().clone() for name, parameter in model.base.named_parameters()}
    x = torch.randn(2, 4, 3)
    truth = torch.zeros(2, 4, 2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    scores = iter([1., .5, .75])  # Initial checkpoint wins, so a late maximum cannot trigger extension.
    cfg = {'weight_decay': 0., 'steps': 2, 'max_steps': 4, 'context': 4, 'val_interval': 1}
    output = tmp_path / stage
    output.mkdir()
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    if stage == 'source':
        penalty = lambda current, unused: experiment.mapping_penalty(experiment.effective_maps(current))
    else:
        penalty = lambda current, anchors: experiment.mapping_penalty(experiment.effective_maps(current), anchors)
    result = experiment.train_loop(
        model, lambda: (x, truth, mask, 0), lambda: next(scores), cfg, .01, penalty,
        output, stage, {'toy': True},
    )
    assert result['best_step'] == 0
    assert result['completed_steps'] == result['final_budget'] == 2
    assert result['extensions'] == [] and result['converged'] is True
    assert result['frozen_parameter_audit_pass'] is True
    assert all(torch.equal(parameter, base_before[name]) for name, parameter in model.base.named_parameters())
