"""CPU contracts for source-only session pretraining utilities."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import ssm_decode.session_pretraining as pretraining

from ssm_decode.session_frontend import SessionFrontendBank, fold_session_input
from ssm_decode.session_pretraining import (SessionAwareDecoder, continuous_source_windows,
                                             sample_balanced_batch, source_statistics)


def record(seed, length=12):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(session=f"s{seed}", neural=rng.normal(size=(length, 3)).astype("float32"),
                           behavior=rng.normal(size=(length, 2)).astype("float32"),
                           eval_mask=np.array([True, False] * (length // 2)), trial_bounds=((0, 4), (4, 8), (8, length)))


def test_source_statistics_ignore_validation_and_future_values():
    base = record(1); splits = [([(0, 4), (4, 8)], [(8, 12)])]
    before = source_statistics([base], splits)
    changed = record(1)
    changed.neural[8:] += 1000; changed.behavior[8:] -= 1000
    changed.behavior[np.flatnonzero(~changed.eval_mask[:8])] += 999
    after = source_statistics([changed], splits)
    for first, second in zip(before[0], after[0]):
        assert np.array_equal(first, second)


def test_continuous_windows_keep_cross_trial_context_and_mask_labels():
    item = record(2); stats = source_statistics([item], [([(4, 8)], [(0, 4), (8, 12)])])
    window = continuous_source_windows([item], [[(4, 8)]], stats, "cpu")[0]
    x, y, mask = window
    assert torch.equal(x[:4], torch.from_numpy(((item.neural[:4]-stats[0][0])/stats[0][1]).astype("float32")))
    assert torch.isnan(y[:4]).all() and not mask[:4].any()
    assert torch.isnan(y[8:]).all() and not mask[8:].any()
    altered = record(2); altered.behavior[:4] += 999; altered.behavior[8:] -= 999
    alternate = continuous_source_windows([altered], [[(4, 8)]], stats, "cpu")[0]
    assert torch.equal(y[4:8], alternate[1][4:8]) and torch.equal(mask, alternate[2])


def test_balanced_sampler_returns_matching_session_indices():
    first, second = record(3), record(4)
    stats = source_statistics([first, second], [([(0, 8)], [(8, 12)]), ([(0, 8)], [(8, 12)])])
    windows = continuous_source_windows([first, second], [[(0, 8)], [(0, 8)]], stats, "cpu")
    eligible = [[3, 4, 5, 6, 7], [3, 4, 5, 6, 7]]
    np.random.seed(9); import random; random.seed(9)
    x, _, _, sessions = sample_balanced_batch(windows, eligible, 4, 30)
    assert set(sessions.tolist()) == {0, 1}
    for row, session in enumerate(sessions.tolist()):
        assert any(torch.equal(x[row, -1], windows[session][0][end]) for end in eligible[session])


class CausalBlock(nn.Module):
    def forward(self, z):
        return torch.cumsum(z, dim=1)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(input_size=3)
        self.in_proj = nn.Linear(3, 4); self.blocks = nn.ModuleList([CausalBlock()])
        self.final_norm = nn.Identity(); self.out_proj = nn.Linear(4, 2)

    def forward(self, x):
        z = self.in_proj(x)
        for block in self.blocks: z = block(z)
        return self.out_proj(self.final_norm(z))


def test_session_decoder_is_causal_and_matches_folded_session_input():
    torch.manual_seed(4); base = TinyBase(); bank = SessionFrontendBank(2, 3, 4)
    bank.gain.data[1].copy_(torch.tensor([1.5, .5, 2.])); bank.bias.data[1].normal_(); bank.embedding.data[1].normal_()
    model = SessionAwareDecoder(base, bank).eval(); x = torch.randn(1, 6, 3)
    changed = x.clone(); changed[:, 4:] += 50
    assert torch.equal(model(x, 1)[:, :4], model(changed, 1)[:, :4])
    folded = TinyBase(); folded.load_state_dict(base.state_dict()); folded.in_proj = fold_session_input(base.in_proj, bank, 1)
    assert torch.allclose(model(x, 1), folded(x), atol=2e-6, rtol=2e-6)


def test_cpu_toy_run_exports_plain_replay_without_target_load(monkeypatch, tmp_path):
    class ToyBase(TinyBase):
        def __init__(self, inputs, outputs, width, *unused):
            nn.Module.__init__(self); self.config = SimpleNamespace(input_size=inputs, output_size=outputs)
            self.in_proj = nn.Linear(inputs, width); self.blocks = nn.ModuleList([CausalBlock()])
            self.final_norm = nn.Identity(); self.out_proj = nn.Linear(width, outputs)
    source = record(20, length=30); source.trial_bounds = tuple((i, i+6) for i in range(0, 30, 6))
    source.path = tmp_path / "source.nwb"; source.path.write_bytes(b"source bytes")
    target = record(21, length=30); target.path = tmp_path / "target.nwb"; target.path.write_bytes(b"target bytes")
    calls = []
    def plan(task, root):
        return {"source_held_in_sessions": ["source"], "cross_session_local_dev": {"target_session": "target"}}
    def load(task, split, session, root):
        calls.append(session)
        if session == "target": raise AssertionError("source pretraining loaded a target recording")
        assert session == "source"; return source
    monkeypatch.setattr(pretraining, "source_target_plan", plan)
    monkeypatch.setattr(pretraining, "load_recording", load)
    monkeypatch.setattr(pretraining, "build_mamba3_official", lambda inputs, outputs, width, *args: ToyBase(inputs, outputs, width))
    monkeypatch.setattr(pretraining, "_runtime_identity", lambda: {"toy": True})
    output = tmp_path / "run"
    args = SimpleNamespace(task="m1", output=str(output), data_root=str(tmp_path), device="cpu", width=4, layers=1,
                           state_size=2, dropout=0., context=4, batch_size=2, eval_batch_size=8, steps=3,
                           val_interval=1, max_val_endpoints=20, seed=0, lr=.01, weight_decay=0., gain_sigma=0.,
                           offset_sigma=0., channel_dropout=0.)
    pretraining.run(args)
    assert calls == ["source"]
    exported = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert exported["args"] == vars(args) and "session_pretraining_receipt" in exported
    replay = ToyBase(3, 2, 4); replay.load_state_dict(exported["state_dict"])
    normalizer = np.load(output / "normalizer.npz")
    assert set(normalizer.files) == {"x_mean", "x_std", "y_mean", "y_std"}
    assert all(not key.startswith("bank.") for key in exported["state_dict"])
    # Target/query labels are inaccessible to the source-only loader. Their change cannot alter source weights.
    target.behavior += 100000
    second = tmp_path / "run_again"; args_again = SimpleNamespace(**{**vars(args), "output": str(second)})
    pretraining.run(args_again)
    again = torch.load(second / "best.pt", map_location="cpu", weights_only=False)
    assert calls == ["source", "source"]
    assert all(torch.equal(value, again["state_dict"][name]) for name, value in exported["state_dict"].items())
