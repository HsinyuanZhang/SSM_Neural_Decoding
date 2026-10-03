import argparse
from dataclasses import replace
import hashlib
import json

import numpy as np
import torch
from torch import nn

from ssm_decode import official_payload as export
from ssm_decode.official_calibration import PublicCalibration


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(64, 8)
        core = nn.Module()
        core.ssm = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([core])
        self.final_norm = nn.LayerNorm(8)
        self.out_proj = nn.Linear(8, 16)

    def forward(self, x):
        hidden = self.in_proj(x)
        hidden = hidden + self.blocks[0].ssm(hidden)
        return self.out_proj(self.final_norm(hidden))


def test_public_fit_ignores_pretrial_labels_and_preserves_frozen_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_base_model", lambda *args: ToyDecoder())
    torch.manual_seed(12)
    source = {"args": {}, "state_dict": ToyDecoder().state_dict()}
    rng = np.random.default_rng(3)
    x = rng.normal(size=(45, 64)).astype(np.float32)
    y = rng.normal(size=(45, 16)).astype(np.float32)
    bounds = tuple((5 + 4*i, 9 + 4*i) for i in range(10))
    mask = np.ones(45, bool)
    mask[:5] = False
    path = tmp_path / "public.nwb"
    path.write_bytes(b"synthetic public calibration identity")
    receipt = {"rawfile_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record = PublicCalibration("m1", "held_out", "fixture", path, x, y, mask, np.zeros(45, bool), bounds, receipt)
    poisoned = y.copy()
    poisoned[:5] = 1e8
    stats = (np.zeros(64, np.float32), np.ones(64, np.float32),
             np.zeros(16, np.float32), np.ones(16, np.float32))
    args = argparse.Namespace(device="cpu", seed=0, method="io", lr=1e-3, steps=3, max_steps=3)
    for name, rec in (("clean", record), ("poison", replace(record, behavior=poisoned))):
        try:
            export.fit_bank(rec, source, stats, args, tmp_path / name, {"fixture": True})
        except RuntimeError as error:
            # Three optimizer steps can end at a late validation maximum.
            # Such a fit must retain diagnostic evidence and reject export.
            assert str(error) == "Public bank fails convergence or merged-score acceptance"
    clean = torch.load(tmp_path / "clean/unmerged_best.pt", weights_only=False)
    poison = torch.load(tmp_path / "poison/unmerged_best.pt", weights_only=False)
    assert clean["best_step"] == poison["best_step"]
    assert all(torch.equal(clean["state_dict"][key], poison["state_dict"][key]) for key in clean["state_dict"])
    assert (tmp_path / "clean/train_log.json").read_text() == (tmp_path / "poison/train_log.json").read_text()
    saved = json.loads((tmp_path / "clean/receipt.json").read_text())
    assert saved["frozen_audit"]["status"] == "passed"
    assert saved["frozen_audit"]["count"] == 2
    assert torch.equal(clean["state_dict"]["blocks.0.ssm.weight"], source["state_dict"]["blocks.0.ssm.weight"])
    assert saved["adaptation"]["root_input_bias_trainable"] is True
    assert saved["query_labels_used"] is False
    assert saved["fold_allclose"] is True
    assert (tmp_path / "clean/model.pt").exists() == saved["convergence_satisfied"]
    train, valid = export.partitions(record)
    assert set(train).isdisjoint(valid)
    assert (0, 5) in train
    # Validation is the same encoded-segment rule as the legal M10 probe.
    assert valid == [bounds[3], bounds[8], bounds[9]]


def test_public_full_tuning_keeps_stage_boundaries_and_selected_frozen_core(tmp_path, monkeypatch):
    class ThreeBlockDecoder(ToyDecoder):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([self.blocks[0], self.blocks[0].__class__(), self.blocks[0].__class__()])
            for block in self.blocks[1:]:
                block.ssm = nn.Linear(8, 8)

        def forward(self, x):
            hidden = self.in_proj(x)
            for block in self.blocks:
                hidden = hidden + block.ssm(hidden)
            return self.out_proj(self.final_norm(hidden))

    monkeypatch.setattr(export, "_base_model", lambda *args: ThreeBlockDecoder())
    # The prefix selects step 300, before all-core unfreezing at step 400.
    scores = iter([0., .1, .2, .3, .2, .1, .1, .3])
    monkeypatch.setattr(export.d, "_val", lambda *args: next(scores))
    torch.manual_seed(12)
    source = {"args": {}, "state_dict": ThreeBlockDecoder().state_dict()}
    rng = np.random.default_rng(3)
    path = tmp_path / "public.nwb"
    path.write_bytes(b"public staged full-tuning fixture")
    record = PublicCalibration("m1", "held_out", "fixture", path,
        rng.normal(size=(40, 64)).astype(np.float32),
        rng.normal(size=(40, 16)).astype(np.float32), np.ones(40, bool),
        np.zeros(40, bool), tuple((4*i, 4*i+4) for i in range(10)),
        {"rawfile_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    stats = (np.zeros(64, np.float32), np.ones(64, np.float32),
             np.zeros(16, np.float32), np.ones(16, np.float32))
    args = argparse.Namespace(device="cpu", seed=0, method="full", lr=1e-3, steps=501, max_steps=501)
    receipt = export.fit_bank(record, source, stats, args, tmp_path / "fit", {"fixture": True})
    assert [stage["step"] for stage in receipt["optimizer_stages"]] == [0, 200, 400]
    assert all(row["status"] == "passed" for row in receipt["stage_freeze_audits"])
    groups = receipt["optimizer_stages"][-1]["groups"]
    names = [name for group in groups for name in group["names"]]
    assert len(names) == len(set(names))
    assert all(f"blocks.{i}.ssm.weight" in names for i in range(3))
    selected = torch.load(tmp_path / "fit/model.pt", weights_only=False)["state_dict"]
    assert receipt["best_step"] == 300
    assert torch.equal(selected["blocks.0.ssm.weight"], source["state_dict"]["blocks.0.ssm.weight"])
    assert not torch.equal(selected["blocks.2.ssm.weight"], source["state_dict"]["blocks.2.ssm.weight"])
