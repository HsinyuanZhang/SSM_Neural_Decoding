"""Integration checks for PEFT prefix selection and checkpoint replay."""
from types import SimpleNamespace
from pathlib import Path
from ssm_decode.data import Recording
import numpy as np
import torch
from torch import nn
import ssm_decode.peft_experiment as pe


class _Toy(nn.Module):
    def __init__(self, inp, out, width=4, layers=1, state_size=2, dropout=0.0):
        super().__init__()
        self.config = SimpleNamespace(output_size=out)
        self.in_proj = nn.Linear(inp, width)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, out)
    def forward(self, x):
        z = self.in_proj(x)
        for block in self.blocks: z = block(z)
        return self.out_proj(self.final_norm(z))


def _record(query_shift=0.0):
    trials, length = 34, 60
    rng = np.random.default_rng(4)
    x = rng.normal(size=(trials * length, 3)).astype("float32")
    y = (x[:, :2] * .4).astype("float32")
    y[33 * length:] += query_shift
    return Recording('m1', 'held_in', 'target', Path('unused.nwb'), x, y,
        np.zeros(len(x), dtype=bool), np.ones(len(x), dtype=bool),
        tuple((i * length, (i + 1) * length) for i in range(trials)))


def _args(pretrained, output):
    return SimpleNamespace(task="m1", pretrained=str(pretrained), output=str(output), method="io", device="cpu",
        steps=3, context=50, batch_size=2, eval_batch_size=16, val_interval=1, rank=2, alpha=2.,
        lr=1e-2, weight_decay=0., seed=17, policy="trial_causal_fixed_window", data_root="unused")


def test_prefix_selection_ignores_query_labels_and_replays_best(monkeypatch, tmp_path):
    torch.manual_seed(8)
    source = _Toy(3, 2)
    pretrained = tmp_path / "source.pt"
    torch.save({"args":{"width":4,"layers":1,"state_size":2,"dropout":0.},"state_dict":source.state_dict()}, pretrained)
    np.savez(tmp_path / "normalizer.npz", x_mean=np.zeros(3,np.float32), x_std=np.ones(3,np.float32),
             y_mean=np.zeros(2,np.float32), y_std=np.ones(2,np.float32))
    current={"record":_record()}
    plan={"cross_session_local_dev":{"target_session":"target"}}
    monkeypatch.setattr(pe, "build_mamba3_official", _Toy)
    monkeypatch.setattr(pe, "source_target_plan", lambda *a, **k: plan)
    monkeypatch.setattr(pe, "load_recording", lambda *a, **k: current["record"])
    one=pe.run(_args(pretrained,tmp_path/"one"))
    first=torch.load(one/"best.pt",map_location="cpu",weights_only=False)
    current["record"]=_record(query_shift=999.)
    two=pe.run(_args(pretrained,tmp_path/"two"))
    second=torch.load(two/"best.pt",map_location="cpu",weights_only=False)
    assert first["best_step"] == second["best_step"]
    assert all(torch.equal(first["state_dict"][k],second["state_dict"][k]) for k in first["state_dict"])
    saved,_=pe.load_saved_model(one/"best.pt", "cpu")
    original=_Toy(3,2); original.load_state_dict(first["state_dict"])
    probe=torch.randn(2,50,3)
    assert torch.equal(saved(probe),original(probe))
    m1=__import__("json").load(open(one/"metrics.json")); m2=__import__("json").load(open(two/"metrics.json"))
    assert m1["best_step"] == m2["best_step"] and m1["final"] != m2["final"]
