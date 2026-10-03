import torch
import numpy as np
from types import SimpleNamespace

import ssm_decode.debug_experiment as debug
from ssm_decode.debug_experiment import (_forward, _model, _right_aligned,
                                         _sample_batch, _val, _validation_selection)


def test_debug_runner_model_adapter_masked_batched_validation_restores_train_mode():
    class Args:
        kind = "s4d"; width = 16; layers = 1; state_size = 4; dropout = 0.0
    model = _model(Args(), 3, 2)
    x, y = torch.randn(192, 3), torch.randn(192, 2)
    mask = torch.zeros(192, dtype=torch.bool); mask[64:] = True
    prediction = _forward(model, x[:128].unsqueeze(0))
    assert prediction.shape == (1, 128, 2)
    assert torch.isfinite(prediction).all()
    windows = [(x, y, mask)]
    selected, digest = _validation_selection(windows, 128, 8)
    assert len(selected) == 8 and len(digest) == 64
    model.train()
    assert torch.isfinite(torch.tensor(_val(model, windows, 128, selected)))
    assert model.training


def test_short_trials_are_left_padded_and_sampling_uses_only_masked_valid_endpoints():
    x, y = torch.ones(12, 3), torch.ones(12, 2)
    mask = torch.zeros(12, dtype=torch.bool); mask[-2:] = True
    xx, yy, mm = _right_aligned([(x, y, mask)], [(0, 11)], 128)
    assert xx.shape == (1, 128, 3)
    assert not mm[:, :116].any() and mm[0, -1]
    xx, yy, mm = _sample_batch([(x, y, mask)], [[10, 11]], 128, 32)
    assert xx.shape[0] == 32
    assert mm[:, -1].all()


def test_split_proportions_and_io_freeze_leave_ssm_block_frozen():
    bounds = tuple((i * 60, (i + 1) * 60) for i in range(10))
    train, validation, query = debug._split_60_20_20(bounds)
    assert (len(train), len(validation), len(query)) == (6, 2, 2)
    class Args:
        kind = "s4d"; width = 16; layers = 1; state_size = 4; dropout = 0.0
    model = _model(Args(), 3, 2)
    debug._freeze_io(model)
    states = dict(model.named_parameters())
    assert states["input.weight"].requires_grad and states["readout.weight"].requires_grad
    assert not states["blocks.0.inp.weight"].requires_grad


def test_io_freeze_does_not_unfreeze_nested_mamba_projection_names():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.in_proj=torch.nn.Linear(4,4); self.out_proj=torch.nn.Linear(4,4)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.in_proj=torch.nn.Linear(3,4); self.blocks=torch.nn.ModuleList([Core()]); self.final_norm=torch.nn.LayerNorm(4); self.out_proj=torch.nn.Linear(4,2)
    model=Model(); debug._freeze_io(model); states=dict(model.named_parameters())
    assert states['in_proj.weight'].requires_grad and states['out_proj.weight'].requires_grad
    assert not states['blocks.0.in_proj.weight'].requires_grad
    assert not states['blocks.0.out_proj.weight'].requires_grad


def test_warmstart_statistics_are_loaded_from_checkpoint_parent(tmp_path):
    checkpoint = tmp_path / "best.pt"; checkpoint.write_bytes(b"placeholder")
    np.savez(tmp_path / "normalizer.npz", x_mean=np.array([7.]), x_std=np.array([2.]),
             y_mean=np.array([3.]), y_std=np.array([4.]))
    stats, path = debug._warm_stats(checkpoint)
    assert path.endswith("normalizer.npz") and stats[0].tolist() == [7.]


def test_ridge_calibration_receives_support_labels_not_query_labels(monkeypatch):
    target = SimpleNamespace(
        neural=np.zeros((200, 1), dtype=np.float32),
        behavior=np.arange(200, dtype=np.float32)[:, None],
        trial_bounds=((0, 200),), eval_mask=np.ones(200, dtype=bool),
    )
    captured = {}

    monkeypatch.setattr(debug, "split_support_query", lambda *args, **kwargs: (
        np.array([0], dtype=np.int64), np.array([60], dtype=np.int64)))
    monkeypatch.setattr(debug, "_predict_endpoints", lambda *args, **kwargs: np.zeros((51, 1), dtype=np.float32))
    def fake_ridge(prediction, residual):
        captured["residual"] = residual.numpy().copy()
        return lambda query_prediction: torch.zeros_like(query_prediction)
    monkeypatch.setattr(debug, "fit_ridge", fake_ridge)
    model = SimpleNamespace(config=SimpleNamespace(output_size=1))
    cache = debug._target_predictions(model, target,
                                      (np.zeros(1, np.float32), np.ones(1, np.float32),
                                       np.zeros(1, np.float32), np.ones(1, np.float32)),
                                      torch.device("cpu"), 50)
    result = debug._score_target(cache, (np.zeros(1, np.float32), np.ones(1, np.float32),
                                          np.zeros(1, np.float32), np.ones(1, np.float32)), "ridge")
    assert captured["residual"].shape == (50, 1)
    assert 109 not in captured["residual"]
    assert result["n_query_bins"] == 1


def test_mock_end_to_end_run_writes_json_and_both_query_sets(monkeypatch, tmp_path):
    def record(session, trials):
        length = trials * 55
        return SimpleNamespace(session=session, neural=np.random.randn(length, 2).astype("float32"),
                               behavior=np.random.randn(length, 1).astype("float32"),
                               eval_mask=np.ones(length, dtype=bool),
                               trial_bounds=tuple((i * 55, (i + 1) * 55) for i in range(trials)))
    source, target = record("source", 10), record("target", 40)
    plan = {"source_held_in_sessions": ["source"],
            "cross_session_local_dev": {"target_session": "target"}, "held_out_accessed": False}
    monkeypatch.setattr(debug, "source_target_plan", lambda *a, **k: plan)
    monkeypatch.setattr(debug, "load_recording", lambda task, split, session, root: source if session == "source" else target)
    args = SimpleNamespace(task="m1", output=str(tmp_path / "run"), device="cpu", kind="gru", mode="cross_session",
                           width=4, layers=1, state_size=4, dropout=0., context=50, batch_size=2, eval_batch_size=8,
                           steps=20, val_interval=10, max_val_endpoints=16, lr=1e-3, weight_decay=0., seed=0,
                           warmstart=None, adapt_parameters="all", data_root=str(tmp_path))
    output = debug.run(args)
    metrics = __import__("json").loads((output / "metrics.json").read_text())
    manifest = __import__("json").loads((output / "manifest.json").read_text())
    npz = np.load(output / "target_query_predictions.npz")
    assert metrics["status"] == "completed" and metrics["final"]["zero"]["n_query_bins"] == 42
    assert metrics["final"]["zero"]["all_valid_query_trial_n_bins"] == 385
    assert manifest["source_val_fixed_endpoint_count"] <= 16
    assert set(npz.files) >= {"legacy_query_indices", "all_valid_query_indices", "ridge_all_valid_prediction_physical"}
