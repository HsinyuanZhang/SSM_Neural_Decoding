import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("m2_probe", ROOT / "scripts/run_m2_official_budget_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class Record:
    def __init__(self):
        self.neural = np.arange(16, dtype=np.float32).reshape(8, 2)
        self.behavior = np.arange(16, dtype=np.float32).reshape(8, 2)
        self.path = Path("raw")
        self.eval_mask = np.ones(8, bool)
        self.trial_bounds = [(0, 8)]


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path, monkeypatch, mutate_on_fit=None, mutation_target="source"):
    stage_a, stage_b, output = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    source_dir = stage_b / "source"
    source_dir.mkdir(parents=True)
    source = source_dir / "best.pt"
    source.write_bytes(b"source-v1")
    np.savez(source_dir / "normalizer.npz", x_mean=np.zeros(2), x_std=np.ones(2), y_mean=np.zeros(2), y_std=np.ones(2))
    put_json(source_dir / "manifest.json", {"ok": True})
    put_json(stage_a / "matrix.json", {"data_root": str(tmp_path / "data")})
    put_json(stage_a / "provenance.json", {"a": 1})
    put_json(stage_b / "matrix.json", {"capacities": [{"name": "wide"}]})
    put_json(stage_b / "provenance.json", {"b": 1})
    formal_fit = stage_b / "adapt/wide/m2/m2/none/seed0/lr0"
    none_fit = {"seed": 0, "fit": str(formal_fit), "best_pt_sha256": "formal-none-best"}
    selection = {"tasks": {"m2": {"methods": {
        "none": {"selected": {"lr": 0.0, "fits": [none_fit]}}, "lora": {"selected": {"lr": 0.003}},
    }}}}
    put_json(stage_a / "selections.json", selection)
    put_json(stage_a / "m2/none/seed0/lr0/manifest.json", {"target_session": "target"})
    raw = tmp_path / "raw.nwb"; raw.write_bytes(b"raw")
    calibration = SimpleNamespace(path=raw, neural=np.zeros((4, 2), np.float32), behavior=np.zeros((4, 2), np.float32),
        receipt={"rawfile_sha256": probe.digest(raw), "realtrialbounds": [[0, 1]] * 33, "query_labels_used": False})
    archive = stage_b / "adapt/wide/m2/m2/none/seed0/lr0/query_evaluation/predictions.npz"
    archive.parent.mkdir(parents=True)
    record = Record(); record.path = raw; indices = np.array([1, 3, 5], np.int64); historical = np.full((3, 2), -9, np.float32)
    np.savez(archive, allvalid_indices=indices, truth_allvalid=record.behavior[indices], zero_allvalid=historical)
    # The production n=14115 check is intentionally narrowed only in this CPU orchestration test.
    calls, query_reads = [], []
    monkeypatch.setattr(probe, "source_path", lambda *args: source_dir)
    monkeypatch.setattr(probe, "source_is_current", lambda *args: True)
    monkeypatch.setattr(probe, "verify_live", lambda *args: None)
    monkeypatch.setattr(probe, "_select", lambda *args, **kwargs: selection)
    evaluation_rows = []
    def evaluation_is_current(*args):
        evaluation_rows.append(args[-1])
        return args[-1] == none_fit
    monkeypatch.setattr(probe, "_evaluation_is_current", evaluation_is_current)
    monkeypatch.setattr(probe, "canonical_allvalid_indices", lambda *args: indices)
    monkeypatch.setattr(probe, "reader_identity", lambda: {"reader": "fake"})
    monkeypatch.setattr(probe, "_runtime_identity", lambda: {"runtime": "fake"})
    monkeypatch.setattr(probe, "load_public_calibration", lambda *args: calibration)
    monkeypatch.setattr(probe.torch, "load", lambda *args, **kwargs: {"args": {"kind": "mamba3_official", "width": 1, "layers": 1, "state_size": 1, "dropout": 0.0}, "state_dict": {}})

    def fake_fit(cal, src, stats, args, destination, identity):
        calls.append(args.method + str(args.seed)); destination.mkdir(parents=True)
        (destination / "model.pt").write_bytes(b"model-" + bytes(str(len(calls)), "ascii"))
        np.savez(destination / "normalizer.npz", x_mean=np.zeros(2), x_std=np.ones(2), y_mean=np.zeros(2), y_std=np.ones(2))
        put_json(destination / "receipt.json", {"ok": True})
        if mutate_on_fit == len(calls):
            {"source": source, "raw": raw, "archive": archive}[mutation_target].write_bytes(
                f"{mutation_target}-mutated".encode())
        return {"best_step": 1, "fold_r2_abs_delta": 0.0}
    monkeypatch.setattr(probe, "fit_bank", fake_fit)

    class Model:
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, *args, **kwargs): return None
    monkeypatch.setattr(probe.d, "_model", lambda *args: Model())
    monkeypatch.setattr(probe.d, "_predict_endpoints", lambda model, x, bounds, ends, *args, **kwargs: np.full((len(ends), 2), 7, np.float32))
    monkeypatch.setattr(probe, "r2", lambda *args: 0.5)
    formal = formal_fit / "query_evaluation"
    put_json(formal / "metrics.json", {"target_data_sha256": probe.digest(raw), "cohort_index_hashes": {"all_valid_indices": __import__('hashlib').sha256(indices.tobytes()).hexdigest()}, "cohort_truth_hashes": {"all_valid_truth_physical": __import__('hashlib').sha256(record.behavior[indices].tobytes()).hexdigest()}, "query_cutoff_trials": 33})
    put_json(stage_b / "adapt/wide/m2/matrix.json", {"tasks": ["m2"]})
    put_json(stage_b / "adapt/wide/m2/provenance.json", {"w": 1})
    put_json(stage_b / "adapt/wide/m2/selections.json", selection)
    original_load = probe.np.load
    def guarded_load(path, *args, **kwargs):
        if Path(path) == archive:
            assert len(calls) == 4
            query_reads.append("archive")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(probe.np, "load", guarded_load)
    def guarded_record(*args, **kwargs):
        assert len(calls) == 4
        query_reads.append("record")
        return record
    monkeypatch.setattr(probe, "load_recording", guarded_record)
    monkeypatch.setattr(sys, "argv", ["probe", "--stage-a-root", str(stage_a), "--stage-b-root", str(stage_b), "--output-root", str(output)])
    return output, calls, query_reads, historical, evaluation_rows, none_fit


def test_all_four_fits_precede_query_and_zero_is_current_prediction(tmp_path, monkeypatch):
    output, calls, query_reads, historical, evaluation_rows, none_fit = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, "COHORT_N", 3)
    probe.main()
    assert calls == ["none0", "lora0", "lora1", "lora2"]
    assert query_reads == ["record", "archive"]
    assert evaluation_rows == [none_fit]
    with np.load(output / "scores/none_seed0.npz") as score:
        assert np.array_equal(score["zero_allvalid"], score["prediction_allvalid"])
        assert not np.array_equal(score["zero_allvalid"], historical)


def test_none_selection_requires_exactly_one_fit_row(tmp_path, monkeypatch):
    output, calls, query_reads, _, _, none_fit = setup(tmp_path, monkeypatch)
    stage_a_selection = tmp_path / "a/selections.json"
    stage_b_selection = tmp_path / "b/adapt/wide/m2/selections.json"
    selection = json.loads(stage_a_selection.read_text())
    selection["tasks"]["m2"]["methods"]["none"]["selected"]["fits"].append(dict(none_fit))
    put_json(stage_a_selection, selection)
    put_json(stage_b_selection, selection)
    monkeypatch.setattr(probe, "_select", lambda *args, **kwargs: selection)
    with pytest.raises(RuntimeError, match="exactly one fit"):
        probe.main()
    assert calls == []
    assert query_reads == []
    assert not output.exists()


@pytest.mark.parametrize("mutation_target", ["source", "raw", "archive"])
def test_late_frozen_input_change_aborts_before_query(tmp_path, monkeypatch, mutation_target):
    output, calls, query_reads, _, _, _ = setup(tmp_path, monkeypatch, mutate_on_fit=2,
                                                  mutation_target=mutation_target)
    monkeypatch.setattr(probe, "COHORT_N", 3)
    with pytest.raises(RuntimeError, match="frozen|raw file|metrics/archive"):
        probe.main()
    assert calls == ["none0", "lora0"]
    assert query_reads == []
    assert not (output / "summary.json").exists()


def test_canonical_oracle_skips_first_33_usable_trials_and_short_trials():
    record = SimpleNamespace(trial_bounds=[(0, 49)] + [(50 + 60*i, 110 + 60*i) for i in range(34)],
                             eval_mask=np.ones(50 + 60*34, bool))
    result = probe.canonical_allvalid_indices(record, 33)
    assert result[0] == 50 + 60*33


def test_float_archive_indices_are_rejected_before_integer_coercion(tmp_path, monkeypatch):
    output, calls, query_reads, _, _, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, "COHORT_N", 3)
    archive = tmp_path / "b/adapt/wide/m2/m2/none/seed0/lr0/query_evaluation/predictions.npz"
    indices = np.array([1, 3, 5], dtype=np.float64)
    behavior = np.arange(16, dtype=np.float32).reshape(8, 2)
    np.savez(archive, allvalid_indices=indices, truth_allvalid=behavior[indices.astype(np.int64)],
             zero_allvalid=np.full((3, 2), -9, np.float32))
    with pytest.raises(RuntimeError, match="1-D integer"):
        probe.main()
    assert calls == ["none0", "lora0", "lora1", "lora2"]
    assert query_reads == ["record", "archive"]
    assert not (output / "summary.json").exists()
