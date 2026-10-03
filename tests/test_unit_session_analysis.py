"""CPU negative checks for the read-only unit/session analysis verifier."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_unit_session_iteration.py"
SPEC = importlib.util.spec_from_file_location("unit_session_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _metric(indices, truth):
    return {
        "indices_sha256": analysis.array_sha(indices),
        "truth_sha256": analysis.array_sha(truth),
    }


def test_query_archive_requires_exact_dtypes_finiteness_and_canonical_hashes(tmp_path):
    indices = np.array([1, 3], dtype=np.int64)
    truth = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    prediction = truth + .25
    path = tmp_path / "query.npz"
    np.savez(path, indices=indices, truth=truth, prediction=prediction)
    task = {"expected_query_bins": 2, "expected_indices_sha256": analysis.array_sha(indices),
            "expected_truth_sha256": analysis.array_sha(truth)}
    loaded = analysis.validate_query_archive(path, task, _metric(indices, truth), "good")
    assert np.array_equal(loaded["indices"], indices)
    np.savez(path, indices=indices.astype(np.int32), truth=truth, prediction=prediction)
    with pytest.raises(ValueError, match="invalid query indices"):
        analysis.validate_query_archive(path, task, _metric(indices, truth), "int32")
    np.savez(path, indices=indices, truth=truth, prediction=np.full_like(truth, np.nan))
    with pytest.raises(ValueError, match="invalid query truth or prediction"):
        analysis.validate_query_archive(path, task, _metric(indices, truth), "nan")


def test_seal_rejects_uppercase_digest_and_fold_receipt_mismatch(tmp_path):
    seal = tmp_path / "fit_seal.json"
    content = '{"schema":"unit_session_fit_seal_v1","tasks":{"m1":{"completion_sha256":"' + "a" * 64 + '","contract_sha256":"' + "b" * 64 + '"},"m2":{"completion_sha256":"' + "c" * 64 + '","contract_sha256":"' + "d" * 64 + '"}}}'
    seal.write_text(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        analysis.verify_seal(tmp_path, {"tasks": {"m1": {}, "m2": {}}}, digest.upper())
    with pytest.raises(ValueError, match="inconsistent fold receipt"):
        analysis.validate_fold_receipt({"fold_accepted": True, "point_allclose_pass": False, "r2_delta": 0.}, "bad")


def test_scores_closure_rejects_unexpected_archive_and_uses_canonical_fit_name(tmp_path):
    task = tmp_path / "m1"
    row = {"method": "ui", "seed": 2, "lr": .003}
    expected = analysis.expected_archive_path(task, "unit", row)
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"expected")
    analysis.verify_scores_closure(task, {expected})
    extra = task / "scores" / "unit" / "wrong_name.npz"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="scores archive closure differs"):
        analysis.verify_scores_closure(task, {expected})
