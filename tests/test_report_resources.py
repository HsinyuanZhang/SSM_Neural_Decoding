import pytest

from ssm_decode.report import _resource_index


def _model(task: str, macs: int, state_bytes: int, seed: int = 0) -> dict:
    inputs, outputs = {"m1": (64, 16), "m2": (96, 2)}[task]
    return {"checkpoint": f"results/round1/{task}/diag_s{seed}.pt", "kind": "diag",
            "config": {"input_size": inputs, "output_size": outputs},
            "resources": {"macs_per_step": {"total": macs}, "state_bytes": state_bytes}}


def test_resource_join_keeps_task_specific_costs_and_rejects_seed_disagreement():
    index = _resource_index([_model("m1", 5184, 256), _model("m2", 6336, 256)])
    assert index[("M1", "diag")]["macs_per_step"]["total"] == 5184
    assert index[("M2", "diag")]["macs_per_step"]["total"] == 6336
    with pytest.raises(ValueError, match="inconsistent"):
        _resource_index([_model("m1", 5184, 256), _model("m1", 6000, 256, seed=1)])
