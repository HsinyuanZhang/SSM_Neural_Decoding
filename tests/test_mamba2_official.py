import json

from ssm_decode.mamba2_official import cpu_import_constructor_attempt, write_cpu_compatibility


def test_mamba2_cpu_import_and_constructor_only(tmp_path):
    result = cpu_import_constructor_attempt()
    # An environment dependency failure is represented with its full causal
    # chain rather than hidden by a substitute implementation.
    assert result["status"] in {"cpu_import_constructor_supported", "cpu_import_or_constructor_unavailable"}
    if result["status"] == "cpu_import_constructor_supported":
        assert result["observed_core"]["use_mem_eff_path"] is False
        assert result["observed_core"]["headdim"] == 64
        assert result["observed_core"]["d_state"] == 16
        assert result["observed_core"]["dt_bias_finite"] is True
    else:
        assert len(result["exception_chain"]) >= 1
    saved = write_cpu_compatibility(tmp_path / "mamba2.json")
    assert json.loads((tmp_path / "mamba2.json").read_text())["status"] == saved["status"]
