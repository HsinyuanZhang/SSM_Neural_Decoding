import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from ssm_decode.models import ModelConfig, build_model

from ssm_decode.fixedpoint import (Q_SCALE, Real2x2FixedPoint, Real2x2ModeBank, quantize_q14, quantize_stable_matrix,
                                   run_zero_input_characterization, spectral_radius_2x2_q14)

ROOT = Path(__file__).resolve().parents[1]
LOCAL_IVL = ROOT / ".tools" / "iverilog" / "usr" / "bin" / "iverilog"
LOCAL_VVP = ROOT / ".tools" / "iverilog" / "usr" / "bin" / "vvp"
LOCAL_IVL_LIB = ROOT / ".tools" / "iverilog" / "usr" / "lib" / "x86_64-linux-gnu" / "ivl"

def test_rounding_is_nearest_ties_away_and_saturates():
    mode = Real2x2FixedPoint((Q_SCALE - 1, 0, 0, Q_SCALE - 1), (0, Q_SCALE), 16)
    assert mode.step(1) == (0, 1)
    assert mode.step(-1) == (0, 0)  # 0.99994 rounds toward the nearest integer
    driven = Real2x2FixedPoint((0, 0, 0, 0), (Q_SCALE, 0), 16)
    assert driven.step(32767) == (32767, 0)
    assert driven.step(32767) == (32767, 0)
    assert driven.saturation_count == 0

def test_quantized_matrix_must_be_strictly_stable():
    safe = quantize_stable_matrix((0.95, -0.1, 0.1, 0.95))
    assert spectral_radius_2x2_q14(safe) < 1.0
    with pytest.raises(ValueError, match="unsafe"):
        quantize_stable_matrix((1.0, 0.0, 0.0, 1.0))

def test_zero_input_100k_characterization_has_no_saturation():
    mode = Real2x2FixedPoint((15565, -2344, 2344, 15565), (0, Q_SCALE), 24, p=1_000_000, q=-700_000)
    report = run_zero_input_characterization(mode)
    assert report["saturation_count"] == 0
    # A damped rotation can temporarily redistribute magnitude between coordinates.
    assert report["peak_abs_state"] <= 1_100_000
    assert report["repeat"] is not None

def test_json_export_is_integer_register_layout(tmp_path):
    mode = Real2x2FixedPoint((15565, -2344, 2344, 15565), (0, Q_SCALE), 24)
    target = tmp_path / "mode.json"; mode.export_json(target)
    payload = json.loads(target.read_text())
    assert payload["a_q14"] == [15565, -2344, 2344, 15565]
    assert payload["register_layout"][0]["bits"] == 24
    assert payload["format"]["rounding"] == "nearest, ties away from zero"

def test_mode_bank_accepts_per_mode_a_arrays():
    bank = Real2x2ModeBank([(15565, -2344, 2344, 15565), (14746, 0, 0, 14746)], state_bits=16)
    assert bank.step([1, -2]) == [(0, 1), (0, -2)]
    assert bank.export_layout()["mode_count"] == 2

def test_width64_bank_export_matches_actual_model_coefficients_and_injection_contract():
    payload = json.loads((ROOT / "hardware" / "bank_w64_q14.json").read_text())
    model = build_model(ModelConfig(96, 2, width=64, kind="bank")).eval()
    actual = [[quantize_q14(float(v)) for row in matrix for v in row]
              for matrix in model.recurrence_matrices().detach().cpu().tolist()]
    assert payload["shape"] == [64, 4]
    assert payload["a_q14"] == actual
    assert payload["b_q14_per_mode"] == [Q_SCALE, 0]
    assert payload["allocated_state_values"] == 128
    assert len({tuple(x) for x in actual}) == 4
    drives = [int(row.split()[0]) for row in (ROOT / "hardware" / "fixedpoint_vectors.txt").read_text().splitlines()]
    for bits in (16, 24, 32):
        bank = Real2x2ModeBank(actual, [(Q_SCALE, 0)] * 64, state_bits=bits)
        for drive in drives:
            bank.step([drive] * 64)
        reference = payload["integer_bank_reference_same_256_drives"][f"int{bits}"]
        assert sum(mode.saturation_count for mode in bank.modes) == reference["saturation_count_total"]
        assert [bank.modes[0].p, bank.modes[0].q] == reference["first_mode_final_state"]
        assert [bank.modes[-1].p, bank.modes[-1].q] == reference["last_mode_final_state"]

def test_generated_vectors_match_python_reference(tmp_path):
    script = ROOT / "hardware" / "generate_vectors.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    lines = [x for x in (ROOT / "hardware" / "fixedpoint_vectors.txt").read_text().splitlines() if not x.startswith("#")]
    assert len(lines) == 256
    mode = Real2x2FixedPoint((15565, -2344, 2344, 15565), (0, Q_SCALE), 16)
    for row in lines:
        u, p, q, sat = map(int, row.split())
        assert (*mode.step(u), mode.saturation_count) == (p, q, sat)

@pytest.mark.skipif(not (LOCAL_IVL.is_file() and LOCAL_VVP.is_file() and LOCAL_IVL_LIB.is_dir()) and shutil.which("iverilog") is None,
                    reason="iverilog unavailable; RTL syntax/simulation unchecked")
def test_rtl_is_bitexact_against_generated_vectors(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "hardware" / "generate_vectors.py")], cwd=ROOT, check=True)
    binary = tmp_path / "sim.out"
    compiler = str(LOCAL_IVL) if LOCAL_IVL.is_file() else "iverilog"
    runtime = str(LOCAL_VVP) if LOCAL_VVP.is_file() else "vvp"
    compile_cmd = [compiler]
    if LOCAL_IVL_LIB.is_dir():
        compile_cmd.extend(["-B", str(LOCAL_IVL_LIB)])
    compile_cmd.extend(["-g2012", "-o", str(binary), "hardware/real2x2_ssm_engine.sv", "hardware/tb_real2x2_ssm_engine.sv"])
    subprocess.run(compile_cmd, cwd=ROOT, check=True)
    run = subprocess.run([runtime, str(binary)], cwd=ROOT, text=True, capture_output=True, check=True)
    assert "PASS 256 fixed-point vectors" in run.stdout, run.stdout + run.stderr
