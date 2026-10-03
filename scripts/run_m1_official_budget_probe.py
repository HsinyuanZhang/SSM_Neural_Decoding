"""Probe the official M1 M10 support budget before any private submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_session_pretraining_iteration import adaptation_matrix


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_write(path, value):
    if path.exists() and read(path) != value:
        raise RuntimeError(f"Existing frozen probe differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", required=True, type=Path)
    parser.add_argument("--stage-b-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    a, b, out = (path.resolve() for path in (args.stage_a_root, args.stage_b_root, args.output_root))
    config_path = ROOT / "configs/official_m1_m10_probe_20261003.json"
    cfg, bcfg = read(config_path), read(b / "matrix.json")
    # Authenticate the parent's prefix-only selection before copying any LR.
    from scripts.run_cross_session_iteration import _select
    if read(a / "selections.json") != _select(read(a / "matrix.json"), a, read(a / "provenance.json"), persist=False):
        raise RuntimeError("Stage A prefix selection does not authenticate")
    # Official M1 marks real trial starts. The pilot adds a leading segment
    # from bin zero, which can be a neural-only pretrial interval. It must not
    # consume one of the ten real-trial calibration slots.
    from ssm_decode.data import load_recording
    original_manifest = read(a / "m1/none/seed0/lr0/manifest.json")
    target = load_recording("m1", "held_in", original_manifest["target_session"],
                            root=Path(read(a / "matrix.json")["data_root"]))
    starts = np.flatnonzero(np.asarray(target.trial_change).reshape(-1).astype(bool))
    leading = int(starts[0] > 0)
    support_segments = cfg["official_nwb_trial_budget"] + leading
    if any(end - start < 50 for start, end in target.trial_bounds[:support_segments]):
        raise RuntimeError("Usable-trial filtering would change the legal M10 raw-trial budget")
    if leading and target.eval_mask[:starts[0]].any():
        raise RuntimeError("The leading pretrial segment unexpectedly contains supervised labels")
    frozen_write(out / "probe_contract.json", dict(cfg,
        config_sha256=digest(config_path), launcher_sha256=digest(__file__),
        stage_a_provenance_sha256=digest(a / "provenance.json"),
        stage_b_provenance_sha256=digest(b / "provenance.json"),
        pilot_support_segments=support_segments, leading_neural_only_segment=bool(leading)))
    for profile in cfg["profiles"]:
        matrix = adaptation_matrix(bcfg, a, b, "m1", profile)
        matrix.update(name=f"official_m1_m10_development_{profile}", support_trials=support_segments,
                      query_cutoff_trials=33, validation_split="interleaved")
        matrix["methods"] = {method: matrix["methods"][method] for method in cfg["methods"]}
        matrix["official_budget_probe"] = dict(cfg)
        path = out / "matrices" / f"{profile}.json"
        frozen_write(path, matrix)
        with path.with_suffix(".log").open("w") as stream:
            subprocess.run([sys.executable, str(ROOT / "scripts/run_cross_session_iteration.py"),
                            "--matrix", str(path), "--output-root", str(out / profile),
                            "--phase", "all"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
