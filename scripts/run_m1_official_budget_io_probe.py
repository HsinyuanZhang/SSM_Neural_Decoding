"""Run the IO-only M1 M10 probe after a prefix-only convergence rejection."""
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
from scripts.run_session_pretraining_iteration import adaptation_matrix, verify_live, source_is_current, source_path


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
    parser.add_argument("--failed-probe-root", required=True, type=Path)
    args = parser.parse_args()
    a, b, out = (path.resolve() for path in (args.stage_a_root, args.stage_b_root, args.output_root))
    config_path = ROOT / "configs/official_m1_m10_io_probe_20261003.json"
    cfg, bcfg = read(config_path), read(b / "matrix.json")
    failed = args.failed_probe_root.resolve()
    for parent in (a, b, failed):
        if out == parent or parent in out.parents or out in parent.parents:
            raise RuntimeError("IO-only probe must preserve all parent roots")
    parent_provenance = read(b / "provenance.json")
    verify_live(bcfg, parent_provenance, a)
    fits = [read(p) for p in (failed / "old").glob("*/*/seed*/lr*/fit_result.json")]
    queries = list(failed.rglob("query_evaluation"))
    rejected = [row for row in fits if row.get("method") == "affine_lora" and row.get("convergence_satisfied") is False]
    if len(fits) != 7 or queries or not rejected or any(row.get("status") != "completed" for row in fits):
        raise RuntimeError("Prior probe does not prove a prefix-only affine+LoRA rejection")
    from scripts.run_cross_session_iteration import _completed_fit_is_current
    failed_matrix = read(failed / "old/matrix.json")
    failed_provenance = read(failed / "old/provenance.json")
    if any(not _completed_fit_is_current(p.parent, failed_matrix, failed_provenance, "m1")
           for p in (failed / "old").glob("*/*/seed*/lr*/fit_result.json")):
        raise RuntimeError("Prior prefix fit identity or checkpoint hashes do not authenticate")
    def failed_tree():
        # Triton compilation files are the only ephemeral directory. All
        # checkpoints, logs, source snapshots, matrices and receipts are bound.
        return {str(p.relative_to(failed)): digest(p) for p in failed.rglob("*")
                if p.is_file() and ".triton_cache" not in p.relative_to(failed).parts}
    prior_identity = {"failed_probe_contract_sha256": digest(failed / "probe_contract.json"),
                      "failed_tree_sha256": failed_tree(),
                      "excluded_ephemeral_directory": ".triton_cache",
                      "prior_query_evaluations": len(queries)}
    # Authenticate the parent's prefix-only selection before copying any LR.
    from scripts.run_cross_session_iteration import _select
    authenticated_selection = read(a / "selections.json")
    if authenticated_selection != _select(read(a / "matrix.json"), a, read(a / "provenance.json"), persist=False):
        raise RuntimeError("Stage A prefix selection does not authenticate")
    locked_io_lr = authenticated_selection["tasks"]["m1"]["methods"]["io"]["selected"]["lr"]
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
    contract = dict(cfg,
        config_sha256=digest(config_path), launcher_sha256=digest(__file__),
        prior_probe=prior_identity,
        stage_a_provenance_sha256=digest(a / "provenance.json"),
        stage_b_provenance_sha256=digest(b / "provenance.json"),
        pilot_support_segments=support_segments, leading_neural_only_segment=bool(leading))
    frozen_write(out / "probe_contract.json", contract)
    for profile in cfg["profiles"]:
        if prior_identity["failed_tree_sha256"] != failed_tree():
            raise RuntimeError("Prior rejected-probe artifacts changed after freeze")
        verify_live(bcfg, parent_provenance, a)
        if profile != "old":
            capacity = next(row for row in bcfg["capacities"] if row["name"] == profile)
            if not source_is_current(source_path(b, "m1", capacity), bcfg, parent_provenance, "m1", capacity):
                raise RuntimeError("IO-only probe source fails Stage B source gate")
        if digest(config_path) != contract["config_sha256"] or digest(__file__) != contract["launcher_sha256"]:
            raise RuntimeError("IO-only probe contract changed during execution")
        matrix = adaptation_matrix(bcfg, a, b, "m1", profile)
        matrix.update(name=f"official_m1_m10_io_development_{profile}", support_trials=support_segments,
                      query_cutoff_trials=33, validation_split="interleaved")
        matrix["methods"] = {"none": [0.], "io": [locked_io_lr]}
        matrix["official_budget_probe"] = dict(cfg)
        path = out / "matrices" / f"{profile}.json"
        frozen_write(path, matrix)
        with path.with_suffix(".log").open("w") as stream:
            subprocess.run([sys.executable, str(ROOT / "scripts/run_cross_session_iteration.py"),
                            "--matrix", str(path), "--output-root", str(out / profile),
                            "--phase", "all"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
