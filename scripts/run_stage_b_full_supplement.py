"""Run the missing Stage B full-tuning control with a prefix-selected LR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m1_official_budget_probe import digest, frozen_write, read
from scripts.run_session_pretraining_iteration import adaptation_matrix, verify_live, source_is_current, source_path
from scripts.run_cross_session_iteration import (_select, _evaluation_is_current,
    _prepare_root, _run_fit, _evaluate)


def verify_finished_parent(b):
    root = b / "adapt/wide/m2"
    if not (root / "selections.json").is_file():
        return False
    matrix, provenance = read(root / "matrix.json"), read(root / "provenance.json")
    selected = read(root / "selections.json")
    if selected != _select(matrix, root, provenance, persist=False):
        raise RuntimeError("Parent Stage B selection fails prefix reconstruction")
    fits = [fit for item in selected["tasks"]["m2"]["methods"].values()
            for fit in item["selected"]["fits"]]
    return len(fits) == 13 and all(_evaluation_is_current(Path(row["fit"]),
        Path(row["fit"]) / "query_evaluation", matrix, provenance, "m2", row) for row in fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    a, b, out = (p.resolve() for p in (args.stage_a_root, args.stage_b_root, args.output_root))
    for parent in (a, b):
        if out == parent or parent in out.parents or out in parent.parents:
            raise ValueError("Supplement output must be separate from formal A and B roots")
    path = ROOT / "configs/session_pretraining_iteration_b_full_supplement.json"
    cfg, bcfg = read(path), read(b / "matrix.json")
    # This gate is part of the frozen launcher. It releases GPU 1 only after
    # every parent M2 profile has finished, and authenticates the last profile.
    while not verify_finished_parent(b):
        time.sleep(5)
    parent_provenance = read(b / "provenance.json")
    verify_live(bcfg, parent_provenance, a)
    selections = read(a / "selections.json")
    if selections != _select(read(a / "matrix.json"), a, read(a / "provenance.json"), persist=False):
        raise RuntimeError("Stage A prefix selection does not authenticate")
    contract = dict(cfg,
        config_sha256=digest(path), launcher_sha256=digest(__file__),
        stage_a_provenance_sha256=digest(a / "provenance.json"),
        stage_b_provenance_sha256=digest(b / "provenance.json"))
    frozen_write(out / "supplement_contract.json", contract)
    for profile in cfg["profiles"]:
        verify_live(bcfg, parent_provenance, a)
        capacity = next(row for row in bcfg["capacities"] if row["name"] == profile)
        if not source_is_current(source_path(b, "m2", capacity), bcfg, parent_provenance, "m2", capacity):
            raise RuntimeError("Supplement source fails Stage B source gate")
        matrix = adaptation_matrix(bcfg, a, b, cfg["task"], profile)
        if matrix["full_initialization"] != "source":
            raise RuntimeError("This supplement requires source-initialized full tuning")
        matrix["name"] = f"stage_b_full_supplement_{profile}_m2"
        matrix["methods"] = {"full": [selections["tasks"]["m2"]["methods"]["full"]["selected"]["lr"]]}
        matrix["full_supplement"] = cfg
        target = out / "matrices" / f"{profile}.json"
        frozen_write(target, matrix)
        destination = out / profile
        frozen, provenance = _prepare_root(destination, target)
        # The main queue always trains IO warmstarts. Source-initialized full
        # tuning does not use them, so this closed queue runs only six full fits.
        warmstarts = {str(seed): {"candidates": [], "selected": None,
                      "selection_basis": "source initialization; no IO warmstart"} for seed in cfg["seeds"]}
        frozen_write(destination / "m2/ui_warmstarts.json", warmstarts)
        shutil.copyfile(__file__, destination / "code_snapshot/run_stage_b_full_supplement.py")
        for seed in cfg["seeds"]:
            if digest(path) != contract["config_sha256"] or digest(__file__) != contract["launcher_sha256"]:
                raise RuntimeError("Supplement contract changed during execution")
            verify_live(bcfg, parent_provenance, a)
            _run_fit(frozen, provenance, destination, "m2", "1", "full", seed, matrix["methods"]["full"][0])
        selected = _select(frozen, destination, provenance)
        _evaluate(frozen, destination, selected, provenance)


if __name__ == "__main__":
    main()
