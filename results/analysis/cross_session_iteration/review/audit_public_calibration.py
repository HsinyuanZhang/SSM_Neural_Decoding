"""Independently verify the public M1/M2 calibration loader contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from apst.data.load import list_sessions, load_nwb_file
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import DATASET_HELDINOUT_MAP
from ssm_decode.official_calibration import BUDGET, load_public_calibration
from ssm_decode.official_payload import partitions


ROOT = Path("/mnt/data/work_host/SPINT/SPINT-main/data")
OUTPUT = Path(__file__).with_name("public_calibration_loader_audit.json")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    body = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    rows = []
    failures = []
    expected_counts = {"m1": 7, "m2": 13}
    for task in ("m1", "m2"):
        config = FalconConfig(task=getattr(FalconTask, task))
        expected_tags = set(DATASET_HELDINOUT_MAP[task]["held_in"] +
                            DATASET_HELDINOUT_MAP[task]["held_out"])
        observed_tags = set()
        for split in ("held_in", "held_out"):
            for item in list_sessions(task, split, root=ROOT):
                path = Path(item["path"]).resolve()
                neural, behavior, change, eval_mask = load_nwb_file(path, task)
                neural = np.asarray(neural, dtype=np.float32)
                behavior = np.asarray(behavior, dtype=np.float32)
                change = np.asarray(change, dtype=bool).reshape(-1)
                eval_mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
                starts = np.flatnonzero(change)
                budget = BUDGET[task]
                prefix_end = int(starts[budget]) if len(starts) > budget else len(neural)
                expected_bounds = tuple(
                    (int(starts[index]),
                     int(starts[index + 1]) if index + 1 < budget else prefix_end)
                    for index in range(budget)
                )
                with h5py.File(path, "r") as nwb:
                    trials = nwb["intervals"]["trials"]
                    trial_ids = np.asarray(trials["id"])
                record = load_public_calibration(task, split, item["session"], ROOT)
                tag = config.hash_dataset(path.stem)
                observed_tags.add(tag)
                train_bounds, val_bounds = partitions(record)
                train_label_mask = np.zeros(prefix_end, dtype=bool)
                val_label_mask = np.zeros(prefix_end, dtype=bool)
                for start, end in train_bounds:
                    train_label_mask[start:end] = record.eval_mask[start:end]
                for start, end in val_bounds:
                    val_label_mask[start:end] = record.eval_mask[start:end]
                segments = list(record.trial_bounds)
                if segments[0][0] > 0:
                    segments.insert(0, (0, segments[0][0]))
                checks = {
                    "budget_bound_count": len(record.trial_bounds) == budget,
                    "first_raw_trials_exact": record.trial_bounds == expected_bounds,
                    "prefix_crop_exact": len(record.neural) == prefix_end,
                    "neural_bytes_exact": sha_array(record.neural) == sha_array(neural[:prefix_end]),
                    "label_bytes_exact": sha_array(record.behavior) == sha_array(behavior[:prefix_end]),
                    "mask_exact": np.array_equal(
                        record.eval_mask,
                        eval_mask[:prefix_end] & (np.arange(prefix_end) >= starts[0]),
                    ),
                    "trial_table_count_matches_marks": len(trial_ids) == len(starts),
                    "first_n_trial_ids_exact": record.receipt["raw_nwb_trial_ids_first_n"] ==
                    [int(value) for value in trial_ids[:budget]],
                    "no_query_claim": record.receipt["query_labels_used"] is False,
                    "public_only_claim": record.receipt["publiccalibration_only"] is True,
                    "evaluator_tag_known": tag in expected_tags,
                    "train_val_bounds_disjoint": not set(train_bounds) & set(val_bounds),
                    "train_val_bounds_cover_public_prefix":
                    sorted(train_bounds + val_bounds) == segments,
                    "train_val_labels_disjoint": not np.any(train_label_mask & val_label_mask),
                    "train_val_labels_cover_public_mask":
                    np.array_equal(train_label_mask | val_label_mask, record.eval_mask),
                    "validation_segment_count":
                    len(val_bounds) == math.ceil(len(segments) / 5),
                    "nonempty_train_and_validation_labels":
                    bool(train_label_mask.any() and val_label_mask.any()),
                }
                if not all(checks.values()):
                    failures.append({"task": task, "split": split,
                                     "session": item["session"], "checks": checks})
                lengths = [end - start for start, end in record.trial_bounds]
                rows.append({
                    "task": task,
                    "split": split,
                    "session": item["session"],
                    "evaluator_tag": tag,
                    "path": str(path),
                    "raw_trial_count": int(len(starts)),
                    "selected_trial_count": budget,
                    "leading_pretrial_bins": int(starts[0]),
                    "selected_prefix_end": prefix_end,
                    "selected_shorter_than_50_count": int(sum(length < 50 for length in lengths)),
                    "selected_min_trial_length": int(min(lengths)),
                    "selected_max_trial_length": int(max(lengths)),
                    "train_segment_count": len(train_bounds),
                    "validation_segment_count": len(val_bounds),
                    "train_label_count": int(train_label_mask.sum()),
                    "validation_label_count": int(val_label_mask.sum()),
                    "first_n_trial_ids": [int(value) for value in trial_ids[:budget]],
                    "checks": checks,
                })
        if observed_tags != expected_tags:
            failures.append({
                "task": task,
                "reason": "public calibration/evaluator roster mismatch",
                "missing": sorted(expected_tags - observed_tags),
                "extra": sorted(observed_tags - expected_tags),
            })
        if sum(row["task"] == task for row in rows) != expected_counts[task]:
            failures.append({"task": task, "reason": "unexpected public file count"})
    report = {
        "schema": "independent_public_calibration_loader_audit_v1",
        "status": "passed" if not failures else "failed",
        "data_root": str(ROOT.resolve()),
        "implementation_sha256": {
            "audit_public_calibration.py": sha_file(Path(__file__)),
            "official_calibration.py": sha_file(
                Path(__file__).parents[4] / "ssm_decode" / "official_calibration.py"
            ),
            "official_payload.py": sha_file(
                Path(__file__).parents[4] / "ssm_decode" / "official_payload.py"
            ),
        },
        "task_file_counts": {task: sum(row["task"] == task for row in rows)
                             for task in ("m1", "m2")},
        "total_file_count": len(rows),
        "selected_shorter_than_50_total": sum(
            row["selected_shorter_than_50_count"] for row in rows
        ),
        "zero_start_file_count": sum(row["leading_pretrial_bins"] == 0 for row in rows),
        "nonzero_leading_file_count": sum(row["leading_pretrial_bins"] > 0 for row in rows),
        "failures": failures,
        "files": rows,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    print(json.dumps({key: report[key] for key in (
        "status", "task_file_counts", "total_file_count",
        "selected_shorter_than_50_total", "zero_start_file_count",
        "nonzero_leading_file_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
