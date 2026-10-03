"""Independently audit the fixed M2 unmerged-LoRA public bank payload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from apst.data.load import list_sessions
from falcon_challenge.config import FalconConfig, FalconTask

from ssm_decode.mamba3_cpu_lora import CPUDecoder
from ssm_decode.official_calibration import load_public_calibration
from ssm_decode.official_payload import KEYS, sha
from ssm_decode.session_normalization import fit_support_statistics


WRAPPED = (
    "in_proj",
    "out_proj",
    "blocks.0.ssm.in_proj",
    "blocks.0.ssm.out_proj",
    "blocks.1.ssm.in_proj",
    "blocks.1.ssm.out_proj",
    "blocks.2.ssm.in_proj",
    "blocks.2.ssm.out_proj",
    "blocks.3.ssm.in_proj",
    "blocks.3.ssm.out_proj",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_closure(root: Path, *, exclude_manifest: bool = False) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if exclude_manifest and rel == "payload_manifest.json":
            continue
        result[rel] = sha(path)
    return result


def state_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def check_unmerged_state(state: dict, source_state: dict) -> tuple[list[str], int]:
    errors: list[str] = []
    expected: set[str] = set()
    wrapped = set(WRAPPED)
    for key, source_value in source_state.items():
        path, leaf = key.rsplit(".", 1)
        if path in wrapped and leaf in {"weight", "bias"}:
            target = f"{path}.base.{leaf}"
            expected.add(target)
            if target not in state:
                errors.append(f"missing {target}")
            elif path == "in_proj" and leaf == "bias":
                if not torch.isfinite(state[target]).all():
                    errors.append("nonfinite trainable root input bias")
            elif not torch.equal(state[target], source_value):
                errors.append(f"frozen base differs: {target}")
        else:
            expected.add(key)
            if key not in state or not torch.equal(state[key], source_value):
                errors.append(f"frozen source tensor differs: {key}")
    for path in WRAPPED:
        base = state.get(path + ".base.weight")
        if not isinstance(base, torch.Tensor) or base.ndim != 2:
            errors.append(f"missing or invalid base weight: {path}")
            continue
        for suffix in ("A", "B", "rows"):
            expected.add(path + "." + suffix)
        a, b, rows = (state.get(path + "." + suffix) for suffix in ("A", "B", "rows"))
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(rows, torch.Tensor):
            errors.append(f"missing LoRA tensors: {path}")
            continue
        if tuple(a.shape) != (base.shape[0], 4) or tuple(b.shape) != (4, base.shape[1]):
            errors.append(f"LoRA rank geometry differs: {path}")
        if rows.dtype != torch.int64 or not torch.equal(rows, torch.arange(base.shape[0], dtype=torch.int64)):
            errors.append(f"LoRA rows differ: {path}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            errors.append(f"nonfinite LoRA tensor: {path}")
    if set(state) != expected:
        errors.append("unmerged state key closure differs")
    return errors, len(expected)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--failed-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    failed = args.failed_root.resolve()
    probe = args.probe_root.resolve()
    payload = root / "payload"
    manifest_path = payload / "payload_manifest.json"
    contract_path = root / "export_contract.json"
    manifest = read(manifest_path)
    contract = read(contract_path)
    errors: list[str] = []

    if manifest.get("schema") != "ssm_falcon_cpu_payload_v1" or manifest.get("task") != "m2":
        errors.append("payload schema or task differs")
    expected_header = {
        "input_size": 96,
        "output_size": 2,
        "context": 128,
        "calibration_trials": 33,
        "method": "lora_unmerged",
        "public_calibration_only": True,
        "query_labels_used": False,
        "test_time_parameter_updates": False,
    }
    for key, value in expected_header.items():
        if manifest.get(key) != value:
            errors.append(f"payload header differs: {key}")
    if manifest.get("export_contract") != contract:
        errors.append("payload and root export contracts differ")

    frozen_failed = contract.get("failed_root_file_sha256")
    current_failed = {
        str(path.relative_to(failed)): sha(path)
        for path in sorted(failed.rglob("*"))
        if path.is_file() and ".triton_cache" not in str(path) and "__pycache__" not in str(path)
    }
    if frozen_failed != current_failed:
        errors.append("immutable failed-root closure changed")

    implementation = contract.get("new_implementation_sha256", {})
    repo = Path(__file__).resolve().parents[4]
    live_runtime = repo / "ssm_decode/mamba3_cpu_lora.py"
    live_adapter = repo / "ssm_decode/falcon_decoder.py"
    live_license = repo / "third_party/mamba/LICENSE"
    expected_impl = {
        str(live_runtime): sha(live_runtime),
        str(live_adapter): sha(live_adapter),
        str(live_license): sha(live_license),
    }
    for path, digest in expected_impl.items():
        if implementation.get(path) != digest:
            errors.append(f"implementation binding differs: {path}")
    copied = {
        payload / "ssm_decode/mamba3_cpu.py": sha(live_runtime),
        payload / "ssm_decode/falcon_decoder.py": sha(live_adapter),
        payload / "MAMBA_LICENSE": sha(live_license),
    }
    for path, digest in copied.items():
        if not path.is_file() or sha(path) != digest:
            errors.append(f"copied runtime differs: {path.name}")

    actual_payload = file_closure(payload, exclude_manifest=True)
    if manifest.get("files") != actual_payload:
        errors.append("payload file closure or file digest differs")
    if len(actual_payload) != 30:
        errors.append("payload must contain exactly 30 files excluding its manifest")
    forbidden = ("receipt", "train_log", "acceptance", ".nwb", "raw")
    if any(any(token in rel.lower() for token in forbidden) for rel in actual_payload):
        errors.append("payload contains a diagnostic or raw-data file")

    original_identity = contract["original_export_contract"]
    source_path = Path(original_identity["args"]["source"])
    source_normalizer = source_path.parent / "normalizer.npz"
    if sha(source_path) != original_identity["source_checkpoint_sha256"]:
        errors.append("source checkpoint hash differs")
    if sha(source_normalizer) != original_identity["source_normalizer_sha256"]:
        errors.append("source normalizer hash differs")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = source["state_dict"]
    with np.load(source_normalizer, allow_pickle=False) as archive:
        source_stats = tuple(archive[key].astype(np.float32) for key in KEYS)

    task_config = FalconConfig(task=FalconTask.m2)
    data_root = Path(original_identity["args"]["data_root"])
    expected_sessions = []
    for split in ("held_in", "held_out"):
        for item in list_sessions("m2", split, root=data_root):
            expected_sessions.append((split, item["session"]))
    rows = manifest.get("models", [])
    row_by_tag = {row.get("tag"): row for row in rows if isinstance(row, dict)}
    if len(rows) != 13 or len(row_by_tag) != 13:
        errors.append("payload model roster is not exactly 13 unique rows")

    probe_state = torch.load(probe / "fits/lora_seed0/unmerged_best.pt", map_location="cpu", weights_only=False)["state_dict"]
    bank_reports = []
    target_state_exact = False
    max_abs_score_delta = 0.0
    fold_rejected = []
    for split, session in expected_sessions:
        record = load_public_calibration("m2", split, session, data_root)
        tag = task_config.hash_dataset(record.path.stem)
        row = row_by_tag.get(tag)
        bank = root / "banks" / tag
        if row is None:
            errors.append(f"missing manifest row: {tag}")
            continue
        checkpoint = bank / "unmerged_best.pt"
        normalizer = bank / "normalizer.npz"
        receipt_path = bank / "receipt.json"
        log_path = bank / "train_log.json"
        acceptance_path = bank / "unmerged_acceptance.json"
        required = (checkpoint, normalizer, receipt_path, log_path, acceptance_path)
        if any(not path.is_file() for path in required):
            errors.append(f"incomplete bank files: {tag}")
            continue
        receipt = read(receipt_path)
        acceptance = read(acceptance_path)
        public = receipt.get("public_calibration", {})
        if public != record.receipt:
            errors.append(f"public reader receipt differs: {tag}")
        if (
            public.get("raw_nwb_trial_ids_first_n") != list(range(33))
            or len(public.get("realtrialbounds", [])) != 33
            or public.get("calibration_trials") != 33
            or public.get("publiccalibration_only") is not True
            or public.get("query_labels_used") is not False
        ):
            errors.append(f"raw M33 budget differs: {tag}")
        if sha(Path(record.path)) != public.get("rawfile_sha256"):
            errors.append(f"public raw hash differs: {tag}")
        if receipt.get("query_labels_used") is not False or receipt.get("query_used_for_selection") is not False:
            errors.append(f"query isolation differs: {tag}")
        if receipt.get("convergence_satisfied") is not True or receipt.get("frozen_audit", {}).get("status") != "passed":
            errors.append(f"convergence or frozen audit failed: {tag}")
        scope = receipt.get("adaptation", {})
        expected_scope = {
            "method": "lora",
            "rank": 4,
            "alpha": 4,
            "lora_scope": "all",
            "train_input_bias": True,
            "trainable_count": 37000,
        }
        if any(scope.get(key) != value for key, value in expected_scope.items()):
            errors.append(f"LoRA scope differs: {tag}")
        if sha(log_path) != receipt.get("train_log_sha256") or sha(normalizer) != receipt.get("normalizer_sha256"):
            errors.append(f"bank receipt hashes differ: {tag}")
        stats = fit_support_statistics(record.neural, [(0, len(record.neural))], source_stats)
        with np.load(normalizer, allow_pickle=False) as archive:
            saved_stats = tuple(archive[key].astype(np.float32) for key in KEYS)
        if not all(np.array_equal(left, right) for left, right in zip(stats, saved_stats)):
            errors.append(f"normalizer differs from raw M33 recomputation: {tag}")

        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_args = {"method": "lora", "seed": 0, "lr": 0.003, "steps": 2000, "max_steps": 8000}
        if any(obj.get("args", {}).get(key) != value for key, value in expected_args.items()):
            errors.append(f"checkpoint fit arguments differ: {tag}")
        if obj.get("best_step") != receipt.get("best_step"):
            errors.append(f"checkpoint best step differs: {tag}")
        state_errors, state_keys = check_unmerged_state(obj.get("state_dict", {}), source_state)
        errors.extend(f"{tag}: {message}" for message in state_errors)
        try:
            decoder = CPUDecoder.from_state_dict(obj["state_dict"])
            prediction = decoder.forward(torch.zeros(1, 1, 96))
            if tuple(prediction.shape) != (1, 1, 2) or not torch.isfinite(prediction).all():
                errors.append(f"CPU smoke failed: {tag}")
        except Exception as exc:
            errors.append(f"CPU load failed: {tag}: {type(exc).__name__}")

        if acceptance.get("status") != "PASS" or acceptance.get("scope") != "all_public_prefix_validation_points":
            errors.append(f"unmerged acceptance status differs: {tag}")
        score_delta = acceptance.get("score_delta")
        if not isinstance(score_delta, (int, float)) or isinstance(score_delta, bool) or not np.isfinite(score_delta) or abs(score_delta) > 1e-3:
            errors.append(f"prefix CPU/GPU score gate failed: {tag}")
        else:
            max_abs_score_delta = max(max_abs_score_delta, abs(float(score_delta)))
        legacy = acceptance.get("legacy_merge_status")
        if legacy == "converged_original_pointwise_fold_rejected":
            fold_rejected.append(tag)
            if receipt.get("fold_allclose") is not False:
                errors.append(f"fold-rejected status differs: {tag}")
        elif legacy == "original_merged_export_passed":
            if receipt.get("fold_allclose") is not True or receipt.get("fold_r2_abs_delta", 2) > 1e-3:
                errors.append(f"legacy merged pass status differs: {tag}")
        else:
            errors.append(f"unknown legacy merge status: {tag}")
        if row.get("calibration_receipt_sha256") != sha(acceptance_path):
            errors.append(f"acceptance hash differs: {tag}")
        for key, source_file in (("checkpoint_file", checkpoint), ("normalizer_file", normalizer)):
            rel = row.get(key)
            expected_rel = f"banks/{tag}/{source_file.name}"
            if rel != expected_rel or sha(payload / expected_rel) != sha(source_file):
                errors.append(f"payload bank copy differs: {tag} {key}")

        original = failed / "banks" / tag
        reused_exact = None
        if (original / "unmerged_best.pt").is_file():
            reused_exact = all(
                sha(bank / name) == sha(original / name)
                for name in ("unmerged_best.pt", "normalizer.npz", "receipt.json", "train_log.json")
            )
            if not reused_exact:
                errors.append(f"reused bank differs from failed-root bytes: {tag}")
        if tag == "Run1_20201028":
            target_state_exact = state_equal(obj["state_dict"], probe_state)
            if not target_state_exact:
                errors.append("target public bank differs tensor-wise from legal lora seed0")

        bank_reports.append(
            {
                "tag": tag,
                "split": split,
                "session": session,
                "checkpoint_sha256": sha(checkpoint),
                "normalizer_sha256": sha(normalizer),
                "best_step": receipt["best_step"],
                "final_budget": receipt["final_budget"],
                "legacy_merge_status": legacy,
                "prefix_validation_points": acceptance.get("points"),
                "gpu_r2": acceptance.get("gpu_r2"),
                "cpu_r2": acceptance.get("cpu_r2"),
                "score_delta": score_delta,
                "unmerged_state_key_count": state_keys,
                "reused_failed_root_bytes_exact": reused_exact,
                "raw_trial_ids_sha256": array_sha(np.arange(33, dtype=np.int64)),
            }
        )

    expected_tags = [
        task_config.hash_dataset(load_public_calibration("m2", split, session, data_root).path.stem)
        for split, session in expected_sessions
    ]
    if [row.get("tag") for row in rows] != expected_tags:
        errors.append("manifest roster order differs from official loader order")
    if sorted(fold_rejected) != ["Run1_20201030", "Run2_20201124"]:
        errors.append("unexpected set of legacy pointwise-fold rejections")

    report = {
        "schema": "m2_unmerged_public_bank_independent_audit_v1",
        "status": "GO" if not errors else "NO_GO",
        "errors": errors,
        "root": str(root),
        "root_contract_sha256": sha(contract_path),
        "payload_manifest_sha256": sha(manifest_path),
        "payload_file_count_excluding_manifest": len(actual_payload),
        "roster_count": len(rows),
        "roster": [row.get("tag") for row in rows],
        "fold_rejected_unmerged_only": sorted(fold_rejected),
        "max_absolute_prefix_cpu_gpu_score_delta": max_abs_score_delta,
        "target_bank_exact_to_legal_probe_state": target_state_exact,
        "source_checkpoint_sha256": sha(source_path),
        "runtime_sha256": sha(live_runtime),
        "banks": bank_reports,
        "checks": {
            "failed_root_closure_exact": frozen_failed == current_failed,
            "payload_closure_exact": manifest.get("files") == actual_payload,
            "all_raw_m33_recomputed": len(bank_reports) == 13,
            "all_unmerged_states_source_scoped": not any("source tensor" in error or "base differs" in error or "state key" in error for error in errors),
            "all_cpu_smokes_finite": not any("CPU" in error for error in errors),
            "all_prefix_score_deltas_within_1e-3": max_abs_score_delta <= 1e-3 and len(bank_reports) == 13,
            "target_bank_exact_to_legal_probe_state": target_state_exact,
        },
        "script_sha256": sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "errors": len(errors), "banks": len(bank_reports), "max_delta": max_abs_score_delta}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
