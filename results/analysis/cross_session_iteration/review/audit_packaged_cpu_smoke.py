"""Build temporary task payloads and run the packaged CPU decoder contract."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from apst.data.load import list_sessions
from falcon_challenge.config import FalconConfig, FalconTask
from ssm_decode.falcon_decoder import SSMFalconDecoder


ROOT = Path(__file__).resolve().parents[4]
DATA = Path("/mnt/data/work_host/SPINT/SPINT-main/data")
OUTPUT = Path(__file__).with_name("packaged_cpu_smoke.json")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_task(task: str) -> dict:
    source_path = (ROOT / "results/cross_session_iteration_b/official_20261003" /
                   "source" / task / "small" / "best.pt")
    normalizer_path = source_path.parent / "normalizer.npz"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    config = FalconConfig(task=getattr(FalconTask, task))
    input_size, output_size, budget, cap = {
        "m1": (64, 16, 10, 4),
        "m2": (96, 2, 33, 7),
    }[task]
    public = []
    for split in ("held_in", "held_out"):
        public.extend(list_sessions(task, split, root=DATA))
    with tempfile.TemporaryDirectory(prefix=f"ssm_{task}_payload_smoke_") as directory:
        payload = Path(directory)
        package = payload / "ssm_decode"
        package.mkdir()
        (package / "__init__.py").write_text('"""Temporary smoke package."""\n')
        for name in ("mamba3_cpu.py", "falcon_decoder.py"):
            shutil.copyfile(ROOT / "ssm_decode" / name, package / name)
        shutil.copyfile(ROOT / "third_party/mamba/LICENSE", payload / "MAMBA_LICENSE")
        rows = []
        for item in public:
            tag = config.hash_dataset(Path(item["path"]).stem)
            bank = payload / "banks" / tag
            bank.mkdir(parents=True)
            torch.save({"state_dict": source["state_dict"]}, bank / "model.pt")
            shutil.copyfile(normalizer_path, bank / "normalizer.npz")
            rows.append({
                "tag": tag,
                "checkpoint_file": f"banks/{tag}/model.pt",
                "normalizer_file": f"banks/{tag}/normalizer.npz",
            })
        files = {str(path.relative_to(payload)): sha(path)
                 for path in payload.rglob("*") if path.is_file()}
        manifest = {
            "schema": "ssm_falcon_cpu_payload_v1",
            "task": task,
            "input_size": input_size,
            "output_size": output_size,
            "context": 128,
            "normalization": "fixed_calibration_zscore",
            "output_space": "official_physical",
            "output_postprocess": "none",
            "calibration_trials": budget,
            "models": rows,
            "files": files,
        }
        (payload / "payload_manifest.json").write_text(json.dumps(manifest) + "\n")
        decoder = SSMFalconDecoder(config, payload, batch_size=cap)
        decoder.reset([Path(item["path"]) for item in public[:cap]])
        zeros = np.zeros((cap, input_size), dtype=np.float32)
        ones = np.ones((cap, input_size), dtype=np.float32)
        began = time.perf_counter()
        first = decoder.predict(zeros)
        cold_ms = 1000 * (time.perf_counter() - began)
        warm = []
        for _ in range(3):
            began = time.perf_counter()
            prediction = decoder.predict(zeros)
            warm.append(1000 * (time.perf_counter() - began))
        before = decoder.history.copy()
        decoder.observe(ones)
        after = decoder.history.copy()
        final = decoder.predict(zeros)
        if (first.shape != (cap, output_size) or final.shape != first.shape or
                not np.isfinite(first).all() or not np.isfinite(final).all() or
                np.array_equal(before, after)):
            raise RuntimeError(f"{task} packaged CPU smoke failed")
        if any((payload / relative).is_file() is False for relative in files):
            raise RuntimeError(f"{task} payload closure changed during smoke")
        return {
            "task": task,
            "capacity": "small",
            "source_checkpoint_sha256": sha(source_path),
            "source_normalizer_sha256": sha(normalizer_path),
            "public_model_count": len(rows),
            "batch_size": cap,
            "payload_file_count": len(files),
            "cold_predict_ms": cold_ms,
            "warm_predict_ms_mean": float(np.mean(warm)),
            "warm_predict_ms_max": float(np.max(warm)),
            "first_output_max_abs": float(np.abs(first).max()),
            "observe_advanced_history": True,
            "finite_output": True,
            "output_shape": list(first.shape),
        }


def main() -> None:
    tasks = [run_task(task) for task in ("m1", "m2")]
    report = {
        "schema": "independent_packaged_cpu_smoke_v1",
        "status": "passed",
        "implementation_sha256": {
            "audit_packaged_cpu_smoke.py": sha(Path(__file__)),
            "falcon_decoder.py": sha(ROOT / "ssm_decode" / "falcon_decoder.py"),
            "mamba3_cpu.py": sha(ROOT / "ssm_decode" / "mamba3_cpu.py"),
        },
        "torch_threads": torch.get_num_threads(),
        "tasks": tasks,
        "interpretation": (
            "This smoke verifies payload closure, roster routing, task geometry, "
            "batch caps, finite CPU execution, and observe history. Timing is diagnostic."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
