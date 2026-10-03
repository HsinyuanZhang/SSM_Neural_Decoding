"""Run an independent one-step Mamba-3 smoke test for all Stage A methods."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from ssm_decode import debug_experiment as debug
from ssm_decode.cross_session_iteration import support_partition
from ssm_decode.data import load_recording, sha256_file, source_target_plan
from ssm_decode.input_adaptation import configure_adaptation
from ssm_decode.mamba3_official import COMMIT, OFFICIAL, build_mamba3_official
from ssm_decode.session_normalization import fit_support_statistics


ROOT = Path(__file__).resolve().parents[4]
METHODS = (
    "none", "io", "lora", "affine", "affine_lora", "offset_rotated",
    "offset_original", "full",
)
LR = {"none": 0.0, "io": 3e-4, "lora": 3e-4, "affine": 3e-4,
      "affine_lora": 3e-4, "offset_rotated": 3e-4,
      "offset_original": 3e-4, "full": 1e-4}


def parameter_hashes(model):
    return {name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, parameter in model.named_parameters()}


def code_hashes():
    paths = [Path(__file__), ROOT / "ssm_decode/input_adaptation.py",
             ROOT / "ssm_decode/peft.py", ROOT / "ssm_decode/mamba3_official.py",
             ROOT / "ssm_decode/session_normalization.py",
             ROOT / "ssm_decode/debug_experiment.py"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def causal_difference(model, inputs):
    changed = inputs.clone()
    cutoff = inputs.shape[1] // 2
    perturbation = torch.linspace(1.0, 2.0, inputs.shape[-1], device=inputs.device)
    changed[:, cutoff:] = changed[:, cutoff:] + perturbation
    model.eval()
    with torch.inference_mode():
        first = model(inputs)[:, :cutoff]
        second = model(changed)[:, :cutoff]
    return float((first-second).abs().max()), bool(torch.equal(first, second))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_dir = ROOT / "results/debug_round2/latest/m1/cross_session_mamba3_official_w256_l4_n32_ctx128_seed0"
    checkpoint_path = source_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    with np.load(source_dir / "normalizer.npz") as saved:
        source_stats = tuple(saved[key].astype(np.float32) for key in
                             ("x_mean", "x_std", "y_mean", "y_std"))
    plan = source_target_plan("m1")
    target = load_recording("m1", "held_in", plan["cross_session_local_dev"]["target_session"])
    train_bounds, _, support_bounds = support_partition(target.trial_bounds, "interleaved", 33)
    stats = fit_support_statistics(target.neural, support_bounds, source_stats)
    windows = debug._windows([target], [train_bounds], stats, device)
    eligible = debug._eligible(windows)
    debug._seed(20261003)
    inputs, truth, valid = debug._sample_batch(windows, eligible, 128, 4)
    score_slice = slice(64, None)
    score_mask = valid[:, score_slice] & torch.isfinite(truth[:, score_slice]).all(-1)
    if not score_mask.any():
        raise AssertionError("the smoke batch has no valid scored bins")

    results = {}
    for method in METHODS:
        debug._seed(41)
        model = build_mamba3_official(
            target.neural.shape[1], target.behavior.shape[1], width=cfg["width"],
            layers=cfg["layers"], state_size=cfg["state_size"], dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            base_output = model(inputs)
        receipt = configure_adaptation(
            model, method=method, rank=4, alpha=4, lora_scope="all",
            train_input_bias=True,
        )
        model.eval()
        with torch.inference_mode():
            startup_output = model(inputs)
        startup_max = float((base_output-startup_output).abs().max())
        startup_exact = bool(torch.equal(base_output, startup_output))
        if not startup_exact:
            raise AssertionError(f"{method} changed the zero-step output by {startup_max}")
        pre_causal_max, pre_causal_exact = causal_difference(model, inputs)
        if not pre_causal_exact:
            raise AssertionError(f"{method} failed pre-step causality by {pre_causal_max}")

        before = parameter_hashes(model)
        trainable = {name: parameter for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        gradient_none = []
        gradient_nonfinite = []
        gradient_nonzero = []
        gradient_norm = None
        loss_value = None
        if trainable:
            optimizer = torch.optim.AdamW(trainable.values(), lr=LR[method], weight_decay=1e-4)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, score_slice]
            loss = (prediction[score_mask]-truth[:, score_slice][score_mask]).square().mean()
            loss.backward()
            loss_value = float(loss.detach())
            for name, parameter in trainable.items():
                if parameter.grad is None:
                    gradient_none.append(name)
                elif not torch.isfinite(parameter.grad).all():
                    gradient_nonfinite.append(name)
                elif torch.count_nonzero(parameter.grad).item():
                    gradient_nonzero.append(name)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                trainable.values(), 1.0, error_if_nonfinite=True,
            ))
            if gradient_none or gradient_nonfinite or not gradient_nonzero:
                raise AssertionError(
                    f"{method} invalid gradients: none={gradient_none[:2]}, "
                    f"nonfinite={gradient_nonfinite[:2]}, nonzero={len(gradient_nonzero)}"
                )
            optimizer.step()

        after = parameter_hashes(model)
        changed = sorted(name for name in before if before[name] != after[name])
        frozen_changed = sorted(name for name in changed if name not in trainable)
        if frozen_changed:
            raise AssertionError(f"{method} changed frozen parameters: {frozen_changed[:3]}")
        if trainable and not changed:
            raise AssertionError(f"{method} made no parameter update")
        post_causal_max, post_causal_exact = causal_difference(model, inputs)
        if not post_causal_exact:
            raise AssertionError(f"{method} failed post-step causality by {post_causal_max}")
        results[method] = {
            "receipt": receipt,
            "startup_exact": startup_exact,
            "startup_max_abs": startup_max,
            "pre_step_causality_exact": pre_causal_exact,
            "pre_step_causality_max_abs": pre_causal_max,
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "gradient_none": gradient_none,
            "gradient_nonfinite": gradient_nonfinite,
            "gradient_nonzero_count": len(gradient_nonzero),
            "changed_parameter_names": changed,
            "frozen_changed_parameter_names": frozen_changed,
            "post_step_causality_exact": post_causal_exact,
            "post_step_causality_max_abs": post_causal_max,
        }
        del model
        torch.cuda.empty_cache()

    actual_commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True,
    ).strip()
    report = {
        "schema": "independent_stage_a_gpu_smoke_v1",
        "status": "passed",
        "task": "m1",
        "formal_training": False,
        "steps_per_trainable_method": 1,
        "query_labels_used": False,
        "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "target_file": str(target.path),
        "target_file_sha256": sha256_file(target.path),
        "support_bounds": support_bounds,
        "batch_shape": list(inputs.shape),
        "valid_scored_bins": int(score_mask.sum()),
        "results": results,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": str(device), "cuda_name": torch.cuda.get_device_name(device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "official_expected_commit": COMMIT,
                    "official_actual_commit": actual_commit},
        "code_sha256": code_hashes(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "methods": list(results)}, indent=2))


if __name__ == "__main__":
    main()
