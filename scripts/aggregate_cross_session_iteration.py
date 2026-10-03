"""Combine verified A/B results without using query scores for model selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", required=True, type=Path)
    parser.add_argument("--stage-b", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    roots = [("a_trial", args.stage_a.resolve(), analysis / "stage_a")]
    roots += [(profile, args.stage_b.resolve() / "adapt" / profile / task,
               analysis / "stage_b" / profile / task)
              for profile in ("old", "small", "wide") for task in ("m1", "m2")]
    rows, fit_count, eval_count = [], 0, 0
    identities, metrics = {}, {}
    for profile, root, summary in roots:
        verified = read_json(summary / "verification.json")
        if verified["status"] != "verified_complete" or verified["errors"]:
            raise ValueError(f"Cannot aggregate unverified results: {summary}")
        fit_count += verified["found_train_fits"]
        eval_count += verified["found_selected_evaluations"]
        grouped = read_csv(summary / "groupedsummary.csv")
        selected = read_csv(summary / "selected_metrics.csv")
        for row in selected:
            task, method, seed = row["task"], row["method"], int(row["seed"])
            identity = tuple(row[key] for key in
                             ("all_valid_index_hash", "all_valid_truth_hash",
                              "legacy_index_hash", "legacy_truth_hash"))
            if task in identities and identities[task] != identity:
                raise ValueError(f"Cross-profile cohort differs: {task}")
            identities[task] = identity
            metrics[(profile, task, method, seed)] = float(row["zero_all_valid_r2"])
        for group in grouped:
            task, method = group["task"], group["method"]
            entries = [row for row in selected if row["task"] == task and row["method"] == method]
            receipts = [read_json(Path(row["fit"]) / "fit_result.json") for row in entries]
            baseline = next(float(row["zero_all_valid_r2"]) for row in selected
                            if row["task"] == task and row["method"] == "none")
            result = dict(group, profile=profile,
                          selected_lr=float(entries[0]["lr"]),
                          direct_gain_over_same_profile_none=float(group["zero_all_valid_r2_mean"]) - baseline,
                          direct_seed_gains_over_none=json.dumps([float(row["zero_all_valid_r2"]) - baseline for row in entries]),
                          base_parameters=receipts[0]["base_parameters"],
                          initial_trainable_parameters=receipts[0]["initial_trainable_params"],
                          peak_trainable_parameters=max(item["peak_trainable_params"] for item in receipts),
                          best_steps=json.dumps([item["best_step"] for item in receipts]),
                          completed_steps=json.dumps([item["completed_steps"] for item in receipts]),
                          fit_seconds_mean=float(np.mean([item["fit_seconds"] for item in receipts])),
                          peak_cuda_allocated_mb_max=max(item["peak_cuda_allocated_mb"] for item in receipts))
            rows.append(result)
    for row in rows:
        if row["profile"] in ("small", "wide"):
            deltas = [value - metrics[("old", task, method, seed)]
                      for (profile, task, method, seed), value in metrics.items()
                      if (profile, task, method) == (row["profile"], row["task"], row["method"])]
            row["paired_direct_gain_over_old_recording_mean"] = float(np.mean(deltas))
            row["paired_direct_gain_over_old_recording_population_sd"] = float(np.std(deltas))
            row["paired_direct_seed_gains_over_old_recording"] = json.dumps(deltas)
    sources = []
    for task in ("m1", "m2"):
        for capacity in ("small", "wide"):
            manifest = read_json(args.stage_b / "source" / task / capacity / "manifest.json")
            fields = ("base_parameters", "session_frontend_parameters", "best_source_val_r2", "best_step",
                      "fit_seconds", "peak_cuda_allocated_mb")
            sources.append(dict(task=task, capacity=capacity,
                                **{field: manifest.get(field) for field in fields}))
    write_csv(analysis / "comparison.csv", rows)
    write_csv(analysis / "source_resources.csv", sources)
    (analysis / "combined_contract.json").write_text(json.dumps({
        "fit_entries": fit_count, "source_fits": len(sources),
        "total_run_entries": fit_count + len(sources), "selected_evaluations": eval_count,
        "cohort_identical_across_all_profiles": True, "query_used_for_selection": False,
        "a": "old source; trial-reset context; prefix-validation LR sweep",
        "b": "recording causal windows; A prefix-selected LR transfer",
        "direct": "float64 physical-unit variance-weighted R2; primary",
        "ridge": "support residual calibration; auxiliary",
        "spread": "population SD across adaptation seeds; none has one deterministic run",
        "source_uncertainty": "one source-training seed; not measured by adaptation-seed SD",
        "cost": "training allocated VRAM and wall time, including startup; no deployment latency claim"
    }, indent=2) + "\n")
    plot(rows, sources, analysis)


def plot(rows, sources, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ("none", "io", "lora", "affine", "affine_lora", "offset_rotated", "offset_original", "full")
    colors = {"a_trial": "#999999", "old": "#377eb8", "small": "#4daf4a", "wide": "#e41a1c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, task in zip(axes, ("m1", "m2")):
        for profile in colors:
            subset = [row for row in rows if row["task"] == task and row["profile"] == profile]
            x = [methods.index(row["method"]) for row in subset]
            y = [float(row["zero_all_valid_r2_mean"]) for row in subset]
            sd = [float(row["zero_all_valid_r2_population_sd"]) for row in subset]
            ax.errorbar(x, y, yerr=sd, fmt="o", capsize=3, color=colors[profile], label=profile)
        ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        ax.set_title(task.upper())
        ax.set_ylabel("Direct all-valid R2 (mean +/- adaptation seed SD)")
        ax.grid(axis="y", alpha=.25)
        ax.legend()
    fig.savefig(output / "adaptation_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, task in zip(axes, ("m1", "m2")):
        for profile in ("old", "small", "wide"):
            subset = [row for row in rows if row["task"] == task and row["profile"] == profile]
            baseline = next(row for row in subset if row["method"] == "none")
            best = max((row for row in subset if row["method"] != "none"),
                       key=lambda row: float(row["zero_all_valid_r2_mean"]))
            x = float(best["base_parameters"]) / 1e6
            y, y0 = float(best["zero_all_valid_r2_mean"]), float(baseline["zero_all_valid_r2_mean"])
            ax.plot([x, x], [y0, y], color=colors[profile], alpha=.4)
            ax.scatter([x], [y0], marker="x", color=colors[profile], s=50)
            ax.errorbar([x], [y], yerr=[float(best["zero_all_valid_r2_population_sd"])],
                        fmt="o", capsize=3, color=colors[profile], label=profile)
            ax.annotate(best["method"], (x, y), fontsize=8, xytext=(6, 6), textcoords="offset points")
        ax.set_title(task.upper())
        ax.set_xlim(.1, 3.65)
        ax.set_xlabel("Plain decoder parameters (millions)")
        ax.set_ylabel("Direct all-valid R2")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Local development: x = no tuning; circle = best adapted method", fontsize=10)
    fig.savefig(output / "capacity_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
