"""Strict, local-only aggregation for completed pilot runs (not an official scorer)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_complete(run: Path) -> tuple[dict, list[dict], list[dict]]:
    if (run / "metrics.json").is_file():
        manifest = json.loads((run / "manifest.json").read_text())
        status = json.loads((run / "run.json").read_text())
        if status.get("status") != "completed": raise ValueError(f"{run}: profile run.json is not completed")
        args, plan = manifest.get("args", {}), manifest.get("plan", {})
        expected = {(kind, int(seed)) for kind in args.get("kinds", []) for seed in args.get("seeds", [])}
        completed = {(r["kind"], int(r["seed"])) for r in _jsonl(run / "progress.jsonl") if r.get("status") == "completed"}
        if expected != completed: raise ValueError(f"{run}: profile completed checkpoints do not match expected")
        absent = [f"{kind}_s{seed}.pt" for kind, seed in expected if not (run / f"{kind}_s{seed}.pt").is_file()]
        if absent: raise ValueError(f"{run}: missing full profile checkpoint(s): {', '.join(absent)}")
        normalized = bool(manifest.get("support_normalize_neural", False))
        target = plan.get("cross_session_local_dev", {}).get("target_session")
        if not target or target in set(plan.get("source_held_in_sessions", [])):
            raise ValueError(f"{run}: profile target absent or leaks into source")
        rows = json.loads((run / "metrics.json").read_text())
        if not rows or not {r.get("condition") for r in rows}.issuperset({"profile", "profile_drop20"}):
            raise ValueError(f"{run}: profile metrics lack profile/profile_drop20")
        for row in rows: row["run_family"] = "profile_normalized" if normalized else "profile"
        return manifest, rows, []
    status = json.loads((run / "run.json").read_text())
    if status.get("status") != "completed":
        raise ValueError(f"{run}: run.json is not completed")
    manifest = json.loads((run / "manifest.json").read_text())
    args = manifest.get("args", {})
    expected = {(kind, int(seed)) for kind in args.get("kinds", []) for seed in args.get("seeds", [])}
    completed = {(r["kind"], int(r["seed"])) for r in _jsonl(run / "models.jsonl") if r.get("status") == "completed"}
    if expected != completed:
        raise ValueError(f"{run}: completed checkpoints {sorted(completed)} do not match expected {sorted(expected)}")
    absent = [f"{kind}_s{seed}.pt" for kind, seed in expected if not (run / f"{kind}_s{seed}.pt").is_file()]
    if absent:
        raise ValueError(f"{run}: missing full checkpoint(s): {', '.join(absent)}")
    rows = _jsonl(run / "per_session_r2.jsonl")
    if not rows:
        raise ValueError(f"{run}: completed run has no per_session_r2.jsonl rows")
    target = manifest.get("cross_session_local_dev", {}).get("target_session")
    if not target or target in set(manifest.get("source_held_in_sessions", [])):
        raise ValueError(f"{run}: local held-in target is absent or leaks into source_held_in_sessions")
    skips = _jsonl(run / "skips.jsonl") if (run / "skips.jsonl").exists() else []
    minivals = set(manifest.get("minival_targets", []))
    if minivals:
        if any(row.get("surface") == "minival" and row.get("session") in minivals for row in rows):
            raise ValueError(f"{run}: minival produced scored rows; this aggregation expects skip-only minival")
        if not minivals.issubset({row.get("session") for row in skips if row.get("status") == "skipped"}):
            raise ValueError(f"{run}: missing skip receipt for at least one minival target")
    return manifest, rows, skips


def _task_name(run: Path) -> str:
    return run.name.upper()


def _summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["task"], row.get("run_family", "raw"), row.get("condition", "raw"), row["kind"], row["adaptation"], int(row["support_trials"]), row["session"], row["surface"])
        groups[key].append(float(row["r2_variance_weighted"]))
    output = []
    for key, values in sorted(groups.items()):
        task, family, condition, kind, adaptation, budget, session, surface = key
        output.append({"task": task, "run_family": family, "condition": condition, "kind": kind, "adaptation": adaptation, "support_trials": budget,
                       "session": session, "surface": surface, "n_seeds": len(values),
                       "r2_variance_weighted_mean": mean(values),
                       "r2_variance_weighted_sample_sd": stdev(values) if len(values) > 1 else math.nan,
                       "seed_values": values})
    return output


def _save_summary(path: Path, values: list[dict]) -> None:
    fields = ["task", "run_family", "condition", "kind", "adaptation", "support_trials", "session", "surface", "n_seeds",
              "r2_variance_weighted_mean", "r2_variance_weighted_sample_sd", "seed_values"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)


def _plot_r2(summary: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    tasks = sorted({x["task"] for x in summary})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        data = [x for x in summary if x["task"] == task]
        labels = sorted({f'{x["run_family"]}:{x["condition"]}/{x["kind"]}/{x["adaptation"]}/k{x["support_trials"]}' for x in data})
        for index, label in enumerate(labels):
            row = next(x for x in data if f'{x["run_family"]}:{x["condition"]}/{x["kind"]}/{x["adaptation"]}/k{x["support_trials"]}' == label)
            ax.scatter([index] * len(row["seed_values"]), row["seed_values"], color="tab:blue", alpha=.7, zorder=2)
            ax.errorbar(index, row["r2_variance_weighted_mean"], yerr=row["r2_variance_weighted_sample_sd"], color="black", capsize=3, zorder=3)
        ax.axhline(0, color="grey", linewidth=.8); ax.set_title(task); ax.set_ylabel("Physical variance-weighted R2")
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(output.with_suffix(".png"), dpi=180); fig.savefig(output.with_suffix(".pdf")); plt.close(fig)


def _plot_budget(summary: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for task in sorted({x["task"] for x in summary}):
        for kind in sorted({x["kind"] for x in summary if x["task"] == task}):
            rows = [x for x in summary if x["task"] == task and x["kind"] == kind and x["adaptation"] == "rls" and x["run_family"] == "raw"]
            if not rows: continue
            # A run's protocol holds the query tail fixed across budgets; label the actual session.
            rows.sort(key=lambda x: x["support_trials"])
            ax.errorbar([x["support_trials"] for x in rows], [x["r2_variance_weighted_mean"] for x in rows],
                        yerr=[x["r2_variance_weighted_sample_sd"] for x in rows], marker="o", capsize=3,
                        label=f"{task} {kind} ({rows[0]['session']})")
    ax.set_xlabel("Support trials (same query tail within session)"); ax.set_ylabel("Physical variance-weighted R2")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output.with_suffix(".png"), dpi=180); fig.savefig(output.with_suffix(".pdf")); plt.close(fig)


def _plot_normalized_profile_ridge(summary: list[dict], output: Path) -> bool:
    import matplotlib.pyplot as plt
    rows = [x for x in summary if x["run_family"] == "profile_normalized" and x["condition"] == "profile" and x["adaptation"] == "ridge"]
    if not rows: return False
    tasks = sorted({x["task"] for x in rows}); fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks),4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        values=[x for x in rows if x["task"] == task]
        ax.errorbar(range(len(values)), [x["r2_variance_weighted_mean"] for x in values],
                    yerr=[x["r2_variance_weighted_sample_sd"] for x in values], fmt="o", capsize=3)
        ax.axhline(0,color="grey",linewidth=.8); ax.set_title(f"{task}: normalized profile ridge")
        ax.set_ylabel("Physical variance-weighted R2"); ax.set_xticks(range(len(values)), [x["kind"] for x in values])
    fig.tight_layout(); fig.savefig(output.with_suffix(".png"),dpi=180); fig.savefig(output.with_suffix(".pdf")); plt.close(fig); return True


def _resource_index(models: list[dict]) -> dict[tuple[str, str], dict]:
    """Index resources by the checkpoint's task directory and kind; reject inconsistent seeds."""
    expected_dims = {"M1": (64, 16), "M2": (96, 2)}
    index: dict[tuple[str, str], dict] = {}
    for model in models:
        checkpoint = Path(model["checkpoint"])
        task, kind = checkpoint.parent.name.upper(), model["kind"]
        if task not in expected_dims:
            continue
        config = model.get("config", {})
        if (config.get("input_size"), config.get("output_size")) != expected_dims[task]:
            raise ValueError(f"{checkpoint}: config dimensions do not match {task}")
        resource = model["resources"]
        signature = (resource["macs_per_step"]["total"], resource["state_bytes"])
        key = (task, kind)
        if key in index:
            prior = index[key]
            if (prior["macs_per_step"]["total"], prior["state_bytes"]) != signature:
                raise ValueError(f"inconsistent resources across seeds for {task}/{kind}")
        else:
            index[key] = resource
    return index


def _optional_plots(results_root: Path, output: Path, summary: list[dict]) -> list[str]:
    """Read the explicit characterize/robustness schemas; never synthesize values."""
    import matplotlib.pyplot as plt
    made: list[str] = []
    payloads = []
    for path in results_root.rglob("*.json"):
        if "analysis" in path.parts: continue
        try: payloads.append((path, json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError): pass
    models = [m for _, x in payloads if isinstance(x, dict) and isinstance(x.get("models"), list) for m in x["models"] if "resources" in m]
    if models:
        resources = _resource_index(models)
        raw = [x for x in summary if x["run_family"] == "raw" and x["adaptation"] == "ridge"]
        max_budget = max(x["support_trials"] for x in raw)
        joint = [dict(task=x["task"], kind=x["kind"], r2_mean=x["r2_variance_weighted_mean"],
                      macs_total=resources[(x["task"], x["kind"])]["macs_per_step"]["total"], state_bytes=resources[(x["task"], x["kind"])]["state_bytes"])
                 for x in raw if x["support_trials"] == max_budget and (x["task"], x["kind"]) in resources]
        if joint:
            with (output / "resource_accuracy.csv").open("w", newline="") as h:
                writer=csv.DictWriter(h,fieldnames=list(joint[0])); writer.writeheader(); writer.writerows(joint)
            tasks=sorted({x["task"] for x in joint}); fig, axes=plt.subplots(1,len(tasks),figsize=(5*len(tasks),4),squeeze=False)
            for ax,task in zip(axes[0],tasks):
                vals=[x for x in joint if x["task"]==task]; ax.scatter([x["macs_total"] for x in vals],[x["r2_mean"] for x in vals])
                for x in vals: ax.annotate(x["kind"],(x["macs_total"],x["r2_mean"]),fontsize=8)
                ax.axhline(0,color="grey",linewidth=.8); ax.set_title(task); ax.set_xlabel("Estimated MACs per step"); ax.set_ylabel("Physical variance-weighted R2")
            fig.tight_layout(); base=output/"resource_accuracy_pareto"; fig.savefig(base.with_suffix(".png"),dpi=180); fig.savefig(base.with_suffix(".pdf")); plt.close(fig); made.append(base.name)
    robust = [r for _, x in payloads if isinstance(x, dict) and isinstance(x.get("rows"), list) for r in x["rows"] if r.get("condition") == "w8" and "ridge" in r]
    if robust:
        fig, ax = plt.subplots(figsize=(6,4)); labels=[f"{x.get('task','?')}/{x.get('kind','?')}/{x.get('state','?')}" for x in robust]
        ax.bar(range(len(robust)),[x["ridge"]["variance_weighted_r2"] for x in robust]); ax.set_ylabel("W8 ridge physical variance-weighted R2"); ax.set_xticks(range(len(robust)),labels,rotation=60,ha="right",fontsize=7); fig.tight_layout()
        base=output/"quantized_r2"; fig.savefig(base.with_suffix(".png"),dpi=180); fig.savefig(base.with_suffix(".pdf")); plt.close(fig); made.append(base.name)
    return made


def _baseline_table(results_root: Path, output: Path) -> list[dict]:
    path = results_root / "round1" / "readout_baselines.json"
    if not path.is_file(): return []
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if not isinstance(rows, list): return []
    fields = ["task", "family", "kind", "seed", "support_trials", "lags", "feature_dim", "r2_variance_weighted", "readout_bytes_float64", "readout_macs_per_bin"]
    clean = [{field: row.get(field) for field in fields} for row in rows]
    with (output / "readout_baselines.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(clean)
    return clean


def build(runs: list[Path], profile_runs: list[Path], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    all_rows, skipped = [], []
    for run in runs + profile_runs:
        manifest, rows, skip_rows = _assert_complete(run)
        task = _task_name(run)
        for row in rows:
            row = dict(row); row["task"] = task; row["run_dir"] = str(run); row.setdefault("run_family", "raw"); row.setdefault("condition", "raw"); all_rows.append(row)
        skipped.extend([dict(x, task=task, run_dir=str(run)) for x in skip_rows])
    summary = _summary(all_rows); _save_summary(output / "r2_summary.csv", summary)
    (output / "completed_rows.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (output / "skip_receipts.json").write_text(json.dumps(skipped, indent=2) + "\n")
    raw_max = max(x["support_trials"] for x in summary if x["run_family"] == "raw")
    seed_plot = [x for x in summary if x["run_family"] == "raw" and x["support_trials"] == raw_max and x["adaptation"] in {"zero", "ridge", "delta"}]
    _plot_r2(seed_plot, output / "r2_seed_points_and_means"); _plot_budget(summary, output / "calibration_budget_curves")
    normalized_figure = _plot_normalized_profile_ridge(summary, output / "profile_normalized_ridge")
    optional = _optional_plots(Path("results"), output, summary)
    baselines = _baseline_table(Path("results"), output)
    lines = ["# Completed pilot aggregation", "", "This is a local pilot report, not a POSSM replication, official evaluation, or superiority claim.", "",
             "Only directories with `run.json: completed`, every expected full checkpoint, and their schema-specific metrics (`per_session_r2.jsonl` for raw; `metrics.json` for profile) were included.", "",
             "## Mean ± sample SD of physical variance-weighted R2", "", "| Task | Family | Condition | Kind | Adaptation | Support trials | Session | n | Mean ± SD |", "|---|---|---|---|---|---:|---|---:|---:|"]
    for x in summary:
        lines.append(f"| {x['task']} | {x['run_family']} | {x['condition']} | {x['kind']} | {x['adaptation']} | {x['support_trials']} | {x['session']} | {x['n_seeds']} | {x['r2_variance_weighted_mean']:.4f} ± {x['r2_variance_weighted_sample_sd']:.4f} |")
    lines += ["", "M1 is reported as 16-dimensional EMG and M2 as 2-dimensional finger velocity when their manifests identify those tasks. The held-in latest target is excluded from source normalization by the run protocol; minival rows with fewer than three usable trials are represented only by skip receipts and are not scored.", "",
              "Figures: `r2_seed_points_and_means.{png,pdf}`, `calibration_budget_curves.{png,pdf}`" + (", `profile_normalized_ridge.{png,pdf}`" if normalized_figure else "") + (", plus " + ", ".join(optional) if optional else ".") + (" Readout baselines: `readout_baselines.csv`." if baselines else "")]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--profile-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.runs, args.profile_runs, args.output_dir)


if __name__ == "__main__": main()
