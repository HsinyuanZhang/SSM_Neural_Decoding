"""Reproducible aggregation for the SSM-only round-2 debugging protocol."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import matplotlib.pyplot as plt


def _rows(root: Path):
    audit_path = root / "debug_audit" / "debug_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for value in audit.get("sampling_audit", {}).get("rows", []):
            task = value.get("task", "")
            for model, scores in value.get("models", {}).items():
                for scope, key in (("legacy_tail", "old_tail_r2"), ("all_valid_trial", "all_valid_same_query_trials_r2")):
                    if key in scores:
                        yield {"family":"sampling_audit","task":task,"model":model,"adaptation":"ridge",
                               "score_scope":scope,"r2_variance_weighted":scores[key],"source_artifact":str(audit_path),
                               "protocol":"same model, separately scored old tail and all-valid query trials"}
    # Modern runner metrics are nested so both legacy tail and all-valid-trial
    # values survive aggregation instead of being flattened into an old CSV.
    for metrics_path in root.rglob("metrics.json"):
        relative = metrics_path.relative_to(root)
        if relative.parts[0] != "debug_round2":
            continue
        if any("smoke" in part or "interrupted" in part or "failed" in part for part in relative.parts):
            continue
        try:
            payload = json.loads(metrics_path.read_text())
            if payload.get("status") != "completed":
                continue
            if "records" in payload and "final" not in payload:  # linear_baselines schema
                for record in payload["records"]:
                    row={"family":"linear_baseline","task":payload.get("task",""),"model":"linear_"+record["feature"],
                         "adaptation":"ridge" if record["feature"] not in {"zero","support_mean"} else record["feature"],
                         "score_scope":record["scope"],"source_artifact":str(metrics_path),"run_dir":str(metrics_path.parent),
                         "seed":"deterministic","protocol":payload.get("protocol",""),**record}
                    yield row
                continue
            final = payload.get("final", {})
            args = payload.get("args", {})
            for adaptation in ("zero", "ridge"):
                value = final.get(adaptation)
                if not isinstance(value, dict):
                    continue
                base = {"family": "modern_debug", "adaptation": adaptation,
                        "source_artifact": str(metrics_path), "run_dir": str(metrics_path.parent),
                        "protocol": payload.get("protocol", ""), "params": payload.get("params", ""),
                        "best_source_val_r2": payload.get("best_source_val_r2", ""),
                        **args}
                base["context_policy"] = "recording_causal_fixed_window" if "session_context" in relative.parts else "trial_causal_fixed_window"
                base["model"] = args.get("kind", "")
                base["formal"] = True
                base["depth"] = 1 if args.get("kind") in {"osc", "gru", "diag", "bank", "selective"} else args.get("layers", "")
                for scope, metric in (("legacy_tail", value.get("legacy")), ("all_valid_trial", value.get("all_valid"))):
                    if isinstance(metric, dict):
                        row = dict(base); row.update(metric); row["score_scope"] = scope; yield row
                if not isinstance(value.get("legacy"), dict) and "r2_variance_weighted" in value:
                    row=dict(base); row.update(value); row["score_scope"]="legacy_tail"; yield row
        except (OSError, json.JSONDecodeError):
            continue
    for csv_path in root.rglob("per_session_r2.csv"):
        relative = csv_path.relative_to(root)
        allowed = {"debug_widthonly", "round1", "audit", "debug_round2"}
        if relative.parts[0] not in allowed or not (csv_path.parent / "run.json").exists():
            continue
        try:
            run=json.loads((csv_path.parent / "run.json").read_text())
            if run.get("status") != "completed": continue
            manifest_path = csv_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            manifest_task = manifest.get("args", {}).get("task", manifest.get("task", csv_path.parent.name))
            manifest_steps = manifest.get("args", {}).get("steps", manifest.get("steps", ""))
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    if relative.parts[0] == "round1" and row.get("seed") != "0":
                        continue
                    row["source_csv"] = str(csv_path)
                    row["source_artifact"] = str(csv_path)
                    row["run_dir"] = str(csv_path.parent)
                    row["elapsed_seconds"] = run.get("elapsed_seconds", "")
                    row["task"] = manifest_task
                    m=re.search(r"(?:^|_)w(\d+)(?:_|$)", csv_path.parent.name)
                    row["width"] = m.group(1) if m else ""
                    m=re.search(r"(?:^|_)s(\d+)(?:_|$)", csv_path.parent.name)
                    row["steps"] = m.group(1) if m else str(manifest_steps)
                    row["family"] = "width_only" if "debug_widthonly" in csv_path.parts else ("modern_debug" if "debug_round2" in csv_path.parts else "round1")
                    row["model"] = row.get("kind", "")
                    row["depth"] = "1"
                    row["score_scope"] = "legacy_tail"
                    row["context_policy"] = "trial_causal_fixed_window"
                    yield row
        except (OSError, json.JSONDecodeError):
            continue


def run(results: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows=list(_rows(results))
    fields=sorted({k for r in rows for k in r})
    with (output/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    width=[r for r in rows if r["family"]=="width_only" and r.get("adaptation")=="ridge"]
    # Raw width-64 seed-0 anchors use the same legacy-tail/support-33 contract.
    for row in rows:
        if row.get("family")=="round1" and row.get("kind")=="osc" and row.get("adaptation")=="ridge" and row.get("support_trials")=="33":
            anchor=dict(row);anchor["family"]="raw64_anchor";anchor["width"]="64";width.append(anchor)
    fig,ax=plt.subplots(figsize=(6,4))
    for task in sorted({r.get("task", "") for r in width}):
        data=sorted([r for r in width if r.get("task")==task and (r["steps"]=="600" or r["family"]=="raw64_anchor")],key=lambda r:int(r["width"]))
        ax.plot([int(r["width"]) for r in data],[float(r["r2_variance_weighted"]) for r in data],"o-",label=f"{task.upper()} ridge, seed 0")
    ax.set(xlabel="oscillator width (600 optimizer steps)",ylabel="target variance-weighted R²");ax.legend();fig.tight_layout()
    fig.savefig(output/"scale_vs_r2.png",dpi=180);fig.savefig(output/"scale_vs_r2.pdf");plt.close(fig)
    modern=[r for r in rows if r.get("family")=="modern_debug" and r.get("mode")=="cross_session" and r.get("score_scope")=="all_valid_trial" and r.get("adaptation")=="ridge"]
    for filename,xkey,xlabel in (("params_vs_allvalid","params","parameters"),("sourceval_vs_target","best_source_val_r2","best source validation R²")):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,task in zip(axes, sorted({r.get("task","") for r in modern})):
            d=[r for r in modern if r.get("task")==task and r.get(xkey,"") not in ("",None)]
            for policy, marker in (("trial_causal_fixed_window", "o"), ("recording_causal_fixed_window", "s")):
                q=[r for r in d if r.get("context_policy")==policy]
                if q:
                    ax.scatter([float(r[xkey]) for r in q],[float(r.get("r2_variance_weighted",float("nan"))) for r in q],marker=marker,label=f"{task.upper()} {policy.split('_')[0]}")
                    for r in q:
                        if r.get("model") in {"s4d", "mamba2_official", "mamba3_official"}:
                            ax.annotate(f"{r['model'].replace('_official','')} {r.get('width')}",(float(r[xkey]),float(r.get('r2_variance_weighted',0))),fontsize=5)
            ax.set_title(task.upper());ax.set(xlabel=xlabel,ylabel="all-valid target R²");ax.legend(fontsize=7)
            if xkey=="params": ax.set_xscale("log")
        fig.tight_layout();fig.savefig(output/(filename+".png"),dpi=180);fig.savefig(output/(filename+".pdf"));plt.close(fig)
    curves=[]
    for log in results.rglob("train_log.jsonl"):
        if "smoke" in str(log): continue
        try:
            metrics = log.parent / "metrics.json"
            if not metrics.exists():
                continue
            metric_payload = json.loads(metrics.read_text())
            if metric_payload.get("status") != "completed": continue
            values=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
            if values: curves.append((log,values,metric_payload.get("args",{})))
        except (OSError,json.JSONDecodeError): pass
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=False)
    for log,values,args in curves:
        metric=values[0].get("source_val_physical_variance_weighted_r2")
        if metric is None: continue
        if "cross_session" not in log.parent.name or not any(k in log.parent.name for k in ("s4d", "mamba2_official", "mamba3_official")): continue
        task="m1" if "/m1/" in str(log) else "m2"; ax=axes[0 if task=="m1" else 1]
        pairs=[(v.get("step",i),v.get("source_val_physical_variance_weighted_r2")) for i,v in enumerate(values) if v.get("step",i)>=200]
        if not pairs: continue
        xs,ys=zip(*pairs); kind=str(args.get("kind","model")).replace("_official","").replace("mamba","Mamba").replace("s4d","S4D")
        label=f"{kind} {args.get('width','?')}x{args.get('layers','?')} N{args.get('state_size','?')}"
        ax.plot(xs,ys,alpha=.65,label=label)
    for ax,task in zip(axes,("M1","M2")): ax.set(title=f"{task}: after initialization",xlabel="optimizer step",ylabel="source validation physical R²");ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/"traincurves.png",dpi=180);fig.savefig(output/"traincurves.pdf");plt.close(fig)
    lines=["# SSM debug round 2 aggregation", "", "此文件由 `python -m ssm_decode.debug_report` 生成；只纳入 completed artifact。", "", "指标按 `score_scope` 分开：`legacy_tail` 与 `all_valid_trial` 不能横向混合。width-only/round1 的 CSV 为 legacy tail；modern metrics 保留两种 scope。", "", "## 数据来源", ""]
    lines += [f"- `{r.get('source_artifact','')}` ({r['family']}, {r.get('model',r.get('kind',''))}, seed {r.get('seed')}, {r.get('adaptation')}, {r.get('score_scope')})" for r in rows]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return rows


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=Path("SSM/results"));p.add_argument("--output",type=Path,default=Path("SSM/results/analysis/debug_round2"));a=p.parse_args(argv);print(len(run(a.results,a.output)))
if __name__=="__main__":main()
