"""Verify and summarize completed PEFT fits without changing experiment data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import hashlib
import torch


def r2(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    numerator = ((y - prediction) ** 2).sum(axis=0)
    denominator = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return float(1.0 - numerator.sum() / max(1e-12, denominator.sum()))


def files(root):
    for metrics_path in root.rglob("metrics.json"):
        fit = metrics_path.parent
        required = [fit / name for name in ("manifest.json", "predictions.npz", "best.pt", "last.pt", "replay_args.json", "train_log.json")]
        yield fit, required


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(fit, required):
    metrics = json.loads((fit / "metrics.json").read_text())
    if metrics.get("status") != "completed":
        return None
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"completed fit missing artifacts: {missing}")
    manifest = json.loads((fit / "manifest.json").read_text())
    args = json.loads((fit / "replay_args.json").read_text())
    if metrics.get("status") != "completed":
        return None
    best = torch.load(fit / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(fit / "last.pt", map_location="cpu", weights_only=False)
    if best["peft_replay"] != last["peft_replay"]:
        raise AssertionError(f"best/last replay metadata differs: {fit}")
    if best['best_step'] != metrics['best_step']:
        raise AssertionError(f'checkpoint best step differs: {fit}')
    if best['args'] != args or last['args'] != args:
        raise AssertionError(f'checkpoint run args differ: {fit}')
    for key in ('method','seed'):
        if args[key] != manifest[key] or args[key] != metrics[key]:
            raise AssertionError(f'run identity differs for {key}: {fit}')
    if manifest.get('datamanifest',{}).get('task',args['task']) != args['task']:
        raise AssertionError(f'run task differs: {fit}')
    normalize_json = lambda value: json.loads(json.dumps(value))
    if normalize_json(best['receipt']) != manifest['receipt'] or normalize_json(last['receipt']) != manifest['receipt']:
        raise AssertionError(f'checkpoint receipt differs: {fit}')
    if hashlib.sha256(Path(args['pretrained']).read_bytes()).hexdigest() != manifest['pretrained_sha256']:
        raise AssertionError(f'source checkpoint hash differs: {fit}')
    if json.loads((fit/'code_hashes_start.json').read_text()) != manifest['code_hashes']:
        raise AssertionError(f'startup code hashes differ: {fit}')
    data = np.load(fit / "predictions.npz")
    copied = np.load(fit / "normalizer.npz")
    if digest(np.frombuffer((fit / "normalizer.npz").read_bytes(), dtype=np.uint8)) != manifest["statsfile_sha256"]:
        raise AssertionError(f"copied normalizer hash differs: {fit}")
    for key in ('x_mean','x_std','y_mean','y_std'):
        if not np.array_equal(copied[key],best['normalizer'][key]):
            raise AssertionError(f'checkpoint normalizer differs: {fit}')
    receipt = manifest['receipt']
    actual_count = sum(best['state_dict'][name].numel() for name in receipt['trainable_paths'])
    if actual_count != receipt['trainable_count'] or actual_count != metrics['trainable_params']:
        raise AssertionError(f'trainable parameter count differs: {fit}')
    authorized = {name for stage in manifest['optimizer_stages'] for group in stage['groups'] for name in group['names']}
    changed = {name for name, value in manifest['initial_parameter_hashes'].items()
               if manifest['final_parameter_hashes'][name] != value}
    if not changed.issubset(authorized):
        raise AssertionError(f'frozen parameter changed: {fit}: {changed-authorized}')
    row = {"fit": str(fit), "task": args.get("task", manifest.get("task")),
           "method": metrics.get("method", manifest.get("method")), "seed": args.get("seed", manifest.get("seed")),
           "policy": manifest.get("policy"), "pretrained_sha256": manifest.get("pretrained_sha256"),
           "normalizer_sha256": manifest.get("statsfile_sha256"),
           "best_step": metrics.get("best_step"), "prefix_val_r2": metrics.get("best_prefix_val_r2"),
           "trainable_params": metrics.get("trainable_params"), "base_params": metrics.get("base_parameters"),
           "resource_fit_seconds": metrics.get("fit_seconds"), "peak_vram_bytes": None if metrics.get('peak_cuda_allocated_mb') is None else metrics['peak_cuda_allocated_mb']*2**20}
    for scope in ("legacy", "allvalid"):
        truth = data[f"truth_{scope}"]
        indices = data[f"{scope}_indices"]
        for adapt in ("zero", "ridge"):
            pred = data[f"{adapt}_{scope}"]
            if not np.isfinite(pred).all():
                raise AssertionError(f"nonfinite query prediction: {fit}")
            value = r2(truth, pred)
            row[f"{adapt}_{scope}_r2"] = value
            reported = metrics["final"][adapt]["all_valid" if scope == "allvalid" else "legacy"].get("r2_variance_weighted")
            if reported is not None and not np.isclose(value, reported, atol=1e-5, rtol=0):
                raise AssertionError(f"R2 mismatch: {fit} {adapt} {scope}: {value} != {reported}")
        row[f"{scope}_index_hash"] = digest(indices)
        row[f"{scope}_truth_hash"] = digest(truth)
    support = data["support_indices"]
    row['support_index_hash'] = digest(support)
    row['support_truth_hash'] = digest(data['support_truth_normalized'])
    query = data["allvalid_indices"]
    if np.intersect1d(support, query).size:
        raise AssertionError(f"support/query overlap: {fit}")
    if row["method"] == "sparse_sdt_m3":
        path = fit / "sparse_selection.json"
        if not path.exists(): raise AssertionError(f"missing sparse selection: {fit}")
        sparse = json.loads(path.read_text())
        normalize = lambda mapping: {str(key):value for key,value in mapping.items()}
        selected = sparse.get('selection_receipt',{}).get('selections',{})
        if normalize(selected) != normalize(best['peft_replay'].get('sparse_selections',{})):
            raise AssertionError(f"sparse replay selections differ: {fit}")
        if sparse['selection_receipt'] != manifest['sparse_selection']:
            raise AssertionError(f'sparse manifest selection differs: {fit}')
        if not sparse.get("warmup_log") or not all(np.isfinite(x["loss"]) for x in sparse["warmup_log"]):
            raise AssertionError(f"invalid sparse warmup log: {fit}")
    log = json.loads((fit / "train_log.json").read_text())
    values = [(x["step"], x["prefix_val_r2"]) for x in log if "prefix_val_r2" in x]
    if row["method"] != "none" and (not values or not any(step==0 for step,score in values)):
        raise AssertionError(f"no recorded prefix validation score: {fit}")
    if values and row["best_step"] != max(values, key=lambda x: x[1])[0]:
        raise AssertionError(f"best step is not prefix validation argmax: {fit}")
    if values and not np.isclose(max(score for step,score in values),metrics['best_prefix_val_r2'],atol=1e-8,rtol=0):
        raise AssertionError(f'best validation score differs: {fit}')
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, errors, reference = [], [], {}
    for fit, required in files(args.root):
        try:
            row = read(fit, required)
            if row is None:
                continue
            key = row["task"]
            signature = tuple(row[name] for name in ('pretrained_sha256','normalizer_sha256','policy',
                'allvalid_index_hash','allvalid_truth_hash','legacy_index_hash','legacy_truth_hash',
                'support_index_hash','support_truth_hash'))
            if key in reference and reference[key] != signature:
                raise AssertionError(f"cohort or metadata differs for task {key}")
            reference[key] = signature
            rows.append(row)
        except Exception as exc:
            errors.append({"fit": str(fit), "error": str(exc)})
    status = "waiting" if not rows and not errors else ("failed" if errors else "verified")
    if rows:
        with (args.output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
        groups = {}
        for row in rows:
            groups.setdefault((row["task"], row["method"]), []).append(row)
        summary=[]
        for (task, method), values in groups.items():
            out={"task":task,"method":method,"n":len(values)}
            for key in ("zero_allvalid_r2","ridge_allvalid_r2","zero_legacy_r2","ridge_legacy_r2","trainable_params","base_params","resource_fit_seconds","peak_vram_bytes"):
                x=np.array([v[key] for v in values if v.get(key) is not None],dtype=float)
                if x.size: out[key+"_mean"]=float(x.mean());out[key+"_std"]=float(x.std(ddof=0))
            summary.append(out)
        with (args.output / "groupedsummary.csv").open("w", newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=sorted({k for x in summary for k in x}));writer.writeheader();writer.writerows(summary)
    (args.output / "verification.json").write_text(json.dumps({"status":status,"fits":len(rows),"errors":errors,"rows":rows},indent=2)+"\n")
    if errors: raise SystemExit("verification failed")

if __name__ == "__main__":
    main()
