"""Measure temporal error structure from saved physical-unit predictions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ssm_decode.data import load_recording


def error_groups(truth,prediction,groups):
    truth,prediction = np.asarray(truth,dtype=np.float64),np.asarray(prediction,dtype=np.float64)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('truth and predictions must have equal finite shapes')
    mean = truth.mean(0)
    global_sse,global_sst = np.square(truth-prediction).sum(),np.square(truth-mean).sum()
    rows = []
    for label,mask in groups:
        y,p = truth[mask],prediction[mask]
        if not len(y):
            continue
        sse = np.square(y-p).sum()
        contribution = np.square(y-mean).sum()
        local_sst = np.square(y-y.mean(0)).sum()
        pred_sst = np.square(p-p.mean(0)).sum()
        rows.append({'group':label,'n_bins':len(y),'bin_fraction':len(y)/len(truth),
                     'r2':float(1-sse/local_sst) if local_sst>0 else None,
                     'sse':float(sse),'sst_global_mean':float(contribution),
                     'sse_fraction':float(sse/global_sse) if global_sse>0 else 0.,
                     'sst_fraction':float(contribution/global_sst) if global_sst>0 else 0.,
                     'prediction_truth_variance_ratio':float(pred_sst/local_sst) if local_sst>0 else None})
    return rows


def write_csv(path,rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    records,rows,evaluations = {},[],0
    for root in args.roots:
        root = root.resolve()
        matrix = json.loads((root/'matrix.json').read_text())
        selection = json.loads((root/'selections.json').read_text())
        for task,data in selection['tasks'].items():
            for method,info in data['methods'].items():
                for fit in info['selected']['fits']:
                    folder = Path(fit['fit'])/'query_evaluation'
                    metrics = json.loads((folder/'metrics.json').read_text())
                    key = (task,metrics['target_session'],matrix['data_root'])
                    if key not in records:
                        records[key] = load_recording(task,'held_in',key[1],root=Path(key[2]))
                    record = records[key]
                    with np.load(folder/'predictions.npz') as archive:
                        evaluations += 1
                        indices,truth = archive['allvalid_indices'],archive['truth_allvalid']
                        if not np.array_equal(truth,record.behavior[indices]):
                            raise ValueError('saved physical labels differ from target labels')
                        starts = np.asarray([a for a,b in record.trial_bounds])
                        position = indices-starts[np.searchsorted(starts,indices,side='right')-1]
                        quartile = np.minimum(np.arange(len(indices))*4//len(indices),3)
                        groups = {'query_quartile':[(f'Q{i+1}',quartile==i) for i in range(4)],
                                  'trial_position':[(f'[{a},{b})',(position>=a)&(position<b))
                                                    for a,b in ((0,10),(10,25),(25,50),(50,128),(128,10**9))]}
                        for calibration in ('zero','ridge'):
                            for split,selection_groups in groups.items():
                                for row in error_groups(truth,archive[calibration+'_allvalid'],selection_groups):
                                    rows.append(dict(row,root=str(root),profile=matrix.get('name',root.name),task=task,
                                                     method=method,seed=metrics['seed'],calibration=calibration,split=split,
                                                     normalization=matrix['normalization'],policy=matrix['policy']))
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in ('profile','task','method','calibration','split','group'))].append(row)
    summaries = []
    for key,values in grouped.items():
        entry = dict(zip(('profile','task','method','calibration','split','group'),key),n_seeds=len(values))
        for metric in ('r2','sse_fraction','sst_fraction','prediction_truth_variance_ratio','bin_fraction'):
            numbers = [row[metric] for row in values if row[metric] is not None]
            entry[metric+'_mean'] = float(np.mean(numbers)) if numbers else None
            entry[metric+'_population_sd'] = float(np.std(numbers)) if numbers else None
        summaries.append(entry)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'temporal_by_fit.csv',rows)
    write_csv(args.output/'temporal_summary.csv',summaries)
    (args.output/'contract.json').write_text(json.dumps({
        'primary_cohort':'all-valid query trial bins; saved raw physical labels match live labels',
        'r2':'each group uses its own per-output truth mean',
        'contributions':'SSE and SST fractions use the full-query truth mean, so disjoint groups sum to one',
        'query_selection':False,'purpose':'diagnostic after checkpoint and LR selection',
        'found_evaluations':evaluations},indent=2)+'\n')


if __name__=='__main__':
    main()
