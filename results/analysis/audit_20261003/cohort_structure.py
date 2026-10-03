"""Eval-only re-analysis of saved PEFT round-1 predictions (no training, no new labels).

Reports, per method (seed-averaged where noted):
  * all-valid R² by query-trial time quartile (within-session drift),
  * SSE/SST share and predicted/true variance ratio by position inside a trial.
"""
import glob, json
import numpy as np
from ssm_decode.data import source_target_plan, load_recording

ROOT = 'results/peft_round1/official_w256_l4_n32'
EDGES = [0, 10, 25, 50, 128, 10**9]
def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

out = {}
for task in ['m1', 'm2']:
    plan = source_target_plan(task)
    t = load_recording(task, 'held_in', plan['cross_session_local_dev']['target_session'])
    q = [b for b in t.trial_bounds if b[1] - b[0] >= 50][33:]
    tid = np.full(t.neural.shape[0], -1); pos = np.full(t.neural.shape[0], -1)
    for i, (a, b) in enumerate(q): tid[a:b] = i; pos[a:b] = np.arange(b - a)
    for meth in ['none', 'io', 'lora', 'full', 'state_offset']:
        quart, share = [], []
        for f in sorted(glob.glob(f'{ROOT}/{task}/{meth}_seed*/predictions.npz')):
            z = np.load(f); idx, y, p = z['allvalid_indices'], z['truth_allvalid'], z['zero_allvalid']
            k = tid[idx]; qs = np.quantile(k, [0, .25, .5, .75, 1.])
            masks = [(k >= lo) & ((k <= hi) if j == 3 else (k < hi)) for j, (lo, hi) in enumerate(zip(qs[:-1], qs[1:]))]
            quart.append([r2(y[m], p[m]) for m in masks])
            ps = pos[idx]; mu = y.mean(0); sse = ((y - p) ** 2).sum(); sst = ((y - mu) ** 2).sum()
            share.append([[float(((ps >= lo) & (ps < hi)).mean()),
                           float(((y[(ps >= lo) & (ps < hi)] - p[(ps >= lo) & (ps < hi)]) ** 2).sum() / sse),
                           float(((y[(ps >= lo) & (ps < hi)] - mu) ** 2).sum() / sst),
                           float(p[(ps >= lo) & (ps < hi)].var(0).sum() / max(y[(ps >= lo) & (ps < hi)].var(0).sum(), 1e-12))]
                          if ((ps >= lo) & (ps < hi)).sum() > 20 else [float('nan')] * 4
                          for lo, hi in zip(EDGES[:-1], EDGES[1:])])
        out[f'{task}/{meth}'] = {'r2_by_query_time_quartile_seedmean': np.mean(quart, 0).tolist(),
                                 'position_edges': EDGES[:-1],
                                 'position_frac_sseshare_sstshare_predvar_over_truthvar_seedmean': np.nanmean(share, 0).tolist()}
        print(task, meth, np.round(out[f'{task}/{meth}']['r2_by_query_time_quartile_seedmean'], 3).tolist(),
              np.round(np.nanmean(share, 0), 2).tolist())
json.dump(out, open('results/analysis/audit_20261003/cohort_structure.json', 'w'), indent=2)
