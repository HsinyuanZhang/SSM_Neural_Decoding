"""Eval-only: causal running (EMA) per-channel z-score, unlabeled, initialized from support."""
import json, numpy as np, torch
from dataclasses import replace
from ssm_decode import debug_experiment as d
from ssm_decode.data import source_target_plan, load_recording
from ssm_decode.mamba3_official import build_mamba3_official
torch.set_num_threads(1); dev=torch.device('cuda:0'); torch.cuda.set_device(dev)
def ema_norm(x, m0, v0, half):
    a=1-0.5**(1/half); m=m0.copy(); v=v0.copy(); out=np.empty_like(x,dtype=np.float64)
    for i in range(x.shape[0]):  # stats use bins < i only (strictly causal)
        out[i]=(x[i]-m)/np.sqrt(np.maximum(v,1e-12)); m=m+a*(x[i]-m); v=(1-a)*(v+a*(x[i]-m)**2)
    return out
res={}
for task in ['m1','m2']:
    plan=source_target_plan(task); t=load_recording(task,'held_in',plan['cross_session_local_dev']['target_session'])
    src=f'results/debug_round2/latest/{task}/cross_session_mamba3_official_w256_l4_n32_ctx128_seed0'
    z=np.load(src+'/normalizer.npz'); stats=tuple(z[k].astype('float32') for k in ('x_mean','x_std','y_mean','y_std'))
    u=[b for b in t.trial_bounds if b[1]-b[0]>=50][:33]; xs=np.concatenate([t.neural[a:b] for a,b in u]).astype(np.float64)
    tm,tv=xs.mean(0),xs.var(0); tv[tv<1e-12]=1.
    ck=torch.load(src+'/best.pt',map_location='cpu',weights_only=False); a=ck['args']
    m=build_mamba3_official(t.neural.shape[1],t.behavior.shape[1],width=a['width'],layers=a['layers'],state_size=a['state_size'],dropout=a['dropout']).to(dev); m.load_state_dict(ck['state_dict']); m.eval()
    for half in [3000, 15000, 50000]:
        xn=ema_norm(t.neural.astype(np.float64),tm,tv,half)
        # support bins keep exact support-zscore so the ridge fit is unchanged in spirit
        rec=replace(t,neural=(xn*stats[1]+stats[0]).astype('float32'))
        cache=d._target_predictions(m,rec,stats,dev,128)
        zz=d._score_target(cache,stats,'zero'); rr=d._score_target(cache,stats,'ridge')
        k=f'{task}/none/ema_half{half}'; res[k]={'zero_allvalid':zz['all_valid_query_trial_r2_variance_weighted'],'ridge_allvalid':rr['all_valid_query_trial_r2_variance_weighted'],'zero_legacy':zz['r2_variance_weighted']}
        print(k,{q:round(v,4) for q,v in res[k].items()},flush=True)
json.dump(res,open('results/analysis/audit_20261003/ema_eval.json','w'),indent=2)
