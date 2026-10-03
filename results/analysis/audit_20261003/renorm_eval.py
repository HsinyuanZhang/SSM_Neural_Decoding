"""Eval-only: source vs target-support (unlabeled) input normalization."""
import sys, json, numpy as np, torch
from dataclasses import replace
from ssm_decode import debug_experiment as d
from ssm_decode.data import source_target_plan, load_recording
from ssm_decode.mamba3_official import build_mamba3_official
from ssm_decode.peft_experiment import load_saved_model
torch.set_num_threads(1); dev=torch.device('cuda:0'); torch.cuda.set_device(dev)
R='results/peft_round1/official_w256_l4_n32'
out={}
for task in ['m1','m2']:
    plan=source_target_plan(task); t=load_recording(task,'held_in',plan['cross_session_local_dev']['target_session'])
    src_ck=f'results/debug_round2/latest/{task}/cross_session_mamba3_official_w256_l4_n32_ctx128_seed0'
    z=np.load(src_ck+'/normalizer.npz'); stats=tuple(z[k].astype('float32') for k in ('x_mean','x_std','y_mean','y_std'))
    u=[b for b in t.trial_bounds if b[1]-b[0]>=50][:33]
    xs=np.concatenate([t.neural[a:b] for a,b in u]).astype(np.float64)
    tm,ts=xs.mean(0),xs.std(0); ts[ts<1e-6]=1.
    variants={'source_norm':t,
              'target_support_mean':replace(t,neural=(t.neural-tm+stats[0]).astype('float32')),
              'target_support_zscore':replace(t,neural=((t.neural-tm)/ts*stats[1]+stats[0]).astype('float32'))}
    models={}
    ck=torch.load(src_ck+'/best.pt',map_location='cpu',weights_only=False); a=ck['args']
    m=build_mamba3_official(t.neural.shape[1],t.behavior.shape[1],width=a['width'],layers=a['layers'],state_size=a['state_size'],dropout=a['dropout']).to(dev); m.load_state_dict(ck['state_dict']); models['none']=m.eval()
    for meth in ['io','lora','full','state_offset']:
        models[meth]=load_saved_model(f'{R}/{task}/{meth}_seed0/best.pt',device=dev)[0]
    for mn,m in models.items():
        for vn,rec in variants.items():
            cache=d._target_predictions(m,rec,stats,dev,128)
            zz=d._score_target(cache,stats,'zero'); rr=d._score_target(cache,stats,'ridge')
            key=f'{task}/{mn}/{vn}'
            out[key]={'zero_allvalid':zz['all_valid_query_trial_r2_variance_weighted'],'ridge_allvalid':rr['all_valid_query_trial_r2_variance_weighted'],'zero_legacy':zz['r2_variance_weighted']}
            print(key,{k:round(v,4) for k,v in out[key].items()},flush=True)
json.dump(out,open('results/analysis/audit_20261003/renorm_eval.json','w'),indent=2)
