"""Target-support-only causal linear diagnostic for round-2 debugging."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from .data import load_recording, source_target_plan, split_support_query

ALPHAS=(.1,1.,10.,100.)

def _r2(y,p):
    sse=((y-p)**2).sum(0); sst=((y-y.mean(0))**2).sum(0)
    return float(1-sse.sum()/max(sst.sum(),1e-12))

def _ridge(x,y,a):
    xm,ym=x.mean(0),y.mean(0); w=np.linalg.solve((x-xm).T@(x-xm)+a*np.eye(x.shape[1]),(x-xm).T@(y-ym)); return w,ym-xm@w

def _features(x,bounds,kind):
    pieces=[]
    for a,b in bounds:
        z=x[a:b]; lag=lambda k:np.vstack([np.zeros((k,x.shape[1]),np.float32),z[:-k]]) if k else z
        if kind=='raw': pieces.append((a,b,z))
        elif kind=='lags': pieces.append((a,b,np.concatenate([lag(k) for k in (0,1,2,4,8)],1)))
        else:
            ema=np.empty_like(z); prev=np.zeros(x.shape[1],np.float32)
            for i,row in enumerate(z): prev=(2/3)*prev+(1/3)*row; ema[i]=prev
            pieces.append((a,b,np.concatenate([ema,lag(0),lag(2),lag(4)],1)))
    out=np.zeros((len(x),pieces[0][2].shape[1]),np.float32)
    for a,b,value in pieces: out[a:b]=value
    return out

def run(task, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    plan=source_target_plan(task); r=load_recording(task,'held_in',plan['cross_session_local_dev']['target_session'])
    usable=[(a,b) for a,b in r.trial_bounds if b-a>=50]
    support_bounds=usable[:33]; train_bounds=usable[:26]; val_bounds=usable[26:33]
    x,y=r.neural.astype(np.float32),r.behavior.astype(np.float32)
    train_idx=np.concatenate([np.arange(a,b) for a,b in train_bounds]); mx=x[train_idx].mean(0); sx=x[train_idx].std(0).clip(1e-6); my=y[train_idx].mean(0); sy=y[train_idx].std(0).clip(1e-6)
    xn=(x-mx)/sx; yn=(y-my)/sy; support_idx=np.concatenate([np.arange(a,b) for a,b in support_bounds]); val_idx=np.concatenate([np.arange(a,b) for a,b in val_bounds])
    valid=lambda idx: idx[r.eval_mask[idx]&np.isfinite(y[idx]).all(1)]
    support_idx,val_idx=valid(support_idx),valid(val_idx)
    starts,qstarts=split_support_query(r,support_trials=33,sequence=50); tail=valid(np.unique(qstarts+49)); allq=valid(np.concatenate([np.arange(a,b) for a,b in usable[33:]]))
    records=[]; arrays={}
    for kind in ('raw','lags','ema'):
        feat=_features(xn,r.trial_bounds,kind); scores=[]
        for alpha in ALPHAS:
            w,c=_ridge(feat[valid(np.concatenate([np.arange(a,b) for a,b in train_bounds]))],yn[valid(np.concatenate([np.arange(a,b) for a,b in train_bounds]))],alpha)
            pv=(feat[val_idx]@w+c)*sy+my; scores.append(_r2(y[val_idx],pv))
        alpha=ALPHAS[int(np.argmax(scores))];w,c=_ridge(feat[support_idx],yn[support_idx],alpha);p=(feat@w+c)*sy+my
        for scope,idx in (('all_valid',allq),('legacy_tail',tail)):
            records.append({'feature':kind,'alpha':alpha,'prefix_val_r2':max(scores),'scope':scope,'r2':_r2(y[idx],p[idx]),'n_query':len(idx),'n_fit':len(support_idx)})
        arrays[kind]=p[allq]
    for name,p in [('zero',np.zeros_like(y)),('support_mean',np.broadcast_to(y[support_idx].mean(0),y.shape))]:
        for scope,idx in (('all_valid',allq),('legacy_tail',tail)): records.append({'feature':name,'alpha':None,'prefix_val_r2':None,'scope':scope,'r2':_r2(y[idx],p[idx]),'n_query':len(idx),'n_fit':len(support_idx)})
    qhash=hashlib.sha256(allq.tobytes()+tail.tobytes()).hexdigest(); np.savez_compressed(output/'predictions.npz',all_valid_indices=allq,legacy_tail_indices=tail,**arrays)
    receipt={'task':task,'status':'completed','protocol':'target prefix first26 train, next7 validation, refit all33; no query labels/features used','alphas':ALPHAS,'normalizer':'first26 usable trials only','query_indices_sha256':qhash,'records':records}
    (output/'metrics.json').write_text(json.dumps(receipt,indent=2)+'\n'); return receipt

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(run(a.task,a.output)['status'])
