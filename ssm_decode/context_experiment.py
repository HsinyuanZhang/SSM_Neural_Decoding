"""Recording-continuous causal-window adapter for the debug runner."""
from __future__ import annotations
import argparse, hashlib
import numpy as np, torch
from . import debug_experiment as d

POLICY='recording_causal_fixed_window'

def _windows(records,bounds,stats,device):
    out=[]
    for r,bs in zip(records,bounds):
        x,y=d._norm(r,stats); mask=np.zeros(len(x),bool)
        for a,b in bs: mask[a:b]=r.eval_mask[a:b]
        out.append((torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),torch.from_numpy(mask).to(device)))
    return out

def _predict_endpoints(m,x,bounds,ends,device,context,batch=256):
    ends=np.asarray(ends); out=np.empty((len(ends),m.config.output_size),np.float32); was=m.training;m.eval()
    with torch.inference_mode():
        for lo in range(0,len(ends),batch):
            ee=ends[lo:lo+batch];w=np.zeros((len(ee),context,x.shape[1]),np.float32)
            for j,e in enumerate(ee):
                left=max(0,int(e)-context+1);w[j,context-(int(e)-left+1):]=x[left:int(e)+1]
            out[lo:lo+len(ee)]=d._forward(m,torch.from_numpy(w).to(device))[:,-1].float().cpu().numpy()
    if was:m.train()
    return out

def run(a):
    oldw,oldp,oldh=d._windows,d._predict_endpoints,d._hashes
    adapter_hash=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    def hashes():
        z=oldh();z['context_experiment.py']=adapter_hash;return z
    d._hashes=hashes
    d._windows,d._predict_endpoints=_windows,_predict_endpoints
    try:
        out=d.run(a)
        import json
        p=out/'manifest.json'; z=json.loads(p.read_text());z['context_policy']=POLICY;z['no_label_fit_crossboundary']=True;z['context_adapter_sha256']=adapter_hash;p.write_text(json.dumps(z,indent=2)+'\n')
        q=out/'metrics.json'; z=json.loads(q.read_text());z['context_policy']=POLICY;z['context_adapter_sha256']=adapter_hash;q.write_text(json.dumps(z,indent=2)+'\n')
        return out
    finally:d._windows,d._predict_endpoints,d._hashes=oldw,oldp,oldh

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['m1','m2'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--width',type=int,required=True);p.add_argument('--layers',type=int,required=True);p.add_argument('--state-size',type=int,default=16);p.add_argument('--steps',type=int,default=2000);p.add_argument('--context',type=int,default=128);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--eval-batch-size',type=int,default=64)
    a=p.parse_args();a.kind='s4d';a.mode='cross_session';a.dropout=.1;a.val_interval=200;a.max_val_endpoints=1024;a.lr=.001;a.weight_decay=1e-4;a.seed=0;a.warmstart=None;a.adapt_parameters='all';a.data_root=str(d.DEFAULT_ROOT);print(run(a))
if __name__=='__main__':main()
