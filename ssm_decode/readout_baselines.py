"""Fixed-alpha local-target readout diagnostics; no query labels are fitted."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np, torch
from .calibration import fit_ridge
from .data import DEFAULT_ROOT,load_recording,source_target_plan,split_support_query
from .experiment import _r2s
from .models import ModelConfig,build_model

ROOT=Path(__file__).resolve().parents[1]/'results'/'round1'
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _indices(record,count,seq):
 s,q=split_support_query(record,support_trials=count,query_start_trials=33,sequence=seq)
 support=np.unique(s[:,None]+np.arange(seq)[None,:]).reshape(-1);query=np.unique(q+seq-1)
 xvalid=record.eval_mask&np.isfinite(record.behavior).all(1);return support[xvalid[support]],query[xvalid[query]]
def _lag_features(x,record,idx,lags):
 out=[]
 for i in idx:
  row=[]
  for lag in lags:
   prior=i-lag; ok=any(a<=prior<=i<b for a,b in record.trial_bounds)
   row.append(x[prior] if ok else np.zeros(x.shape[1],np.float32))
  out.append(np.concatenate(row))
 return np.asarray(out,np.float32)
def _model_outputs_latents(model,x,record):
 ys=[];hs=[]
 with torch.inference_mode():
  for a,b in record.trial_bounds:
   state=None
   for t in range(a,b):
    y,state=model.step(torch.from_numpy(x[t:t+1]),state);h=state if state.ndim==2 else state[...,0]
    ys.append(y.numpy()[0]);hs.append(h.numpy()[0])
 # trial ranges cover full recording contiguously by loader contract
 return np.asarray(ys,np.float32),np.asarray(hs,np.float32)
def run(output=ROOT/'readout_baselines.json'):
 rows=[]; root=DEFAULT_ROOT
 for task in ('m1','m2'):
  run=ROOT/task;manifest=json.loads((run/'manifest.json').read_text());plan=source_target_plan(task,root=root);target=load_recording(task,'held_in',plan['cross_session_local_dev']['target_session'],root=root)
  with np.load(run/'source_normalizer.npz') as z: st={k:z[k] for k in z.files}
  x=((target.neural-st['x_mean'])/st['x_std']).astype(np.float32);y=((target.behavior-st['y_mean'])/st['y_std']).astype(np.float32);phys=target.behavior
  for count in (5,10,33):
   si,qi=_indices(target,count,50)
   for lags in ((0,),(0,1,2,4)):
    f=_lag_features(x,target,si,lags).astype(np.float64);qf=_lag_features(x,target,qi,lags).astype(np.float64);m=fit_ridge(torch.from_numpy(f),torch.from_numpy(y[si].astype(np.float64)),1.)
    p=m(torch.from_numpy(qf)).numpy().astype(np.float32)*st['y_std']+st['y_mean'];vw,ua,pd=_r2s(phys[qi],p)
    rows.append({'task':task,'family':'direct_neural_ridge','support_trials':count,'query_cutoff_trials':33,'lags':list(lags),'alpha':1.,'r2_variance_weighted':vw,'r2_uniform_average':ua,'r2_per_dimension':pd,'n_support':len(si),'n_query':len(qi),'feature_dim':f.shape[1],'readout_bytes_float64':int(f.shape[1]*y.shape[1]*8+y.shape[1]*8),'readout_macs_per_bin':int(f.shape[1]*y.shape[1])})
  for kind in ('diag','osc','bank','selective','gru'):
   ck=torch.load(run/f'{kind}_s0.pt',map_location='cpu');model=build_model(ModelConfig(**ck['model_config']));model.load_state_dict(ck['state_dict']);base,h=_model_outputs_latents(model,x,target)
   for count in (5,10,33):
    si,qi=_indices(target,count,50);m=fit_ridge(torch.from_numpy(h[si].astype(np.float64)),torch.from_numpy((y[si]-base[si]).astype(np.float64)),1.)
    p=(base[qi]+m(torch.from_numpy(h[qi].astype(np.float64))).numpy()).astype(np.float32)*st['y_std']+st['y_mean'];vw,ua,pd=_r2s(phys[qi],p)
    rows.append({'task':task,'family':'frozen_latent64_residual_ridge','kind':kind,'seed':0,'support_trials':count,'query_cutoff_trials':33,'alpha':1.,'r2_variance_weighted':vw,'r2_uniform_average':ua,'r2_per_dimension':pd,'n_support':len(si),'n_query':len(qi),'feature_dim':64,'readout_bytes_float64':int(64*y.shape[1]*8+y.shape[1]*8),'readout_macs_per_bin':int(64*y.shape[1])})
 out={'schema':'round1_fixed_alpha_readout_baselines_v1','alpha':1.,'common_query_cutoff_trials':33,'query_fit':False,'rows':rows,'source_files_sha256':{n:_sha(Path(__file__).parent/n) for n in ('readout_baselines.py','data.py','models.py','calibration.py')}};Path(output).write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':run()
