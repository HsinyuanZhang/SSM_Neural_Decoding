"""Read-only state/reset diagnostic for the completed tiny-SSM pilot."""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np
import torch
from .data import DEFAULT_ROOT, load_recording, source_target_plan, split_support_query
from .models import ModelConfig, build_model
from .calibration import fit_ridge
from .experiment import _r2s

OUT=Path(__file__).resolve().parents[1]/'results'/'debug_audit'
def _hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _mask_idx(r):
 s,q=split_support_query(r,support_trials=33,query_start_trials=33,sequence=50);ss=np.unique(s[:,None]+np.arange(50)[None,:]).reshape(-1);qq=np.unique(q+49)
 good=r.eval_mask&np.isfinite(r.behavior).all(1);return ss[good[ss]],qq[good[qq]]
def _run(model,x,r,mode,device,need=None):
 """Return normalized predictions over recording; each trial is independent."""
 out=np.zeros((len(x),model.config.output_size),np.float32)
 with torch.inference_mode():
  for a,b in r.trial_bounds:
   if mode=='full_trial':
    out[a:b]=model(torch.from_numpy(x[a:b])[None].to(device)).cpu().numpy()[0]
   elif mode=='reset50':
    starts=list(range(a,b,50)); pack=np.zeros((len(starts),50,x.shape[1]),np.float32);lens=[]
    for j,s in enumerate(starts):lens.append(min(50,b-s));pack[j,:lens[-1]]=x[s:s+lens[-1]]
    y=model(torch.from_numpy(pack).to(device)).cpu().numpy()
    for j,s in enumerate(starts):out[s:s+lens[j]]=y[j,:lens[j]]
   else:
    points=np.arange(a,b) if need is None else np.asarray([i for i in need if a<=i<b])
    if not len(points):continue
    for k in range(0,len(points),256):
     pts=points[k:k+256];pack=np.zeros((len(pts),50,x.shape[1]),np.float32)
     for j,t in enumerate(pts): pack[j,max(0,50-(t-a+1)):]=x[max(a,t-49):t+1]
     out[pts]=model(torch.from_numpy(pack).to(device)).cpu().numpy()[:,-1]
 return out
def _metrics(y,p,idx):
 e=y[idx]-p[idx];vw,ua,pd=_r2s(y[idx],p[idx]);return {'r2_variance_weighted':vw,'r2_uniform_average':ua,'r2_per_dimension':pd,'mse':float(np.mean(e*e)),'y_std':np.std(y[idx],0).astype(float).tolist(),'prediction_mean':p[idx].mean(0).astype(float).tolist(),'prediction_std':p[idx].std(0).astype(float).tolist()}
def _quarters(r,y,p,idx):
 rows=[]
 for a,b in r.trial_bounds:
  here=idx[(idx>=a)&(idx<b)]
  if not len(here):continue
  pos=(here-a)/max(1,b-a);rows.append([float(np.mean((y[here[(pos>=q)&(pos<q+.25)]]-p[here[(pos>=q)&(pos<q+.25)]])**2)) if np.any((pos>=q)&(pos<q+.25)) else None for q in (0,.25,.5,.75)])
 arr=np.asarray([[np.nan if v is None else v for v in row] for row in rows]);return np.nanmean(arr,0).astype(float).tolist()
def _decay(model):
 A=model.recurrence_matrices().detach().cpu().numpy()
 if not len(A):return {'kind':'gru','timeconstant_20ms':'not_linear_recurrence'}
 vals=np.linalg.eigvals(A);mag=np.abs(vals);valid=mag[(mag>0)&(mag<1)];tau=(-.02/np.log(valid)).astype(float)
 return {'spectral_radius_mean':float(mag.mean()),'spectral_radius_max':float(mag.max()),'timeconstant_seconds_median':float(np.median(tau)),'timeconstant_seconds_minmax':[float(tau.min()),float(tau.max())]}
def run(device='cuda:0'):
 OUT.mkdir(parents=True,exist_ok=False);dev=torch.device(device);allrows=[]
 for task in ('m1','m2'):
  root=Path(DEFAULT_ROOT);plan=source_target_plan(task,root=root);target=load_recording(task,'held_in',plan['cross_session_local_dev']['target_session'],root=root);runroot=Path(__file__).resolve().parents[1]/'results'/'round1'/task
  with np.load(runroot/'source_normalizer.npz') as z:st={k:z[k] for k in z.files}
  x=((target.neural-st['x_mean'])/st['x_std']).astype(np.float32);y=((target.behavior-st['y_mean'])/st['y_std']).astype(np.float32);yp=target.behavior.astype(np.float32);si,qi=_mask_idx(target)
  reader={'neural_shape':list(target.neural.shape),'behavior_shape':list(target.behavior.shape),'trial_change_shape':list(target.trial_change.shape),'eval_mask_shape':list(target.eval_mask.shape),'trial_count':len(target.trial_bounds),'eval_mask_true':int(target.eval_mask.sum()),'alignment':'APST official load_nwb_file returns co-timed neural/covariates; audit uses identical integer bin indices','query_fit':False}
  for kind in ('diag','osc','gru'):
   ck=torch.load(runroot/f'{kind}_s0.pt',map_location=dev);m=build_model(ModelConfig(**ck['model_config'])).to(dev);m.load_state_dict(ck['state_dict']);m.eval();modes={}
   for mode in ('full_trial','reset50','rolling50'):
    pred=_run(m,x,target,mode,dev,np.unique(np.r_[si,qi]) if mode=='rolling50' else None)
    # fill missing rolling non-index with zero is harmless because only support/query are queried
    zero=_metrics(yp,pred*st['y_std']+st['y_mean'],qi);support=_metrics(y,pred,si)
    residual=y[si]-pred[si];ridge=fit_ridge(torch.from_numpy(pred[si].astype(np.float64)),torch.from_numpy(residual.astype(np.float64)),1.)
    adj=pred.copy();adj[qi]+=ridge(torch.from_numpy(pred[qi].astype(np.float64))).numpy().astype(np.float32)
    ridge_q=_metrics(yp,adj*st['y_std']+st['y_mean'],qi)
    modes[mode]={'zero_query_physical':zero,'ridge_query_physical':ridge_q,'support_normalized':support,'query_error_by_trial_quartile':_quarters(target,y,pred,qi)}
   # Source loss uses same source normalized scaler and first 50-bin reset/full distinction on one fixed source session.
   src=load_recording(task,'held_in',plan['source_held_in_sessions'][0],root=root);sx=((src.neural-st['x_mean'])/st['x_std']).astype(np.float32);sy=((src.behavior-st['y_mean'])/st['y_std']).astype(np.float32);idx=np.flatnonzero(src.eval_mask&np.isfinite(sy).all(1));pf=_run(m,sx,src,'full_trial',dev);pr=_run(m,sx,src,'reset50',dev)
   allrows.append({'task':task,'kind':kind,'seed':0,'checkpoint':str(runroot/f'{kind}_s0.pt'),'checkpoint_sha256':_hash(runroot/f'{kind}_s0.pt'),'reader':reader,'support_rows':int(len(si)),'query_rows':int(len(qi)),'decay':_decay(m),'source_fixed_session_loss':{'session':src.session,'full_trial_mse':float(np.mean((sy[idx]-pf[idx])**2)),'reset50_mse':float(np.mean((sy[idx]-pr[idx])**2))},'modes':modes})
 result={'schema':'tiny_ssm_state_debug_audit_v1','scope':'held-in local development diagnostic, no query fitting except support-only ridge','device':str(dev),'rows':allrows,'source_code_sha256':{n:_hash(Path(__file__).parent/n) for n in ('debug_audit.py','data.py','models.py','calibration.py')}}
 (OUT/'debug_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

def sampling_audit(device='cuda:0'):
 """Append coverage/scale and all-valid-bin scoring without retraining."""
 path=OUT/'debug_audit.json';payload=json.loads(path.read_text());dev=torch.device(device);rows=[]
 for task in ('m1','m2'):
  root=Path(DEFAULT_ROOT);plan=source_target_plan(task,root=root);r=load_recording(task,'held_in',plan['cross_session_local_dev']['target_session'],root=root);runroot=Path(__file__).resolve().parents[1]/'results'/'round1'/task
  with np.load(runroot/'source_normalizer.npz') as z:st={k:z[k] for k in z.files}
  x=((r.neural-st['x_mean'])/st['x_std']).astype(np.float32);y=((r.behavior-st['y_mean'])/st['y_std']).astype(np.float32);yp=r.behavior.astype(np.float32);si,oldq=_mask_idx(r)
  # Match split_support_query: only 50-bin-capable trials participate.  This
  # makes the all-bin denominator exactly the same post-prefix trial cohort as
  # the historical endpoint score, rather than silently adding short trials.
  usable50=[(a,b) for a,b in r.trial_bounds if b-a>=50]
  support_trials=usable50[:33];usable=usable50[33:]
  def valid_bins(bounds, tail=False):
   pieces=[]
   for a,b in bounds:
    start=a+49 if tail else a
    if start<b: pieces.append(np.arange(start,b))
   z=np.concatenate(pieces) if pieces else np.empty(0,dtype=np.int64)
   return z[r.eval_mask[z]&np.isfinite(y[z]).all(1)]
  allq=valid_bins(usable);support_all=valid_bins(support_trials);support_tail=valid_bins(support_trials,tail=True)
  def stats(idx,bounds):
   pos=np.concatenate([(idx[(idx>=a)&(idx<b)]-a)/max(1,b-a) for a,b in bounds if np.any((idx>=a)&(idx<b))])
   return {'bins':int(len(idx)),'physical_std':np.std(yp[idx],0).astype(float).tolist(),'time_quantiles':np.quantile(pos,[0,.25,.5,.75,1]).astype(float).tolist()}
  coverage={'usable50_trials_total':len(usable50),'support_trials':len(support_trials),'query_trials':len(usable),'old_tail_bins':int(len(oldq)),'all_valid_bins':int(len(allq)),'old_tail_fraction_of_all':float(len(oldq)/len(allq)),'old_tail':stats(oldq,usable),'query_all_valid':stats(allq,usable),'support_relative_tail_ge_50':stats(support_tail,support_trials),'support_all_valid':stats(support_all,support_trials)}
  scores={}
  for kind in ('osc','gru'):
   ck=torch.load(runroot/f'{kind}_s0.pt',map_location=dev);m=build_model(ModelConfig(**ck['model_config'])).to(dev);m.load_state_dict(ck['state_dict']);m.eval();pred=_run(m,x,r,'full_trial',dev)
   ridge=fit_ridge(torch.from_numpy(pred[si].astype(np.float64)),torch.from_numpy((y[si]-pred[si]).astype(np.float64)),1.);adj=pred.copy();adj[allq]+=ridge(torch.from_numpy(pred[allq].astype(np.float64))).numpy().astype(np.float32)
   scores[kind]={'old_tail_r2':_metrics(yp,adj*st['y_std']+st['y_mean'],oldq)['r2_variance_weighted'],'all_valid_same_query_trials_r2':_metrics(yp,adj*st['y_std']+st['y_mean'],allq)['r2_variance_weighted'],'ridge_fit':'prefix33_support_only'}
  rows.append({'task':task,'coverage':coverage,'models':scores})
 payload['sampling_audit']={'scope':'same post-prefix query trials; labels used only for score','rows':rows};path.write_text(json.dumps(payload,indent=2)+'\n');return payload['sampling_audit']
if __name__=='__main__':run()
