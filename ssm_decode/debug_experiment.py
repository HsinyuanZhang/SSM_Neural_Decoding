"""Causal SSM diagnostic runner; checkpoint selection is source-only."""
from __future__ import annotations
import argparse, hashlib, json, random, time, subprocess, os
from pathlib import Path
import numpy as np
import torch
from .calibration import fit_ridge
from .data import DEFAULT_ROOT, load_recording, source_target_plan, split_support_query
from .models import ModelConfig, build_model
from .modern_models import ModernConfig, build_modern_model

def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def _r2(y,p): return float(1-(y-p).square().sum()/(y-y.mean(0,keepdim=True)).square().sum().clamp_min(1e-12))
def _trial_split(r,frac=.2):
    b=[z for z in r.trial_bounds if z[1]-z[0]>=2]; cut=max(1,int(len(b)*(1-frac))); return b[:cut],b[cut:]
def _split_60_20_20(bounds):
    usable=[z for z in bounds if z[1]-z[0]>=50];n=len(usable);a60=max(1,int(.6*n));a80=max(a60+1,int(.8*n))
    if a80>=n: raise ValueError('within_session_split requires train, validation, and query trials')
    return usable[:a60],usable[a60:a80],usable[a80:]
def _freeze_io(model):
    roots=("input.","in_proj.","readout.","out_proj.","final_norm.")
    for name,p in model.named_parameters(): p.requires_grad=name.startswith(roots)
def _stats(rs,bs):
    x=np.concatenate([r.neural[a:b] for r,z in zip(rs,bs) for a,b in z]); y=np.concatenate([r.behavior[a:b] for r,z in zip(rs,bs) for a,b in z])
    xm,xs,ym,ys=x.mean(0),x.std(0),y.mean(0),y.std(0); xs[xs<1e-6]=1;ys[ys<1e-6]=1
    return tuple(z.astype('float32') for z in (xm,xs,ym,ys))
def _warm_stats(checkpoint):
    path=Path(checkpoint).parent/'normalizer.npz'
    if not path.exists(): raise FileNotFoundError(f'warmstart normalizer missing: {path}')
    z=np.load(path); return tuple(z[k].astype('float32') for k in ('x_mean','x_std','y_mean','y_std')),str(path)
def _norm(r,s): return ((r.neural-s[0])/s[1]).astype('float32'),((r.behavior-s[2])/s[3]).astype('float32')
def _model(a,c,d):
    if a.kind in {'s4d','mamba2'}: return build_modern_model(ModernConfig(c,d,a.kind,a.width,a.layers,a.state_size,a.dropout))
    if a.kind=='mamba3_official':
        from .mamba3_official import build_mamba3_official
        return build_mamba3_official(input_size=c,output_size=d,width=a.width,layers=a.layers,state_size=a.state_size,dropout=a.dropout)
    if a.kind=='mamba2_official':
        from .mamba2_official import build_mamba2_official
        return build_mamba2_official(input_size=c,output_size=d,width=a.width,layers=a.layers,state_size=a.state_size,dropout=a.dropout)
    return build_model(ModelConfig(c,d,width=a.width,kind=a.kind))
def _forward(m,x):
    y=m(x); return y[0] if isinstance(y,tuple) else y

def _windows(records,bounds,stats,device):
    """All complete trials, including trials shorter than context (left padded later)."""
    ans=[]
    for r,bs in zip(records,bounds):
        x,y=_norm(r,stats)
        for a,b in bs:
            if b-a>=2: ans.append((torch.from_numpy(x[a:b]).to(device),torch.from_numpy(y[a:b]).to(device),torch.from_numpy(r.eval_mask[a:b]).to(device)))
    if not ans: raise ValueError('no usable trials')
    return ans
def _right_aligned(windows, picks, context):
    """Return causal left-zero padded windows, with padding marked invalid."""
    c=windows[0][0].shape[1]; d=windows[0][1].shape[1]; n=len(picks)
    xx=torch.zeros((n,context,c),device=windows[0][0].device); yy=torch.zeros((n,context,d),device=xx.device); mm=torch.zeros((n,context),dtype=torch.bool,device=xx.device)
    for row,(wi,end) in enumerate(picks):
        x,y,m=windows[wi]; left=max(0,end-context+1); length=end-left+1
        xx[row,context-length:]=x[left:end+1]; yy[row,context-length:]=y[left:end+1]; mm[row,context-length:]=m[left:end+1]
    return xx,yy,mm
def _eligible(windows):
    return [[int(i) for i in torch.nonzero(m,as_tuple=False).flatten().cpu().tolist()] for _,_,m in windows]
def _sample_batch(windows,eligible,context,batch_size):
    weights=np.asarray([len(x) for x in eligible],dtype=float)
    if not weights.sum(): raise ValueError('training trials have no eval_mask bins')
    chosen=np.random.choice(len(windows),batch_size,p=weights/weights.sum())
    return _right_aligned(windows,[(int(i),random.choice(eligible[int(i)])) for i in chosen],context)
def _validation_selection(windows,context,maximum):
    # All eligible trial endpoints are candidates; cap is global, deterministic.
    cand=[(wi,end) for wi,ends in enumerate(_eligible(windows)) for end in ends]
    if not cand: raise ValueError('source validation has no evaluated endpoints')
    selected=[cand[i] for i in np.linspace(0,len(cand)-1,min(maximum,len(cand)),dtype=np.int64)]
    encoded=json.dumps(selected,separators=(',',':')); return selected,hashlib.sha256(encoded.encode()).hexdigest()
def _val(m,windows,context,selection=None,batch_size=256,stats=None):
    if selection is None: selection,_=_validation_selection(windows,context,1024)
    was=m.training;m.eval(); ys=[];ps=[]
    with torch.inference_mode():
        for lo in range(0,len(selection),batch_size):
            xx,yy,mm=_right_aligned(windows,selection[lo:lo+batch_size],context); p=_forward(m,xx)[:,-1]
            if not torch.isfinite(p).all(): raise FloatingPointError('non-finite source validation prediction')
            # The selected endpoint is eval-valid by construction; retain finite guard.
            good=mm[:,-1]&torch.isfinite(yy[:,-1]).all(-1)&torch.isfinite(p).all(-1)
            if good.any(): ys.append(yy[:,-1][good]);ps.append(p[good])
    if was:m.train()
    if not ys:return float('-inf')
    y,p=torch.cat(ys),torch.cat(ps)
    if stats is not None:
        scale=torch.as_tensor(stats[3],device=y.device); off=torch.as_tensor(stats[2],device=y.device);y=y*scale+off;p=p*scale+off
    return _r2(y,p)
def _trial_starts(bounds,ends):
    ans=[]
    for e in ends:
        hit=next((a for a,b in bounds if a<=e<b),None)
        if hit is None: raise ValueError(f'endpoint {e} outside trial bounds')
        ans.append(hit)
    return np.asarray(ans)
def _predict_endpoints(m,x,bounds,ends,device,context,batch=256):
    ends=np.asarray(ends,dtype=np.int64); starts=_trial_starts(bounds,ends); out=np.empty((len(ends),m.config.output_size),np.float32); was=m.training;m.eval()
    with torch.inference_mode():
        for lo in range(0,len(ends),batch):
            ee,aa=ends[lo:lo+batch],starts[lo:lo+batch]; win=np.zeros((len(ee),context,x.shape[1]),np.float32)
            for row,(end,a) in enumerate(zip(ee,aa)):
                left=max(int(a),int(end)-context+1);win[row,context-(end-left+1):]=x[left:end+1]
            out[lo:lo+len(ee)]=_forward(m,torch.from_numpy(win).to(device))[:,-1].float().cpu().numpy()
    if was:m.train()
    return out
def _target_predictions(m,target,stats,device,context,query_bounds=None):
    x,y=_norm(target,stats); ss,qq=split_support_query(target,support_trials=33,query_start_trials=33,sequence=50)
    support=np.unique((ss[:,None]+np.arange(50)).ravel()); legacy=np.unique(qq+49)
    usable=[(a,b) for a,b in target.trial_bounds if b-a>=50]
    if query_bounds is not None:
        usable=list(query_bounds); legacy=np.concatenate([np.arange(a+49,b,dtype=np.int64) for a,b in usable])
    else: usable=usable[33:]
    pieces=[np.arange(a,b,dtype=np.int64) for a,b in usable]
    all_valid=np.concatenate(pieces) if pieces else np.empty(0,dtype=np.int64)
    support=support[target.eval_mask[support]];legacy=legacy[target.eval_mask[legacy]];all_valid=all_valid[target.eval_mask[all_valid]]
    needed=np.unique(np.r_[support,legacy,all_valid]); pred=_predict_endpoints(m,x,target.trial_bounds,needed,device,context); at={int(v):i for i,v in enumerate(needed)}
    def take(indices): return pred[[at[int(i)] for i in indices]]
    return {'support_indices':support,'support_prediction':take(support),'support_truth_normalized':y[support],
            'legacy_indices':legacy,'legacy_prediction':take(legacy),'legacy_truth_physical':y[legacy]*stats[3]+stats[2],
            'all_valid_indices':all_valid,'all_valid_prediction':take(all_valid),'all_valid_truth_physical':y[all_valid]*stats[3]+stats[2]}
def _score_target(cache,stats,adapt):
    support_prediction=cache['support_prediction']
    def calibrated(prediction):
        if adapt=='zero': return prediction
        mapper=fit_ridge(torch.from_numpy(support_prediction).double(),torch.from_numpy(cache['support_truth_normalized']-support_prediction).double())
        return prediction+mapper(torch.from_numpy(prediction).double()).float().numpy()
    def score(name):
        truth=cache[f'{name}_truth_physical']; prediction=calibrated(cache[f'{name}_prediction'])*stats[3]+stats[2]
        return {'r2_variance_weighted':_r2(torch.from_numpy(truth),torch.from_numpy(prediction)),'n_query_bins':int(len(truth)),
                'indices':cache[f'{name}_indices'],'truth_physical':truth,'prediction_physical':prediction}
    legacy,all_valid=score('legacy'),score('all_valid')
    return {'r2_variance_weighted':legacy['r2_variance_weighted'],'n_support_bins':int(len(cache['support_indices'])),
            'n_query_bins':legacy['n_query_bins'],'all_valid_query_trial_r2_variance_weighted':all_valid['r2_variance_weighted'],
            'all_valid_query_trial_n_bins':all_valid['n_query_bins'],'legacy':legacy,'all_valid':all_valid}
def _hashes():
    root=Path(__file__).resolve().parent; files=[Path(__file__),root/'modern_models.py',root/'models.py',root/'data.py',root/'calibration.py',root/'mamba3_official.py',root/'mamba2_official.py']
    return {x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files if x.exists()}
def _official_pin(kind):
    if kind not in {'mamba2_official','mamba3_official'}: return None
    module=__import__(f'ssm_decode.{kind}',fromlist=['COMMIT','OFFICIAL']); COMMIT,OFFICIAL=module.COMMIT,module.OFFICIAL
    actual=None
    try: actual=subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()
    except Exception: pass
    triton=None
    try:
        import triton as _triton; triton={'version':_triton.__version__,'file':getattr(_triton,'__file__',None)}
    except Exception: pass
    return {'expected_official_commit':COMMIT,'actual_official_commit':actual,'official_path':str(OFFICIAL),'triton':triton,'pythonpath':os.environ.get('PYTHONPATH')}

def run(a):
    torch.set_num_threads(1);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);_seed(a.seed);dev=torch.device(a.device)
    if dev.type=='cuda': torch.cuda.set_device(dev)
    code_hashes=_hashes();(out/'code_hashes_start.json').write_text(json.dumps(code_hashes,indent=2)+'\n')
    plan=source_target_plan(a.task,root=Path(a.data_root))
    source=[load_recording(a.task,'held_in',s,root=Path(a.data_root)) for s in plan['source_held_in_sessions']];target=load_recording(a.task,'held_in',plan['cross_session_local_dev']['target_session'],root=Path(a.data_root));split=[_trial_split(r) for r in source]
    query_bounds=None;query_cutoff_actual=33
    if a.mode=='within_session_prefix':
        usable=[z for z in target.trial_bounds if z[1]-z[0]>=50][:33];cut=max(1,int(.8*len(usable)));source=[target];split=[(usable[:cut],usable[cut:])];protocol='within-session target-prefix supervised diagnostic; fixed query tail excluded from fitting and selection'
    elif a.mode=='within_session_split':
        train_bounds,val_bounds,query_bounds=_split_60_20_20(target.trial_bounds);source=[target];split=[(train_bounds,val_bounds)];query_cutoff_actual=len(train_bounds)+len(val_bounds)
        protocol='within-session split diagnostic (60% train, 20% validation, 20% query); sample-size/learning-capacity diagnostic, not comparable to cross-session query cohort'
    else: protocol='cross-session source-only pretraining and source-only validation'
    warm_normalizer=None
    if a.warmstart:
        stats,warm_normalizer=_warm_stats(a.warmstart)
    else: stats=_stats(source,[z[0] for z in split])
    np.savez(out/'normalizer.npz',x_mean=stats[0],x_std=stats[1],y_mean=stats[2],y_std=stats[3])
    train=_windows(source,[z[0] for z in split],stats,dev);valid=_windows(source,[z[1] for z in split],stats,dev);eligible=_eligible(train); selection,selection_hash=_validation_selection(valid,a.context,a.max_val_endpoints)
    m=_model(a,source[0].neural.shape[1],source[0].behavior.shape[1]).to(dev)
    if a.warmstart:m.load_state_dict(torch.load(a.warmstart,map_location=dev,weights_only=False)['state_dict'])
    if a.adapt_parameters=='io': _freeze_io(m)
    trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=a.weight_decay);best=-float('inf');beststep=0;started=time.time()
    with (out/'train_log.jsonl').open('w') as f:
        for step in range(1,a.steps+1):
            m.train();xx,yy,mm=_sample_batch(train,eligible,a.context,a.batch_size);opt.zero_grad(set_to_none=True);p=_forward(m,xx)[:,a.context//2:];yt=yy[:,a.context//2:]
            if not torch.isfinite(p).all(): raise FloatingPointError('non-finite training prediction')
            mask=mm[:,a.context//2:]&torch.isfinite(yt).all(-1)
            if not mask.any():raise ValueError('sampled batch has no valid post-burnin bins')
            loss=(p[mask]-yt[mask]).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if step==1 or step%a.val_interval==0 or step==a.steps:
                vr=_val(m,valid,a.context,selection,a.eval_batch_size,stats);event={'step':step,'train_loss':float(loss.detach()),'source_val_physical_variance_weighted_r2':vr,'elapsed_seconds':time.time()-started};f.write(json.dumps(event)+'\n');f.flush()
                if vr>best:best,beststep=vr,step;torch.save({'state_dict':m.state_dict(),'args':vars(a),'best_source_val_r2':best,'best_step':beststep},out/'best.pt')
    if dev.type=='cuda': torch.cuda.synchronize(dev)
    train_seconds=time.time()-started;torch.save({'state_dict':m.state_dict(),'args':vars(a)},out/'last.pt');m.load_state_dict(torch.load(out/'best.pt',map_location=dev,weights_only=False)['state_dict']);cache=_target_predictions(m,target,stats,dev,a.context,query_bounds);zero=_score_target(cache,stats,'zero');ridge=_score_target(cache,stats,'ridge')
    np.savez(out/'target_query_predictions.npz',legacy_query_indices=cache['legacy_indices'],legacy_truth_physical=cache['legacy_truth_physical'],zero_legacy_prediction_physical=zero['legacy']['prediction_physical'],ridge_legacy_prediction_physical=ridge['legacy']['prediction_physical'],all_valid_query_indices=cache['all_valid_indices'],all_valid_truth_physical=cache['all_valid_truth_physical'],zero_all_valid_prediction_physical=zero['all_valid']['prediction_physical'],ridge_all_valid_prediction_physical=ridge['all_valid']['prediction_physical'])
    # Arrays belong in the NPZ, never in the reviewable JSON metrics.
    for result in (zero,ridge):
        for group in ('legacy','all_valid'):
            result[group].pop('indices');result[group].pop('truth_physical');result[group].pop('prediction_physical')
    if dev.type=='cuda': torch.cuda.synchronize(dev)
    final_seconds=time.time()-started-train_seconds; cuda_name=torch.cuda.get_device_name(dev) if dev.type=='cuda' and torch.cuda.is_available() else None
    scope='source train trials only' if a.mode=='cross_session' else 'target train cohort only'
    metrics={'status':'completed','protocol':protocol,'plan':plan,'args':vars(a),'normalizer_scope':scope,'warmstart_normalizer':warm_normalizer,'best_source_val_r2':best,'best_step':beststep,'final':{'zero':zero,'ridge':ridge},'params':sum(p.numel() for p in m.parameters()),'trainable_params':trainable,'train_seconds':train_seconds,'final_evaluation_seconds':final_seconds,'steps_per_second':a.steps/train_seconds,'device':str(dev),'cuda_name':cuda_name,'torch_version':torch.__version__,'code_hashes':code_hashes,'held_out_accessed':False};(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest={'plan':plan,'args':vars(a),'protocol':protocol,'code_hashes':code_hashes,'official_runtime':_official_pin(a.kind),'warmstart_path':a.warmstart,'warmstart_normalizer':warm_normalizer,'adapt_parameters':a.adapt_parameters,'trainable_params':trainable,'query_cutoff_trials_actual':query_cutoff_actual,'query_start_usable_trial_index':query_cutoff_actual,'query_trial_bounds':None if query_bounds is None else [list(x) for x in query_bounds],'source_train_trial_bounds':[[list(x) for x in z[0]] for z in split],'source_val_trial_bounds':[[list(x) for x in z[1]] for z in split],'train_eligible_endpoint_count':int(sum(map(len,eligible))),'source_val_fixed_endpoint_count':len(selection),'source_val_fixed_indices':selection,'source_val_fixed_indices_sha256':selection_hash,'source_val_max_endpoints':a.max_val_endpoints,'burnin_last_half_loss':True,'eval_mask_applied_to_train_and_validation':True,'target_query_used_for_checkpoint_selection':False,'target_channel_identity':'unknown; no channel identity alignment assumed'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return out
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['m1','m2'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--kind',choices=['osc','gru','s4d','mamba2','mamba2_official','mamba3_official'],default='s4d');p.add_argument('--mode',choices=['cross_session','within_session_prefix','within_session_split'],default='cross_session');p.add_argument('--width',type=int,default=128);p.add_argument('--layers',type=int,default=2);p.add_argument('--state-size',type=int,default=16);p.add_argument('--dropout',type=float,default=.1);p.add_argument('--context',choices=[50,128,256],type=int,default=128);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--eval-batch-size',type=int,default=256);p.add_argument('--steps',type=int,default=1000);p.add_argument('--val-interval',type=int,default=200);p.add_argument('--max-val-endpoints',type=int,default=1024);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--seed',type=int,default=0);p.add_argument('--warmstart');p.add_argument('--adapt-parameters',choices=['all','io'],default='all');p.add_argument('--data-root',default=str(DEFAULT_ROOT));a=p.parse_args(argv);print(run(a))
if __name__=='__main__':main()
