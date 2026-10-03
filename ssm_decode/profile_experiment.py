"""Non-official support-profile SSM pilot, separate from the fixed-channel run."""
from __future__ import annotations
import argparse, json, random, time, hashlib, socket
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .calibration import fit_ridge, fit_rls
from .data import DEFAULT_ROOT, load_recording, source_normalizer, source_target_plan, split_support_query
from .experiment import _predict_trials, evaluate_target
from .models import ModelConfig, build_model
from .profile_models import LearnedProfileFrontend

def profile_from_support(x: np.ndarray,y: np.ndarray) -> np.ndarray:
    """Per-unit [mean,std,constant,covariances] profile from support only."""
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x): raise ValueError("support x/y geometry")
    mean=x.mean(0); std=x.std(0); std[std<1e-6]=1.
    cov=((x-mean).T@(y-y.mean(0)))/len(x)
    return np.concatenate((mean[:,None],std[:,None],np.ones((x.shape[1],1),np.float32),cov),axis=1).astype(np.float32)

def _support_rows(record,stats,count,sequence):
    starts,_=split_support_query(record,support_trials=count,sequence=sequence)
    idx=np.unique(starts[:,None]+np.arange(sequence)[None,:]).reshape(-1)
    x=((record.neural-stats['x_mean'])/stats['x_std']).astype(np.float32); y=((record.behavior-stats['y_mean'])/stats['y_std']).astype(np.float32)
    good=record.eval_mask[idx]&np.isfinite(y[idx]).all(1)
    return x[idx][good],y[idx][good]

def _install(model,profile):
    front=LearnedProfileFrontend(profile.shape[1],model.config.width).to(next(model.parameters()).device)
    model.frontend=front; front.set_profile(torch.from_numpy(profile).to(next(model.parameters()).device)); return front

def _tail_segments(record,stats,count,sequence):
    _,starts=split_support_query(record,support_trials=count,query_start_trials=count,sequence=sequence)
    x=((record.neural-stats['x_mean'])/stats['x_std']).astype(np.float32); y=((record.behavior-stats['y_mean'])/stats['y_std']).astype(np.float32)
    ans=[]
    for s in starts[::sequence]:
        valid=record.eval_mask[s:s+sequence]&np.isfinite(y[s:s+sequence]).all(1)
        if valid.any(): ans.append((x[s:s+sequence],y[s:s+sequence],valid))
    return ans

def _support_normalized(record,stats,count,sequence):
    """Optional session-local neural normalization fitted from support only."""
    x,y=_support_rows(record,stats,count,sequence); mean=x.mean(0);std=x.std(0);std[std<1e-6]=1.
    base=((record.neural-stats['x_mean'])/stats['x_std']).astype(np.float32); adjusted=(base-mean)/std
    from dataclasses import replace
    return replace(record,neural=(adjusted*stats['x_std']+stats['x_mean']).astype(np.float32))

def run(args):
    torch.set_num_threads(1); out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False); device=torch.device(args.device)
    started=time.time();plan=source_target_plan(args.task,root=Path(args.data_root)); sources=[load_recording(args.task,'held_in',s,root=Path(args.data_root)) for s in plan['source_held_in_sessions']]
    target=load_recording(args.task,'held_in',plan['cross_session_local_dev']['target_session'],root=Path(args.data_root))
    stats=source_normalizer(sources); np.savez_compressed(out/'source_normalizer.npz',**stats)
    feasible=[k for k in args.support_trials if not _invalid(target,k,args.sequence)]; count=max(feasible) if feasible else None
    if count is None: raise ValueError('local cross-session target has no requested support/query split')
    source_counts={r.session:max(k for k in args.support_trials if not _invalid(r,k,args.sequence)) for r in sources}
    if args.support_normalize_neural:
      sources=[_support_normalized(r,stats,source_counts[r.session],args.sequence) for r in sources];target=_support_normalized(target,stats,count,args.sequence)
    target_profile=profile_from_support(*_support_rows(target,stats,count,args.sequence)); rows=[]
    code={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('profile_experiment.py','profile_models.py','models.py','data.py','calibration.py')}
    manifest={'scope':'non-official local held-in cross-session profile pilot','plan':plan,'args':vars(args),'support_trials':count,'source_profile_budgets':source_counts,'sequence':args.sequence,'profile_dim':int(target_profile.shape[1]),'model_type':'learned_profile_ssm','profile_encoder':{'layers':[int(target_profile.shape[1]),32,args.width],'activation':'ReLU','fold_scale':'1/sqrt(valid_units)'},'support_normalize_neural':bool(args.support_normalize_neural),'code_sha256':code,'device':str(device),'host':socket.gethostname(),'held_out_accessed':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for kind in args.kinds:
      for seed in args.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=build_model(ModelConfig(sources[0].neural.shape[1],sources[0].behavior.shape[1],width=args.width,kind=kind)).to(device)
        # The learned profile encoder persists across all source sessions; only
        # its materialized session matrix changes at calibration boundaries.
        front=LearnedProfileFrontend(target_profile.shape[1],args.width).to(device);model.frontend=front
        # Each source's profile is materialized before its sampled tail segment.
        source_groups=[]
        for r in sources:
          source_count=source_counts[r.session];p=profile_from_support(*_support_rows(r,stats,source_count,args.sequence));items=_tail_segments(r,stats,source_count,args.sequence)
          if items: source_groups.append((p,items))
        if not source_groups: raise ValueError('no source tail segments')
        weights=np.asarray([len(items) for _,items in source_groups],np.float64);weights/=weights.sum()
        opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate); curve=[]
        for step in range(args.steps):
          group=int(np.random.choice(len(source_groups),p=weights));p,items=source_groups[group];take=np.random.randint(len(items),size=args.batch_size);picked=[items[i] for i in take];front.set_profile(torch.from_numpy(p).to(device))
          x=np.stack([z[0] for z in picked]);y=np.stack([z[1] for z in picked]);m=np.stack([z[2] for z in picked]);tx=torch.from_numpy(x).to(device);ty=torch.from_numpy(y).to(device);tm=torch.from_numpy(m)[:,:,None].to(device)
          opt.zero_grad(); pred=model(tx);loss=((pred-ty).square()*tm).sum()/tm.sum().clamp_min(1)/pred.shape[-1];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
          if step==0 or (step+1)%max(1,args.steps//20)==0:curve.append({'step':step+1,'mse':float(loss.detach().cpu())})
        front.set_profile(torch.from_numpy(target_profile).to(device));pred=_predict_trials(model,target,stats,device)
        tag=f'{kind}_s{seed}'; weights=front.export_weights().numpy();np.savez_compressed(out/f'{tag}.weights.npz',weights=weights,profile=target_profile)
        torch.save({'state_dict':model.state_dict(),'model_config':asdict(model.config),'profile_dim':target_profile.shape[1],'profile_encoder':manifest['profile_encoder'],'code_sha256':code},out/f'{tag}.pt');(out/f'{tag}.curve.json').write_text(json.dumps(curve,indent=2)+'\n')
        for adaptation in ('zero','ridge','rls'):
          row=evaluate_target(target,stats,pred,support_trials=count,query_start_trials=count,sequence=args.sequence,adaptation=adaptation);arrays={k:row.pop(k) for k in ('predictions','truth','query_indices')};np.savez_compressed(out/f'{tag}.{adaptation}.npz',**arrays);row.update(kind=kind,seed=seed,condition='profile');rows.append(row)
        # Deterministic 20% channel drop sensitivity; profile and neural mask are jointly aligned.
        keep=np.ones(target.neural.shape[1],bool);keep[np.random.RandomState(seed).choice(len(keep),size=max(1,len(keep)//5),replace=False)]=False
        front.set_profile(torch.from_numpy(target_profile).to(device),torch.from_numpy(keep).to(device));drop_pred=_predict_trials(model,target,stats,device);row=evaluate_target(target,stats,drop_pred,support_trials=count,query_start_trials=count,sequence=args.sequence,adaptation='zero');row.pop('predictions');row.pop('truth');row.pop('query_indices');row.update(kind=kind,seed=seed,condition='profile_drop20');rows.append(row)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,'seed':seed,'status':'completed','elapsed_seconds':time.time()-started})+'\n')
    (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'run.json').write_text(json.dumps({'status':'completed','elapsed_seconds':time.time()-started,'args':vars(args),'device':str(device),'held_out_accessed':False},indent=2)+'\n');return out

def _invalid(record,count,sequence):
    try:split_support_query(record,support_trials=count,sequence=sequence);return False
    except ValueError:return True
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--task',choices=('m1','m2'),required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);p.add_argument('--data-root',default=str(DEFAULT_ROOT));p.add_argument('--steps',type=int,default=600);p.add_argument('--width',type=int,default=64);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--sequence',type=int,default=50);p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--support-normalize-neural',action='store_true');p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--kinds',nargs='+',default=['diag','osc','bank']);p.add_argument('--support-trials',nargs='+',type=int,default=[5,10,33]);print(run(p.parse_args(argv)))
if __name__=='__main__':main()
