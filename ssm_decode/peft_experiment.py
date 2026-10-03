"""Target-prefix PEFT diagnostic for a frozen source-trained Mamba-3 checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from . import debug_experiment as d
from .data import DEFAULT_ROOT, load_recording, source_target_plan
from .mamba3_official import build_mamba3_official
from .peft import configure_peft
from .sparse_state_tuning import begin_dense_selection, selection_from_warmup, install_sparse_state_tuning, assert_stage_ready
from .mamba3_research_adapters import install_research_adapters

RESEARCH_METHODS = {'state_offset', 'memba_causal'}
SPARSE_METHOD = 'sparse_sdt_m3'

def _configure_method(model, method, rank, alpha, selections=None):
    if method == SPARSE_METHOD:
        if selections is None:
            raise ValueError('sparse replay requires selections')
        return install_sparse_state_tuning(model, selections, rank)
    if method in RESEARCH_METHODS:
        return install_research_adapters(model, method, rank)
    return configure_peft(model, method, rank=rank, alpha=alpha)

def load_saved_model(path, device='cpu'):
    """Rebuild a PEFT checkpoint without consulting target data or labels."""
    payload=torch.load(path,map_location=device,weights_only=False)
    meta=payload['peft_replay']
    model=build_mamba3_official(meta['input_size'],meta['output_size'],width=meta['cfg']['width'],layers=meta['cfg']['layers'],state_size=meta['cfg']['state_size'],dropout=meta['cfg']['dropout']).to(device)
    _configure_method(model,meta['method'],meta['rank'],meta['alpha'],meta.get('sparse_selections'))
    model.load_state_dict(payload['state_dict']);return model.eval(),payload

def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def _seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def _split(target):
    u=[b for b in target.trial_bounds if b[1]-b[0]>=50]
    if len(u)<=33: raise ValueError('requires >33 usable trials')
    return u[:26],u[26:33],u[:33]
def _receipt_json(receipt): return receipt if isinstance(receipt,dict) else getattr(receipt,'__dict__',{'receipt':str(receipt)})
def _groups(model,method,base_lr):
    if method!='full': return [{'params':[p for p in model.parameters() if p.requires_grad],'lr':base_lr}]
    blocks=list(getattr(model,'blocks',[])); groups=[]
    for i,p in enumerate(model.parameters()): p.requires_grad=False
    # Root IO only: do not accidentally include blocks.*.ssm.in_proj/out_proj.
    io=[p for n,p in model.named_parameters() if n.split('.',1)[0] in {'in_proj','out_proj','final_norm'}]
    for p in io:p.requires_grad=True
    groups.append({'params':io,'lr':base_lr})
    return groups
def _unfreeze(model,opt,step,total,base_lr):
    # 20/20/60 schedule; newly enabled parameters are appended to preserved AdamW state.
    frac=step/max(1,total); blocks=list(getattr(model,'blocks',[])); wanted=[]
    if frac>=.2 and blocks:wanted+=list(blocks[-1].parameters())
    if frac>=.4:wanted+=[p for b in blocks[:-1] for p in b.parameters()]
    add=[p for p in wanted if not p.requires_grad]
    for p in add:p.requires_grad=True
    for no_decay in (False, True):
        selected = [p for p in add if bool(getattr(p, '_no_weight_decay', False)) == no_decay]
        if selected:
            opt.add_param_group({'params':selected,'lr':base_lr*.5,
                                'weight_decay':0. if no_decay else opt.defaults['weight_decay']})

def _optimizer_receipt(model, opt, step):
    names = {id(p): name for name, p in model.named_parameters()}
    return {'step': step, 'groups': [
        {'names': [names[id(p)] for p in group['params']],
         'count': sum(p.numel() for p in group['params']),
         'lr': group['lr'], 'weight_decay': group['weight_decay']}
        for group in opt.param_groups]}

def _dt_probe(model, probe):
    records, handles = {}, []
    for i, block in enumerate(model.blocks):
        core = getattr(block, 'ssm', None)
        if core is None or not hasattr(core, 'dt_bias') or not hasattr(core, 'd_inner'):
            continue
        def hook(module, inputs, projected, core=core, index=i):
            start = 2*core.d_inner + 2*core.d_state*core.num_bc_heads*core.mimo_rank
            raw_dt = projected[..., start:start+core.nheads]
            raw_a = projected[..., start+core.nheads:start+2*core.nheads].float()
            dt = torch.nn.functional.softplus(raw_dt.float()+core.dt_bias.float())
            decay = -(raw_a.clamp_min(0)+torch.reciprocal(1-raw_a.clamp_max(0))).clamp_min(core.A_floor)*dt
            if not torch.isfinite(dt).all() or not (dt > 0).all() or not torch.isfinite(decay).all() or not (decay < 0).all():
                raise FloatingPointError('invalid Mamba-3 DT/ADT on prefix probe')
            records[str(index)] = {'dt_min':float(dt.min()),'dt_max':float(dt.max()),
                                   'dt_mean':float(dt.mean()),'adt_min':float(decay.min()),
                                   'adt_max':float(decay.max())}
        handles.append(core.in_proj.register_forward_hook(hook))
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(probe)
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    return records
def run(a):
    if a.policy != 'trial_causal_fixed_window':
        raise ValueError('only trial_causal_fixed_window is supported')
    if a.steps < 0 or a.val_interval < 1:
        raise ValueError('steps >= 0 and val_interval >= 1 required')
    torch.set_num_threads(1)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);_seed(a.seed);dev=torch.device(a.device)
    if dev.type=='cuda':
        torch.cuda.set_device(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    hashes={x:_hash(Path(__file__).parent/x) for x in ('peft_experiment.py','peft.py','debug_experiment.py','calibration.py','sparse_state_tuning.py','mamba3_research_adapters.py','data.py','mamba3_official.py') if (Path(__file__).parent/x).exists()};(out/'code_hashes_start.json').write_text(json.dumps(hashes,indent=2)+'\n')
    ck=torch.load(a.pretrained,map_location='cpu',weights_only=False);cfg=ck['args'];normal=Path(a.pretrained).parent/'normalizer.npz';z=np.load(normal);stats=tuple(z[k].astype('float32') for k in ('x_mean','x_std','y_mean','y_std'))
    plan=source_target_plan(a.task,root=Path(a.data_root));target=load_recording(a.task,'held_in',plan['cross_session_local_dev']['target_session'],root=Path(a.data_root));train_b,val_b,support_b=_split(target)
    prefix_labels = np.full_like(target.behavior, np.nan)
    for start, end in support_b:
        prefix_labels[start:end] = target.behavior[start:end]
    fit_target = replace(target, behavior=prefix_labels)
    model=build_mamba3_official(target.neural.shape[1],target.behavior.shape[1],width=cfg['width'],layers=cfg['layers'],state_size=cfg['state_size'],dropout=cfg['dropout']).to(dev);model.load_state_dict(ck['state_dict'])
    base_parameters = sum(p.numel() for p in model.parameters())
    run_started = time.perf_counter()
    sparse_receipt=None
    if a.method == SPARSE_METHOD:
        begin_dense_selection(model)
        assert_stage_ready(model, 'dense_warmup')
        warm_opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
        warm_log=[]
        warm_started = time.perf_counter()
        windows=d._windows([fit_target],[train_b],stats,dev);eligible=d._eligible(windows)
        for step in range(1, a.sparse_warmup_steps + 1):
            model.train(); xx,yy,mm=d._sample_batch(windows,eligible,a.context,a.batch_size); pred=d._forward(model,xx)[:,a.context//2:]; truth=yy[:,a.context//2:]; mask=mm[:,a.context//2:] & torch.isfinite(truth).all(-1)
            if not mask.any() or not torch.isfinite(pred).all(): raise FloatingPointError('invalid sparse warmup batch')
            warm_opt.zero_grad(set_to_none=True); loss=(pred[mask]-truth[mask]).square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); warm_opt.step(); warm_log.append({'step':step,'loss':float(loss.detach())})
        sparse_receipt=selection_from_warmup(model,a.keep_states)
        (out/'sparse_selection.json').write_text(json.dumps({'warmup_receipt':getattr(model,'_sparse_state_warmup',{}),'selection_receipt':sparse_receipt,'warmup_log':warm_log,'warmup_seconds':time.perf_counter()-warm_started},indent=2)+'\n')
        _seed(a.seed)
        receipt=_configure_method(model,a.method,a.rank,a.alpha,sparse_receipt['selections'])
    else:
        receipt=_configure_method(model,a.method,a.rank,a.alpha)
    receipt=_receipt_json(receipt)
    replay={'cfg':cfg,'input_size':target.neural.shape[1],'output_size':target.behavior.shape[1],'method':a.method,'rank':a.rank,'alpha':a.alpha,'sparse_selections': None if sparse_receipt is None else sparse_receipt['selections']}
    def save_checkpoint(path,step): torch.save({'state_dict':model.state_dict(),'best_step':step,'peft_replay':replay,'args':vars(a),'receipt':receipt,'normalizer':dict(zip(('x_mean','x_std','y_mean','y_std'),stats))},path)
    windows=d._windows([fit_target],[train_b],stats,dev);valid=d._windows([fit_target],[val_b],stats,dev);eligible=d._eligible(windows);sel,selhash=d._validation_selection(valid,a.context,10**9)
    probe, _, _ = d._right_aligned(valid, sel[:min(16,len(sel))], a.context)
    _seed(a.seed)
    initial_hashes={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
    trainlog=[]
    frozen_audits=[]
    optimizer_stages=[]
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if a.method=='none':
        best= d._val(model,valid,a.context,sel,a.eval_batch_size,stats);beststep=0;save_checkpoint(out/'best.pt',0)
    else:
        groups=_groups(model,a.method,a.lr)
        dt_params={n:p for n,p in model.named_parameters() if n.endswith('dt_bias')}
        dt_initial={n:p.detach().clone() for n,p in dt_params.items()}
        if a.method == 'bc_dt':
            dt_ids={id(p) for p in dt_params.values()}
            regular=[p for p in model.parameters() if p.requires_grad and id(p) not in dt_ids]
            groups=[{'params':regular,'lr':a.lr,'weight_decay':a.weight_decay},{'params':list(dt_params.values()),'lr':a.lr*.1,'weight_decay':0.}]
        opt=torch.optim.AdamW(groups,weight_decay=a.weight_decay);best=-float('inf');beststep=0
        initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer_stages.append(_optimizer_receipt(model,opt,0))
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        stage_frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        # Step zero is a valid checkpoint-selection candidate.
        best=d._val(model,valid,a.context,sel,a.eval_batch_size,stats);beststep=0;save_checkpoint(out/'best.pt',0)
        trainlog.append({'step':0,'prefix_val_r2':best,'dt_probe':_dt_probe(model,probe)})
        for step in range(1,a.steps+1):
            if a.method=='full' and step in {max(1,int(np.ceil(.2*a.steps))),max(1,int(np.ceil(.4*a.steps)))}:
                changed=[n for n,p in model.named_parameters() if n in stage_frozen and not torch.equal(stage_frozen[n],p.detach().cpu())]
                if changed: raise AssertionError(f'frozen stage parameters changed: {changed[:3]}')
                before_count=sum(p.requires_grad for p in model.parameters())
                _unfreeze(model,opt,step,a.steps,a.lr)
                after_count=sum(p.requires_grad for p in model.parameters())
                if after_count != before_count:
                    frozen_audits.append({'step':step,'frozen_check':'passed','trainable_tensor_count':after_count})
                    optimizer_stages.append(_optimizer_receipt(model,opt,step))
                    stage_frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
            model.train();xx,yy,mm=d._sample_batch(windows,eligible,a.context,a.batch_size);p=d._forward(model,xx)[:,a.context//2:];yt=yy[:,a.context//2:];mask=mm[:,a.context//2:]&torch.isfinite(yt).all(-1)
            if not torch.isfinite(p).all() or not mask.any():raise FloatingPointError('invalid PEFT batch')
            opt.zero_grad(set_to_none=True);loss=(p[mask]-yt[mask]).square().mean();loss.backward()
            gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            opt.step()
            if a.method == 'bc_dt':
                for n,initial in dt_initial.items():
                    dict(model.named_parameters())[n].data.clamp_(initial.data - 1., initial.data + 1.)
            trainlog.append({'step':step,'loss':float(loss.detach()),'grad_norm':gradnorm})
            if step%a.val_interval==0 or step==a.steps:
                score=d._val(model,valid,a.context,sel,a.eval_batch_size,stats)
                trainlog[-1]['prefix_val_r2']=score
                trainlog[-1]['dt_probe']=_dt_probe(model,probe)
                if score>best:best,beststep=score,step;save_checkpoint(out/'best.pt',step)
        frozen_changed=[n for n,p in model.named_parameters() if n in before and not torch.equal(before[n],p.detach().cpu())]
        if frozen_changed and a.method!='full':raise AssertionError(f'frozen parameters changed: {frozen_changed[:3]}')
        stage_changed=[n for n,p in model.named_parameters() if n in stage_frozen and not torch.equal(stage_frozen[n],p.detach().cpu())]
        if stage_changed:raise AssertionError(f'last stage frozen parameters changed: {stage_changed[:3]}')
        frozen_audits.append({'step':a.steps,'frozen_check':'passed','checked_tensors':len(stage_frozen)})
    fit_seconds = time.perf_counter()-run_started
    save_checkpoint(out/'last.pt',0 if a.method=='none' else a.steps)
    model.load_state_dict(torch.load(out/'best.pt',map_location=dev,weights_only=False)['state_dict']);cache=d._target_predictions(model,target,stats,dev,a.context);zero=d._score_target(cache,stats,'zero');ridge=d._score_target(cache,stats,'ridge')
    shutil.copy2(normal,out/'normalizer.npz')
    (out/'train_log.json').write_text(json.dumps(trainlog,indent=2)+'\n');(out/'replay_args.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    np.savez(out/'predictions.npz',legacy_indices=cache['legacy_indices'],allvalid_indices=cache['all_valid_indices'],truth_legacy=cache['legacy_truth_physical'],truth_allvalid=cache['all_valid_truth_physical'],support_indices=cache['support_indices'],support_truth_normalized=cache['support_truth_normalized'],support_prediction=cache['support_prediction'],zero_legacy=zero['legacy']['prediction_physical'],ridge_legacy=ridge['legacy']['prediction_physical'],zero_allvalid=zero['all_valid']['prediction_physical'],ridge_allvalid=ridge['all_valid']['prediction_physical'])
    for r in (zero,ridge):
        for k in ('legacy','all_valid'):
            for q in ('indices','truth_physical','prediction_physical'):r[k].pop(q)
    final_hashes={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
    manifest={'pretrained':str(a.pretrained),'pretrained_sha256':_hash(a.pretrained),'statsfile':str(normal),'statsfile_sha256':_hash(normal),'code_hashes':hashes,'receipt':receipt,'policy':a.policy,'train_bounds':train_b,'val_bounds':val_b,'support_bounds':support_b,'prefix_train_count':26,'prefix_val_count':7,'query_cutoff_trials':33,'prefix_val_indices_sha256':selhash,'datamanifest':plan,'seed':a.seed,'method':a.method,'rank':a.rank,'alpha':a.alpha,'sparse_selection':sparse_receipt,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'frozen_audits':frozen_audits,'trainable_parameters':[n for n,p in model.named_parameters() if p.requires_grad],'initial_parameter_hashes':initial_hashes,'final_parameter_hashes':final_hashes,'changed_parameter_names':[n for n in initial_hashes if initial_hashes[n]!=final_hashes[n]]}
    source_start = Path(a.pretrained).parent/'code_hashes_start.json'
    manifest.update(optimizer_stages=optimizer_stages,selected_dt_probe=_dt_probe(model,probe),
        source_start_code_hashes=json.loads(source_start.read_text()) if source_start.exists() else None,
        official=d._official_pin('mamba3_official'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cohort_index_hashes={key:hashlib.sha256(cache[key].tobytes()).hexdigest() for key in ('support_indices','legacy_indices','all_valid_indices')},
        dt_bias_constraints={'initial_relative_bounds':[-1.,1.],'lr_multiplier':.1,'weight_decay':0.} if a.method=='bc_dt' else None)
    for name, expected in hashes.items():
        if _hash(Path(__file__).parent/name) != expected:raise RuntimeError('implementation changed during experiment')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'metrics.json').write_text(json.dumps({'status':'completed','method':a.method,'seed':a.seed,'best_prefix_val_r2':best,'best_step':beststep,'final':{'zero':zero,'ridge':ridge},'base_parameters':base_parameters,'total_parameters':sum(p.numel() for p in model.parameters()),'initial_trainable_params':initial_trainable,'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'fit_seconds':fit_seconds,'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else None},indent=2)+'\n');return out
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['m1','m2'],required=True);p.add_argument('--pretrained',required=True);p.add_argument('--output',required=True);p.add_argument('--method',choices=['none','io','lora','bc_lora','bc_dt','full',SPARSE_METHOD,*sorted(RESEARCH_METHODS)],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=500);p.add_argument('--context',type=int,default=128);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--eval-batch-size',type=int,default=256);p.add_argument('--val-interval',type=int,default=50);p.add_argument('--rank',type=int,default=4);p.add_argument('--alpha',type=float,default=4);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--sparse-warmup-steps',type=int,default=100);p.add_argument('--keep-states',type=int,default=8);p.add_argument('--seed',type=int,default=0);p.add_argument('--policy',default='trial_causal_fixed_window');p.add_argument('--data-root',default=str(DEFAULT_ROOT));a=p.parse_args(argv);print(run(a))
if __name__=='__main__':main()
