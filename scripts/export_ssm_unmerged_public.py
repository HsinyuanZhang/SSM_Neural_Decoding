"""Export fixed M2 LoRA public banks without changing the selected weights.

The original merged export remains immutable. This exporter keeps the original
base-plus-delta projection arithmetic and validates every legal public prefix
against the official GPU kernel before publishing an unmerged CPU payload.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ssm_decode import official_payload as old
from ssm_decode.input_adaptation import configure_adaptation
from ssm_decode.official_calibration import load_public_calibration
from ssm_decode.session_normalization import fit_support_statistics


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def closure(root):
    return {str(p.relative_to(root)):old.sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '.triton_cache' not in str(p) and '__pycache__' not in str(p)}


def check_unmerged_fit(path,record,identity,stats,legacy_merged_published):
    receipt=read(path/'receipt.json')
    public=record.receipt
    if (public.get('raw_nwb_trial_ids_first_n')!=list(range(33)) or len(public.get('realtrialbounds',[]))!=33
            or public.get('calibration_trials')!=33 or public.get('publiccalibration_only') is not True
            or public.get('query_labels_used') is not False):
        raise RuntimeError('public record is not exactly the first 33 raw calibration trials')
    fold_score=receipt.get('fold_r2_abs_delta')
    if not isinstance(fold_score,(float,int)) or not np.isfinite(fold_score):
        raise RuntimeError('original merged score diagnostic is missing or nonfinite')
    if legacy_merged_published:
        if receipt.get('fold_allclose') is not True or fold_score>1e-3:
            raise RuntimeError('published legacy model violates the original merge gates')
        legacy_status='original_merged_export_passed'
    else:
        if receipt.get('fold_allclose') is not False or receipt.get('convergence_satisfied') is not True:
            raise RuntimeError('only converged pointwise-fold rejection can use the unmerged path')
        legacy_status='converged_original_pointwise_fold_rejected'

    if any(receipt.get(k)!=v for k,v in identity.items()):
        raise RuntimeError('public fit identity differs from the frozen source/export contract')
    if (receipt.get('public_calibration')!=record.receipt or receipt.get('query_labels_used') is not False
            or receipt.get('query_used_for_selection') is not False or receipt.get('convergence_satisfied') is not True
            or receipt.get('frozen_audit',{}).get('status')!='passed'):
        raise RuntimeError('unmerged fit violates raw-trial, selection, convergence or frozen-parameter contract')
    if old.sha(path/'train_log.json')!=receipt['train_log_sha256'] or old.sha(path/'normalizer.npz')!=receipt['normalizer_sha256']:
        raise RuntimeError('public fit diagnostic or normalizer hash changed')
    with np.load(path/'normalizer.npz') as z:
        if any(not np.array_equal(z[k],v) for k,v in zip(old.KEYS,stats)):
            raise RuntimeError('normalizer differs from exact legal prefix/source statistics')
    obj=torch.load(path/'unmerged_best.pt',map_location='cpu',weights_only=False)
    expected={'method':'lora','seed':0,'lr':.003,'steps':2000,'max_steps':8000}
    if any(obj.get('args',{}).get(k)!=v for k,v in expected.items()):
        raise RuntimeError('selected unmerged method, seed, LR or budget changed')
    scope=receipt.get('adaptation',{})
    if any(scope.get(k)!=v for k,v in {'method':'lora','rank':4,'alpha':4,'lora_scope':'all','trainable_count':37000,'train_input_bias':True}.items()):
        raise RuntimeError('unmerged LoRA scale or trainable scope drift')
    if obj.get('best_step')!=receipt['best_step'] or not receipt['best_step']<.8*receipt['final_budget']:
        raise RuntimeError('selected unmerged checkpoint violates the convergence budget')
    # A live official model and strict state loading authenticate paths/shapes;
    # exact hashes are checked below with the production parameter-hash helper.
    return receipt,obj,legacy_status


def prefix_parity(record,stats,source,state,receipt,device):
    from ssm_decode.mamba3_cpu_lora import CPUDecoder
    gpu=old._base_model(source['args'],96,2,device)
    configure_adaptation(gpu,'lora',rank=4,alpha=4,lora_scope='all',train_input_bias=True)
    gpu.load_state_dict(state,strict=True);gpu.eval()
    if old._parameter_hashes(gpu)!=receipt['best_parameter_hashes']:
        raise RuntimeError('selected unmerged parameter hashes do not match the fit receipt')
    train_bounds,val_bounds=old.partitions(record)
    if [list(b) for b in train_bounds]!=receipt['train_bounds'] or [list(b) for b in val_bounds]!=receipt['val_bounds']:
        raise RuntimeError('prefix train/validation split drift')
    valid=old.partition_windows(record,val_bounds,stats,device,'recording_causal_fixed_window')
    selection,index_sha=old.d._validation_selection(valid,128,10**9)
    if index_sha!=receipt['prefix_val_indices_sha256']:
        raise RuntimeError('prefix validation indices drift')
    cpu=CPUDecoder.from_state_dict(state)
    ys,gs,cs=[],[],[]
    with torch.inference_mode():
        for lo in range(0,len(selection),16):
            xx,yy,mask=old.d._right_aligned(valid,selection[lo:lo+16],128)
            if not mask[:,-1].all():raise RuntimeError('public validation endpoints must be eval-valid')
            g=gpu(xx)[:,-1].float().cpu().numpy()*stats[3]+stats[2]
            c=cpu.forward(xx.cpu())[:,-1].numpy()*stats[3]+stats[2]
            y=yy[:,-1].cpu().numpy()*stats[3]+stats[2]
            if not np.isfinite(g).all() or not np.isfinite(c).all() or not np.isfinite(y).all():
                raise FloatingPointError('nonfinite unmerged public prefix predictions')
            ys.append(y);gs.append(g);cs.append(c)
    y,g,c=map(np.concatenate,(ys,gs,cs))
    def r2(z):
        q=y.astype(np.float64);return float(1-np.square(q-z.astype(np.float64)).sum()/np.square(q-q.mean(0)).sum())
    gr,cr=r2(g),r2(c)
    if abs(gr-receipt['best_prefix_val_r2'])>1e-3 or abs(gr-cr)>1e-3:
        raise RuntimeError('unmerged CPU/GPU prefix R2 acceptance failed')
    return {'status':'PASS','scope':'all_public_prefix_validation_points','points':len(y),'gpu_r2':gr,'cpu_r2':cr,'score_delta':cr-gr,'max_abs_output_delta':float(np.abs(g-c).max()),'all_finite':True,'weight_math':'F.linear(base)+F.linear(delta); no weight merge','score_abs_delta_limit':1e-3}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--failed-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    failed,out=a.failed_root.resolve(),a.output.resolve()
    if out.exists() or failed in out.parents or out in failed.parents:
        raise RuntimeError('output must be new and outside the immutable failed root')
    identity=read(failed/'export_contract.json');args=identity['args']
    if args['task']!='m2' or args['method']!='lora' or args['lr']!=.003 or args['seed']!=0 or args['steps']!=2000 or args['max_steps']!=8000:
        raise RuntimeError('only the predeclared fixed M2 LoRA candidate can be resumed')
    def verify_inputs():
        if (identity['code_sha256']!={name:old.sha(ROOT/'ssm_decode'/name) for name in old.CODE}
            or old.sha(args['source'])!=identity['source_checkpoint_sha256']
            or old.sha(Path(args['source']).parent/'normalizer.npz')!=identity['source_normalizer_sha256']
            or old.reader_identity()!=identity['reader_identity'] or old._runtime_identity()!=identity['runtime_identity']
            or closure(failed)!=failed_binding):
            raise RuntimeError('frozen candidate inputs or failed-root evidence changed')
    failed_binding=closure(failed)
    verify_inputs()
    torch.set_num_threads(2)
    source=torch.load(args['source'],map_location='cpu',weights_only=False)
    with np.load(Path(args['source']).parent/'normalizer.npz') as z:
        source_stats=tuple(z[k].astype(np.float32) for k in old.KEYS)
    from apst.data.load import list_sessions
    from falcon_challenge.config import FalconConfig,FalconTask
    task_config=FalconConfig(task=FalconTask.m2)
    extra_binding={str(Path(__file__).resolve()):old.sha(__file__),str(ROOT/'ssm_decode/mamba3_cpu_lora.py'):old.sha(ROOT/'ssm_decode/mamba3_cpu_lora.py'),str(ROOT/'ssm_decode/falcon_decoder.py'):old.sha(ROOT/'ssm_decode/falcon_decoder.py'),str(ROOT/'third_party/mamba/LICENSE'):old.sha(ROOT/'third_party/mamba/LICENSE')}
    out.mkdir()
    contract={'schema':'ssm_public_unmerged_lora_export_v1','original_export_contract':identity,'failed_root':str(failed),'failed_root_file_sha256':failed_binding,'new_implementation_sha256':extra_binding,'selected_seed':0,'method':'lora','public_raw_trials':33,'no_merged_gate_relaxation':True,'runtime':'unmerged base-plus-delta projections','query_labels_used':False}
    write(out/'export_contract.json',contract)
    rows=[]
    for split in ('held_in','held_out'):
        for item in list_sessions('m2',split,root=Path(args['data_root'])):
            verify_inputs()
            if any(old.sha(k)!=v for k,v in extra_binding.items()):raise RuntimeError('unmerged exporter/runtime changed')
            record=load_public_calibration('m2',split,item['session'],args['data_root'])
            tag=task_config.hash_dataset(record.path.stem);dest=out/'banks'/tag;original=failed/'banks'/tag
            if (original/'unmerged_best.pt').is_file():
                if not (original/'receipt.json').is_file():raise RuntimeError('incomplete original fit cannot be reused')
                dest.mkdir(parents=True)
                for name in ('unmerged_best.pt','normalizer.npz','receipt.json','train_log.json'):
                    shutil.copy2(original/name,dest/name)
                origin={'reused_from':str(original),'original_unmerged_sha256':old.sha(original/'unmerged_best.pt')}
                legacy_merged_published=(original/'model.pt').is_file()
            else:
                fit_args=SimpleNamespace(device=a.device,seed=0,method='lora',lr=.003,steps=2000,max_steps=8000)
                try:old.fit_bank(record,source,source_stats,fit_args,dest,identity)
                except RuntimeError as error:
                    if str(error)!='Public bank fails convergence or merged-score acceptance':raise
                origin={'reused_from':None}
                legacy_merged_published=(dest/'model.pt').is_file()
            stats=fit_support_statistics(record.neural,[(0,len(record.neural))],source_stats)
            receipt,obj,legacy_status=check_unmerged_fit(dest,record,identity,stats,legacy_merged_published)
            parity=prefix_parity(record,stats,source,obj['state_dict'],receipt,a.device)
            write(dest/'unmerged_acceptance.json',dict(parity,origin=origin,legacy_merge_status=legacy_status,original_merged_fold_allclose=receipt['fold_allclose'],unmerged_checkpoint_sha256=old.sha(dest/'unmerged_best.pt'),normalizer_sha256=old.sha(dest/'normalizer.npz'),public_raw_trials=33,query_labels_used=False))
            # The published model is the exact selected unmerged checkpoint.
            rows.append({'tag':tag,'checkpoint_file':f'banks/{tag}/unmerged_best.pt','normalizer_file':f'banks/{tag}/normalizer.npz','calibration_receipt_sha256':old.sha(dest/'unmerged_acceptance.json')})
            if old.sha(record.path)!=record.receipt['rawfile_sha256']:raise RuntimeError('raw public calibration changed')
            print(json.dumps({'tag':tag,'status':'PASS','prefix_cpu_score_delta':parity['score_delta']}),flush=True)
    verify_inputs()
    if len(rows)!=13 or len({r['tag'] for r in rows})!=13:raise RuntimeError('M2 full public roster drift')
    if any(old.sha(k)!=v for k,v in extra_binding.items()):raise RuntimeError('unmerged implementation changed before payload publication')
    payload=out/'payload';package=payload/'ssm_decode';package.mkdir(parents=True)
    (package/'__init__.py').write_text('"""Packaged CPU SSM inference only."""\n')
    shutil.copy2(ROOT/'ssm_decode/mamba3_cpu_lora.py',package/'mamba3_cpu.py')
    shutil.copy2(ROOT/'ssm_decode/falcon_decoder.py',package/'falcon_decoder.py')
    shutil.copy2(ROOT/'third_party/mamba/LICENSE',payload/'MAMBA_LICENSE')
    for row in rows:
        for key in ('checkpoint_file','normalizer_file'):
            dst=payload/row[key];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/row[key],dst)
    files={str(f.relative_to(payload)):old.sha(f) for f in payload.rglob('*') if f.is_file()}
    verify_inputs()
    for source_file,copied_file in [(ROOT/'ssm_decode/mamba3_cpu_lora.py',package/'mamba3_cpu.py'),(ROOT/'ssm_decode/falcon_decoder.py',package/'falcon_decoder.py'),(ROOT/'third_party/mamba/LICENSE',payload/'MAMBA_LICENSE')]:
        if old.sha(source_file)!=extra_binding[str(source_file)] or old.sha(copied_file)!=extra_binding[str(source_file)]:
            raise RuntimeError('copied runtime or license differs from the frozen implementation')
    if any(old.sha(k)!=v for k,v in extra_binding.items()):raise RuntimeError('unmerged implementation changed after payload copy')
    write(payload/'payload_manifest.json',{'schema':'ssm_falcon_cpu_payload_v1','task':'m2','input_size':96,'output_size':2,'context':128,'normalization':'fixed_calibration_zscore','output_space':'official_physical','output_postprocess':'none','calibration_trials':33,'method':'lora_unmerged','models':rows,'files':files,'public_calibration_only':True,'query_labels_used':False,'test_time_parameter_updates':False,'export_contract':contract})


if __name__=='__main__':main()
