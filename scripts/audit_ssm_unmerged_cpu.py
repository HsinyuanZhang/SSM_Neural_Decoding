"""Audit one fixed legal M2 unmerged LoRA checkpoint on the complete query.

This command does not train, select a checkpoint, or rank candidates. The caller
supplies the already selected seed-zero fit. It saves the official GPU reference
and CPU output for independent CPU replay.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ssm_decode import official_payload as e
from ssm_decode.data import load_recording
from ssm_decode.input_adaptation import configure_adaptation
from ssm_decode.mamba3_cpu_lora import CPUDecoder


def arr_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def r2(y,p):
 y,p=y.astype(np.float64),p.astype(np.float64);return float(1-np.square(y-p).sum()/np.square(y-y.mean(0)).sum())


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True,type=Path);p.add_argument('--probe-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.output.exists() or a.output.with_name(a.output.stem+'_arrays.npz').exists():raise RuntimeError('audit outputs must not exist')
 initial_script_sha=e.sha(__file__);initial_cpu_sha=e.sha(ROOT/'ssm_decode/mamba3_cpu_lora.py')
 torch.set_num_threads(2)
 fit,root=a.fit.resolve(),a.probe_root.resolve();contract=json.loads((root/'probe_contract.json').read_text());receipt=json.loads((fit/'receipt.json').read_text())
 if fit!=root/'fits/lora_seed0' or receipt['convergence_satisfied'] is not True or receipt['query_labels_used'] is not False:
  raise RuntimeError('only the fixed legal lora seed-zero fit is accepted')
 summary=json.loads((root/'summary.json').read_text());selected=[r for r in summary['rows'] if r['method']=='lora' and r['seed']==0]
 if len(selected)!=1:raise RuntimeError('summary must contain exactly one selected LoRA seed0 row')
 selected=selected[0]
 if (Path(selected['fit'])!=fit or e.sha(fit/'receipt.json')!=selected['receipt_sha256'] or e.sha(fit/'normalizer.npz')!=selected['normalizer_sha256']
     or e.sha(root/'scores/lora_seed0.npz')!=selected['score_npz_sha256'] or receipt['best_step']!=selected['best_step']):
  raise RuntimeError('selected fit/receipt/normalizer/score binding mismatch')
 if receipt.get('query_used_for_selection') is not False:raise RuntimeError('query selection flag is invalid')
 scope=receipt.get('adaptation',{})
 if any(scope.get(k)!=v for k,v in {'method':'lora','rank':4,'alpha':4,'lora_scope':'all','trainable_count':37000}.items()):raise RuntimeError('selected LoRA scale or scope mismatch')
 if e._runtime_identity()!=contract['runtime_identity'] or receipt['runtime_identity']!=contract['runtime_identity']:raise RuntimeError('official runtime identity mismatch')
 for k,sha in contract['code_sha256'].items():
  if e.sha(ROOT/'ssm_decode'/k)!=sha:raise RuntimeError('frozen original probe implementation changed')
 source_path=Path(receipt['args']['source']) if 'args' in receipt else Path(contract['source_checkpoint']) if 'source_checkpoint' in contract else ROOT/'results/cross_session_iteration_b/official_20261003/source/m2/wide/best.pt'
 if e.sha(source_path)!=contract['source_checkpoint_sha256']:raise RuntimeError('source checkpoint changed')
 source=torch.load(source_path,map_location='cpu',weights_only=False)
 ck=fit/'unmerged_best.pt';norm=fit/'normalizer.npz';state=torch.load(ck,map_location='cpu',weights_only=False)
 if any(state.get('args',{}).get(k)!=v for k,v in {'method':'lora','seed':0,'lr':.003,'steps':2000,'max_steps':8000}.items()):raise RuntimeError('selected state arguments differ')
 if state['best_step']!=receipt['best_step']:raise RuntimeError('checkpoint step differs from the selected fit')
 record=load_recording('m2','held_in',contract['target_session'],root=Path(source['args']['data_root']))
 if e.sha(record.path)!=contract['formal_binding']['target_data_sha256']:raise RuntimeError('raw query data changed')
 with np.load(root/'scores/lora_seed0.npz') as z:
  indices=z['allvalid_indices'];truth=z['truth_allvalid'];saved_merged=z['zero_allvalid']
  if not np.isfinite(truth).all() or not np.isfinite(saved_merged).all():raise RuntimeError('saved truth/reference must be finite')
  if indices.dtype.kind not in 'iu' or len(indices)!=14115 or not np.array_equal(truth,record.behavior[indices]):raise RuntimeError('full query cohort/truth differs')
  if arr_sha(indices)!=contract['formal_binding']['cohort_index_hash'] or arr_sha(truth)!=contract['formal_binding']['cohort_truth_hash']:raise RuntimeError('full query hash differs')
 with np.load(norm) as z:stats=tuple(z[k].astype(np.float32) for k in e.KEYS)
 if e.sha(norm)!=receipt['normalizer_sha256']:raise RuntimeError('selected normalizer changed')
 bound_files=[source_path,ck,norm,fit/'receipt.json',root/'probe_contract.json',root/'summary.json',root/'scores/lora_seed0.npz',Path(record.path)]
 initial_file_hashes={str(f):e.sha(f) for f in bound_files}
 gpu=e._base_model(source['args'],96,2,a.device);configure_adaptation(gpu,'lora',rank=4,alpha=4,lora_scope='all',train_input_bias=True);gpu.load_state_dict(state['state_dict'],strict=True);gpu.eval()
 if e._parameter_hashes(gpu)!=receipt['best_parameter_hashes']:raise RuntimeError('unmerged selected parameter hashes differ')
 x=((record.neural-stats[0])/stats[1]).astype(np.float32);started=time.perf_counter()
 with torch.inference_mode():
  gp=e.d._predict_endpoints(gpu,x,[(0,len(x))],indices,a.device,128,batch=256)*stats[3]+stats[2]
 cpu=CPUDecoder.from_state_dict(state['state_dict']);cp=[];cpu_started=time.perf_counter()
 with torch.inference_mode():
  for lo in range(0,len(indices),16):
   windows=np.zeros((len(indices[lo:lo+16]),128,96),np.float32)
   for row,end in enumerate(indices[lo:lo+16]):
    left=max(0,int(end)-127);n=int(end)-left+1;windows[row,-n:]=x[left:int(end)+1]
   cp.append(cpu.forward(torch.from_numpy(windows))[:,-1].numpy()*stats[3]+stats[2])
  cp=np.concatenate(cp)
  original=torch.from_numpy(x[indices[0]-127:indices[0]+1].copy())[None]
  perturbed=original.clone();perturbed[:,96:]+=3
  left,right=cpu.forward(original)[:,:96],cpu.forward(perturbed)[:,:96]
 if not np.isfinite(gp).all() or not np.isfinite(cp).all() or not torch.equal(left,right):raise RuntimeError('finite/future-causality gate failed')
 gr,cr=r2(truth,gp),r2(truth,cp)
 merged_r2=r2(truth,saved_merged)
 if abs(gr-merged_r2)>1e-3 or abs(merged_r2-selected['r2'])>2e-6:raise RuntimeError('unmerged deployment changed the selected GPU performance')
 if (initial_script_sha!=e.sha(__file__) or initial_cpu_sha!=e.sha(ROOT/'ssm_decode/mamba3_cpu_lora.py')
     or any(e.sha(f)!=h for f,h in initial_file_hashes.items()) or e._runtime_identity()!=contract['runtime_identity']):raise RuntimeError('audit code, runtime or bound data changed during evaluation')
 if abs(gr-cr)>1e-3:raise RuntimeError('unmerged full-query CPU/GPU score gate failed')
 a.output.parent.mkdir(parents=True,exist_ok=True);arrays=a.output.with_name(a.output.stem+'_arrays.npz');np.savez(arrays,indices=indices,truth=truth,gpu=gp,cpu=cp)
 report={'schema':'ssm_unmerged_cpu_gpu_full_query_v1','status':'PASS','scope':'full_query','task':'m2','method':'lora_unmerged','seed':0,'lr':.003,'checkpoint':str(ck),'checkpoint_sha256':e.sha(ck),'source_sha256':e.sha(source_path),'normalizer_sha256':e.sha(norm),'probe_contract_sha256':e.sha(root/'probe_contract.json'),'raw_query_sha256':e.sha(record.path),'cohort_sha256':arr_sha(indices),'truth_sha256':arr_sha(truth),'query_bins':len(indices),'gpu_physical_r2':gr,'cpu_physical_r2':cr,'score_delta':cr-gr,'saved_merged_gpu_r2':merged_r2,'unmerged_vs_saved_merged_score_delta':gr-merged_r2,'selected_input_file_sha256':initial_file_hashes,'score_abs_delta_limit':1e-3,'max_absolute_output_delta':float(np.abs(gp-cp).max()),'all_finite':True,'causal_prefix_bitwise_equal':True,'cpu_seconds':time.perf_counter()-cpu_started,'total_seconds':time.perf_counter()-started,'cpu_threads':2,'arrays':str(arrays.resolve()),'arrays_sha256':e.sha(arrays),'script_sha256':e.sha(__file__),'cpu_runtime_sha256':e.sha(ROOT/'ssm_decode/mamba3_cpu_lora.py'),'runtime':e._runtime_identity(),'selection_boundary':'fixed caller-supplied seed0 checkpoint; no model ranking or query HPO'}
 a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','gpu_r2':gr,'cpu_r2':cr,'score_delta':cr-gr}))

if __name__=='__main__':main()
