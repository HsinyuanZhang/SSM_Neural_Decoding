"""Export mutable session-profile folded projections into raw-count affine form."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

from .data import DEFAULT_ROOT, load_recording
from .profile_experiment import _support_rows
from .data import split_support_query

def sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def fold_raw_projection(weights: np.ndarray, mean_raw: np.ndarray, std_raw: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
 """``(raw-mean)/std @ W == raw @ P + bias`` exactly up to FP arithmetic."""
 w=np.asarray(weights,np.float32);mean=np.asarray(mean_raw,np.float32);std=np.asarray(std_raw,np.float32)
 if w.ndim!=2 or mean.shape!=(w.shape[0],) or std.shape!=(w.shape[0],) or np.any(std<=0):raise ValueError('fold geometry/std invalid')
 p=w/std[:,None];return p.astype(np.float32),(-mean@p).astype(np.float32)

def quantize_columns(p: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
 scale=np.max(np.abs(p),axis=0).astype(np.float32)/127.;scale[scale<1e-12]=1.
 return np.rint(p/scale[None]).clip(-127,127).astype(np.int8),scale

def effective_raw_stats(manifest: dict, stats: dict[str,np.ndarray], target) -> tuple[np.ndarray,np.ndarray,str]:
 if not manifest.get('support_normalize_neural',False):return stats['x_mean'],stats['x_std'],'source_only'
 count=int(manifest['support_trials']);x,_=_support_rows(target,stats,count,int(manifest['sequence']))
 mean=x.mean(0);std=x.std(0);std[std<1e-6]=1.
 return stats['x_mean']+stats['x_std']*mean,stats['x_std']*std,'target_support_prefix_only'

def development_sanity(target, stats, count: int, sequence: int) -> dict:
 """Read-only local-dev shift summary; query labels never enter fitting."""
 support,_=_support_rows(target,stats,count,sequence)
 _,starts=split_support_query(target,support_trials=count,query_start_trials=count,sequence=sequence)
 q=np.unique(starts+sequence-1);q=q[target.eval_mask[q]&np.isfinite(target.behavior[q]).all(1)]
 query_x=((target.neural[q]-stats['x_mean'])/stats['x_std']).astype(np.float32)
 sy=target.behavior[np.unique(split_support_query(target,support_trials=count,query_start_trials=count,sequence=sequence)[0][:,None]+np.arange(sequence)[None,:]).reshape(-1)]
 sy=sy[np.isfinite(sy).all(1)];qy=target.behavior[q]
 return {'input_channels_source_normalizer':int(stats['x_mean'].size),'input_channels_target':int(target.neural.shape[1]),'channel_identity_evidence':'official loader exposes fixed ordered channel arrays, not electrode IDs; only channel-count/order contract checked','support_standardized_x_mean_abs':float(np.mean(np.abs(support.mean(0)))),'support_standardized_x_std_mean':float(support.std(0).mean()),'query_standardized_x_mean_abs':float(np.mean(np.abs(query_x.mean(0)))),'query_standardized_x_std_mean':float(query_x.std(0).mean()),'support_behavior_rows':int(len(sy)),'query_behavior_rows':int(len(qy)),'support_behavior_physical_std':np.std(sy,axis=0).astype(float).tolist(),'query_behavior_physical_std':np.std(qy,axis=0).astype(float).tolist(),'query_labels_used_for_fit':False,'query_labels_used_for_diagnostic_only':True}

def export_run(run: Path, output: Path) -> list[dict]:
 run=Path(run).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((run/'manifest.json').read_text());plan=manifest['plan'];target_name=plan['cross_session_local_dev']['target_session'];root=Path(plan['data_root'])
 target=load_recording(manifest['args']['task'] if 'args' in manifest else plan['task'],'held_in',target_name,root=root)
 with np.load(run/'source_normalizer.npz',allow_pickle=False) as z:stats={k:np.asarray(z[k]) for k in z.files}
 mean,std,normalization=effective_raw_stats(manifest,stats,target); rows=[]
 sanity=development_sanity(target,stats,int(manifest['support_trials']),int(manifest['sequence']))
 for path in sorted(run.glob('*.weights.npz')):
  with np.load(path,allow_pickle=False) as z:w=np.asarray(z['weights'],np.float32);profile=np.asarray(z['profile'],np.float32)
  p,b=fold_raw_projection(w,mean,std);q,s=quantize_columns(p)
  raw=target.neural[:50].astype(np.float32);norm=(raw-mean)/std;err=float(np.max(np.abs(norm@w-(raw@p+b))))
  if err>3e-4:raise RuntimeError(f'fold equivalence failed {path.name}: {err}')
  dest=output/(path.stem+'.folded.npz');np.savez_compressed(dest,raw_projection=p,bias=b,weight_int8=q,column_scale=s,profile_weights=w,profile=profile)
  rows.append({'source_weights':str(path),'source_weights_sha256':sha(path),'export':str(dest),'export_sha256':sha(dest),'channels':int(w.shape[0]),'width':int(w.shape[1]),'normalization':normalization,'equivalence_first50_max_abs':err,'quantization':'int8_per_output_column_symmetric','projection_state':'mutable_per_session_profile_generated','hardware_note':'RRAM may store fixed encoder weights; folded session projection is mutable and requires SRAM/write path unless explicitly programmed'})
 provenance={'schema':'profile_folded_raw_projection_v1','run':str(run),'run_manifest_sha256':sha(run/'manifest.json'),'source_normalizer_sha256':sha(run/'source_normalizer.npz'),'target_session':target_name,'target_surface':'held_in_local_only','held_out_accessed':False,'development_sanity':sanity,'rows':rows}
 (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');return rows

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args(argv);print(json.dumps(export_run(a.run,a.output),indent=2))
if __name__=='__main__':main()
