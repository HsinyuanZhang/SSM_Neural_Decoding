"""Offline loader for public FALCON calibration NWBs; never a query loader."""
from __future__ import annotations
import hashlib,sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DEFAULT_ROOT=Path('/mnt/data/work_host/SPINT/SPINT-main/data'); BUDGET={'m1':10,'m2':33}; TOKEN={'held_in':'held-in-calib','held_out':'held-out-calib'}
@dataclass(frozen=True)
class PublicCalibration:
 task:str;split:str;session:str;path:Path;neural:np.ndarray;behavior:np.ndarray;eval_mask:np.ndarray;trial_change:np.ndarray;trial_bounds:tuple;receipt:dict
def _sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def _sha_array(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def _resolve(task,split,session,root):
 apst=Path(__file__).resolve().parents[2]/'APST'/'src';sys.path.insert(0,str(apst)) if str(apst) not in sys.path else None
 from apst.data.load import list_sessions
 rows=[r for r in list_sessions(task,split,root=root) if r['session']==session]
 if len(rows)!=1:raise KeyError(f'expected one public calibration file for {task}/{split}/{session}')
 p=Path(rows[0]['path']).resolve();base=Path(root).resolve();_validate_path(p,base,split)
 return p
def _validate_path(p,base,split):
 if base not in p.parents or p.suffix!='.nwb' or not p.name.endswith('.nwb') or not p.parent.name.endswith('-'+TOKEN[split]):raise ValueError('non-calibration/private path rejected')
def _official(path,task):
 apst=Path(__file__).resolve().parents[2]/'APST'/'src';sys.path.insert(0,str(apst)) if str(apst) not in sys.path else None
 from apst.data.load import load_nwb_file
 return load_nwb_file(path,task)
def _trial_ids(path):
 import h5py
 with h5py.File(path,'r') as f:
  g=f['intervals']['trials']; ids=np.asarray(g['id']) if 'id' in g else np.arange(len(g['start_time']))
  starts=np.asarray(g['start_time']) if 'start_time' in g else None
  stops=np.asarray(g['stop_time']) if 'stop_time' in g else None
 return ids,starts,stops
def load_public_calibration(task,split,session,data_root=DEFAULT_ROOT):
 task=task.lower()
 if task not in BUDGET or split not in TOKEN:raise PermissionError('only m1/m2 held_in or held_out public calibration is permitted')
 root=Path(data_root).resolve();p=_resolve(task,split,session,root); neural,behavior,trial_change,eval_mask=_official(p,task)
 x=np.asarray(neural,np.float32);y=np.asarray(behavior,np.float32);mark=np.asarray(trial_change,bool).reshape(-1);mask=np.asarray(eval_mask,bool).reshape(-1)
 dims={'m1':(64,16),'m2':(96,2)}[task]
 if x.ndim!=2 or y.ndim!=2 or x.shape[1:]!=(dims[0],) or y.shape[1:]!=(dims[1],) or x.shape[0]!=y.shape[0] or mark.size!=len(x) or mask.size!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('official reader geometry drift')
 ids,starts,stops=_trial_ids(p);n=BUDGET[task]
 if np.asarray(ids).ndim!=1 or len(ids)<n or len(starts)!=len(ids) or len(stops)!=len(ids) or not np.isfinite(starts).all() or not np.isfinite(stops).all() or not np.all(np.diff(starts)>0) or not np.all(stops>=starts):raise ValueError('NWB trial metadata drift')
 actual=np.flatnonzero(mark) # A true trial at bin zero is a true calibration trial.
 if len(actual)<n:raise ValueError('SDK trial-change has fewer true trial starts than required')
 # The table, not a generic usable-window rule, defines legal trial IDs 0..N-1.
 if len(actual)!=len(ids):raise ValueError('SDK/NWB actual trial count disagreement')
 prefix_end=int(actual[n]) if len(actual)>n else len(x)
 bounds=tuple((int(actual[i]),int(actual[i+1] if i+1<n else prefix_end)) for i in range(n))
 allowed=np.zeros(prefix_end,bool)
 for a,b in bounds:allowed[a:b]=True
 cropped_mask=mask[:prefix_end]&allowed
 receipt={'rawfile_sha256':_sha_file(p),'raw_nwb_trial_ids_first_n':[int(v) for v in ids[:n]],'total_actual_trialcount':int(len(ids)),'realtrialbounds':[list(z) for z in bounds],'prefixend':prefix_end,'leadingpretrialbins':int(actual[0]),'usedneuralarray_sha256':_sha_array(x[:prefix_end]),'usedlabels_sha256':_sha_array(y[:prefix_end]),'usedmask_sha256':_sha_array(cropped_mask),'query_labels_used':False,'publiccalibration_only':True,'calibration_trials':n,'official_reader':'apst.data.load.load_nwb_file'}
 return PublicCalibration(task,split,session,p,x[:prefix_end],y[:prefix_end],cropped_mask,mark[:prefix_end],bounds,receipt)
