"""Fail-closed FALCON adapter for packaged plain CPU SSM payloads only.

The runtime image must set ``PYTHONDONTWRITEBYTECODE=1``: payload closure is
literal and intentionally does not silently ignore a generated ``__pycache__``.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
try:
    from falcon_challenge.interface import BCIDecoder
except ImportError:  # Allows isolated CPU payload tests; SDK supplies the real base.
    class BCIDecoder:
        def __init__(self, task_config=None, batch_size=1): self.task_config, self.batch_size = task_config, batch_size
from .mamba3_cpu import CPUDecoder

SCHEMA="ssm_falcon_cpu_payload_v1"; CONTEXT=128

def _sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def _safe(root, rel):
 p=(root/rel).resolve()
 if not isinstance(rel,str) or not rel or p==root or root not in p.parents: raise ValueError('unsafe payload relative path')
 return p

class SSMFalconDecoder(BCIDecoder):
 """Recording-causal fixed-window-128 SSM inference, with no target adaptation."""
 def __init__(self, task_config, model_path, batch_size=1):
  super().__init__(task_config=task_config,batch_size=batch_size)
  self.task_config=task_config;self.root=Path(model_path).resolve();self.batch_size=int(batch_size)
  if self.batch_size<1: raise ValueError('batch_size must be positive')
  self.doc=json.loads((self.root/'payload_manifest.json').read_text())
  self._validate_payload();self.ids=[];self.history=None

 def _tag(self,x): return self.task_config.hash_dataset(Path(x).stem)
 def _validate_payload(self):
  d=self.doc
  if d.get('schema')!=SCHEMA or d.get('context')!=CONTEXT or d.get('normalization')!='fixed_calibration_zscore' or d.get('output_space')!='official_physical' or d.get('output_postprocess')!='none': raise ValueError('payload contract drift')
  task=d.get('task'); expected={'m1':(64,16,10),'m2':(96,2,33)}
  if task not in expected or (d.get('input_size'),d.get('output_size'),d.get('calibration_trials'))!=expected[task]: raise ValueError('task geometry/calibration contract drift')
  files=d.get('files'); rows=d.get('models')
  if not isinstance(files,dict) or not isinstance(rows,list) or not rows: raise ValueError('payload files/models required')
  actual={str(p.relative_to(self.root)):p for p in self.root.rglob('*') if p.is_file() and p.relative_to(self.root).as_posix()!='payload_manifest.json'}
  if set(actual)!=set(files): raise ValueError('payload file closure drift')
  for rel,digest in files.items():
   p=_safe(self.root,rel)
   if not isinstance(digest,str) or len(digest)!=64 or _sha(p)!=digest: raise ValueError('payload hash drift '+rel)
  sdk_task=getattr(self.task_config,'task',None); sdk_name=getattr(sdk_task,'name',None)
  if sdk_name is not None and str(sdk_name).lower()!=task: raise ValueError('SDK task/manifest task mismatch')
  if self.batch_size>{'m1':4,'m2':7}[task]: raise ValueError('official task batch cap exceeded')
  self.task,self.input_size,self.output_size=task,expected[task][0],expected[task][1];self.entries={};self.cache={}
  for r in rows:
   if not isinstance(r,dict) or set(('tag','checkpoint_file','normalizer_file'))-set(r): raise ValueError('model row drift')
   tag=r['tag']; ck=_safe(self.root,r['checkpoint_file']); nm=_safe(self.root,r['normalizer_file'])
   if not isinstance(tag,str) or tag in self.entries or str(ck.relative_to(self.root)) not in files or str(nm.relative_to(self.root)) not in files: raise ValueError('model roster/path drift')
   self.entries[tag]=(ck,nm)

 def _load(self,tag):
  if tag in self.cache:return self.cache[tag]
  ck,nm=self.entries[tag]; obj=torch.load(ck,map_location='cpu',weights_only=False); state=obj['state_dict'] if isinstance(obj,dict) and 'state_dict' in obj else obj
  if not isinstance(state,dict): raise ValueError('checkpoint must be plain state_dict or state_dict wrapper')
  model=CPUDecoder.from_state_dict(state)
  if getattr(model,'input_size',None)!=self.input_size or getattr(model,'output_size',None)!=self.output_size or getattr(model,'context',None)!=CONTEXT: raise ValueError('CPU model/manifest geometry drift')
  vals=[]
  with np.load(nm,allow_pickle=False) as z:
   for k,n in [('x_mean',self.input_size),('x_std',self.input_size),('y_mean',self.output_size),('y_std',self.output_size)]:
    if k not in z: raise ValueError('normalizer missing '+k)
    a=np.asarray(z[k],dtype=np.float32)
    if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError('normalizer geometry/nonfinite '+k)
    vals.append(a)
  if not (vals[1]>0).all() or not (vals[3]>0).all():raise ValueError('normalizer std must be positive')
  self.cache[tag]=(model,*vals);return self.cache[tag]

 def reset(self,dataset_tags:Iterable[Path]=(Path(''),)):
  ids=[self._tag(x) for x in dataset_tags]
  if not ids or len(ids)>self.batch_size or len(ids)!=len(set(ids)) or any(x not in self.entries for x in ids):raise ValueError('unknown/duplicate/oversize official roster')
  self.ids=ids;self.history=np.zeros((len(ids),CONTEXT,self.input_size),np.float32)
 def observe(self,neural_observations):
  self.predict(neural_observations)
  return None
 def on_done(self,dones): return None
 @torch.inference_mode()
 def predict(self,neural_observations):
  if self.history is None:raise RuntimeError('reset must precede predict')
  x=np.asarray(neural_observations,dtype=np.float32); active=x.shape[0] if x.ndim==2 else 0
  if x.ndim!=2 or x.shape[1]!=self.input_size or not 1<=active<=len(self.ids) or not np.isfinite(x).all():raise ValueError(f'expected finite active [B,{self.input_size}]')
  out=np.empty((active,self.output_size),np.float32)
  # Each row has an independent full recording clock; inactive padded rows stay frozen.
  for i in range(active):
   model,xm,xs,ym,ys=self._load(self.ids[i]); norm=(x[i]-xm)/xs
   self.history[i,:-1]=self.history[i,1:];self.history[i,-1]=norm
   y=model.forward(torch.from_numpy(self.history[i:i+1]))[0,-1].cpu().numpy().astype(np.float32)
   out[i]=y*ys+ym
  if not np.isfinite(out).all():raise RuntimeError('nonfinite output')
  return out
