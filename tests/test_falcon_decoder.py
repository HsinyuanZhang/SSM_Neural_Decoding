import hashlib,json,sys,types
from pathlib import Path
import numpy as np,pytest,torch
# The production module uses the SDK base when installed; a tiny SDK stand-in
# makes this payload-only test independent of external challenge packages.
pkg=types.ModuleType('falcon_challenge'); iface=types.ModuleType('falcon_challenge.interface')
class Base:
 def __init__(self,task_config=None,batch_size=1):self.task_config=task_config;self.batch_size=batch_size
iface.BCIDecoder=Base;sys.modules.setdefault('falcon_challenge',pkg);sys.modules.setdefault('falcon_challenge.interface',iface)
from ssm_decode import falcon_decoder as fd

class Cfg:
 task=types.SimpleNamespace(name='m1')
 def hash_dataset(self,x):return 'tag-'+x
class Toy:
 input_size=64;output_size=16;context=128
 def __init__(self,shift):self.shift=shift
 def forward(self,x):
  # deterministic last-bin readout: proves normalized cold padding and history
  return torch.stack((x[...,0].sum(-1)+self.shift,)*16,-1) if x.shape[-1]==64 else torch.stack((x[...,0].sum(-1)+self.shift,)*2,-1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def payload(tmp,task='m1',tags=('a','b')):
 c,o,k=(64,16,10) if task=='m1' else (96,2,33); rows=[];files={}
 for i,t in enumerate(tags):
  ck=tmp/f'{t}.pt';torch.save({'state_dict':{'toy':torch.tensor(i)}},ck);nm=tmp/f'{t}.npz';np.savez(nm,x_mean=np.full(c,2,np.float32),x_std=np.full(c,2,np.float32),y_mean=np.full(o,3,np.float32),y_std=np.full(o,4,np.float32));rows.append({'tag':'tag-'+t,'checkpoint_file':ck.name,'normalizer_file':nm.name});files[ck.name]=sha(ck);files[nm.name]=sha(nm)
 d={'schema':'ssm_falcon_cpu_payload_v1','task':task,'input_size':c,'output_size':o,'context':128,'normalization':'fixed_calibration_zscore','output_space':'official_physical','output_postprocess':'none','calibration_trials':k,'models':rows,'files':files};(tmp/'payload_manifest.json').write_text(json.dumps(d));return d
def setup(tmp,task='m1'):
 payload(tmp,task); old=fd.CPUDecoder.from_state_dict
 def make(s):
  z=Toy(float(s['toy']));z.input_size=64 if task=='m1' else 96;z.output_size=16 if task=='m1' else 2;return z
 fd.CPUDecoder.from_state_dict=make;c=Cfg();c.task=types.SimpleNamespace(name=task);d=fd.SSMFalconDecoder(c,tmp,2);return d,old
def test_normalized_zero_cold_start_and_no_m2_scale(tmp_path):
 d,old=setup(tmp_path,'m2')
 try:
  d.reset([Path('a')]); y=d.predict(np.full((1,96),2,np.float32)); assert np.allclose(y,3) # normalized zero, no /5 or EMA
 finally:fd.CPUDecoder.from_state_dict=old
def test_clock_reset_reorder_partial_and_causal_prefix(tmp_path):
 d,old=setup(tmp_path)
 try:
  d.reset([Path('b'),Path('a')]); first=d.predict(np.stack((np.full(64,4,np.float32),np.full(64,2,np.float32)))); second=d.predict(np.full((1,64),6,np.float32)); assert np.allclose(first[0],11) and np.allclose(first[1],3) and np.allclose(second,19)
  # row 1 was inactive in second call: it remains at its old history state.
  before=d.history[1].copy(); d.predict(np.full((1,64),2,np.float32)); assert np.array_equal(d.history[1],before)
  d.reset([Path('a')]);assert np.allclose(d.predict(np.full((1,64),2,np.float32)),3)
 finally:fd.CPUDecoder.from_state_dict=old
def test_payload_rejections(tmp_path):
 payload(tmp_path);d=json.loads((tmp_path/'payload_manifest.json').read_text());d['models'][0]['checkpoint_file']='../bad.pt';(tmp_path/'payload_manifest.json').write_text(json.dumps(d))
 with pytest.raises(ValueError):fd.SSMFalconDecoder(Cfg(),tmp_path,2)
 payload(tmp_path);d=fd.SSMFalconDecoder(Cfg(),tmp_path,2)
 with pytest.raises(ValueError):d.reset([Path('unknown')])
def test_sdk_task_batch_and_nested_manifest_file_are_rejected(tmp_path):
 payload(tmp_path); d=json.loads((tmp_path/'payload_manifest.json').read_text()); sub=tmp_path/'nested';sub.mkdir();(sub/'payload_manifest.json').write_text('x');d['files']['nested/payload_manifest.json']=sha(sub/'payload_manifest.json');(tmp_path/'payload_manifest.json').write_text(json.dumps(d))
 c=Cfg();c.task=types.SimpleNamespace(name='m2')
 with pytest.raises(ValueError):fd.SSMFalconDecoder(c,tmp_path,2)
 payload(tmp_path)
 with pytest.raises(ValueError):fd.SSMFalconDecoder(Cfg(),tmp_path,5)
def test_observe_advances_exactly_like_discarded_predict(tmp_path):
 d,old=setup(tmp_path)
 try:
  a=np.full((1,64),4,np.float32);b=np.full((1,64),6,np.float32)
  d.reset([Path('a')]);d.observe(a); observed=d.predict(b);history=d.history.copy()
  d.reset([Path('a')]);d.predict(a);direct=d.predict(b)
  assert np.array_equal(observed,direct) and np.array_equal(history,d.history)
 finally:fd.CPUDecoder.from_state_dict=old
