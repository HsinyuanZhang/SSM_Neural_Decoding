import numpy as np,pytest
from ssm_decode import official_calibration as oc
def fake(monkeypatch,task='m1',trials=10,leading=5):
 L=leading+trials*3+4; mark=np.zeros(L,bool); starts=np.arange(leading,leading+trials*3,3);mark[starts]=True
 def resolve(*a):return __import__('pathlib').Path('/tmp/public-held-out-calib-x.nwb')
 def read(*a):return np.ones((L,64 if task=='m1' else 96),np.float32),np.ones((L,16 if task=='m1' else 2),np.float32),mark,np.ones(L,bool)
 monkeypatch.setattr(oc,'_resolve',resolve);monkeypatch.setattr(oc,'_official',read);monkeypatch.setattr(oc,'_trial_ids',lambda p:(np.arange(trials),np.arange(trials),np.arange(trials)+1));monkeypatch.setattr(oc,'_sha_file',lambda p:'0'*64);return starts
def test_m1_pretrial_not_counted_and_masked(monkeypatch):
 starts=fake(monkeypatch,'m1',10,5);r=oc.load_public_calibration('m1','held_out','x','/tmp');assert r.neural.shape[0]==5+30+4 and r.receipt['leadingpretrialbins']==5 and len(r.trial_bounds)==10 and not r.eval_mask[:5].any()
def test_m2_first33_not_usable_filter(monkeypatch):
 starts=fake(monkeypatch,'m2',43,2);r=oc.load_public_calibration('m2','held_in','x','/tmp');assert len(r.trial_bounds)==33 and r.neural.shape[0]==starts[33] and r.receipt['raw_nwb_trial_ids_first_n']==list(range(33))
def test_rejects_nonpublic(monkeypatch):
 with pytest.raises(PermissionError):oc.load_public_calibration('m1','minival','x','/tmp')
 fake(monkeypatch,'m1',9)
 with pytest.raises(ValueError):oc.load_public_calibration('m1','held_out','x','/tmp')
def test_zero_start_is_a_real_trial_and_pathguard(monkeypatch):
 starts=fake(monkeypatch,'m2',43,0);r=oc.load_public_calibration('m2','held_out','x','/tmp');assert r.trial_bounds[0][0]==0 and len(r.trial_bounds)==33
 from pathlib import Path
 with pytest.raises(ValueError):oc._validate_path(Path('/tmp/private/query.nwb'),Path('/tmp'),'held_out')
 with pytest.raises(ValueError):oc._validate_path(Path('/tmp/x-held-out-calib/evil.txt'),Path('/tmp'),'held_out')
