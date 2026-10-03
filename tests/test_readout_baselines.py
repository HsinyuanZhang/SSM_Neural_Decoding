import numpy as np
from ssm_decode.readout_baselines import _lag_features
from ssm_decode.data import Recording
from pathlib import Path
def test_lags_are_zero_at_trial_boundary():
 r=Recording('m1','held_in','x',Path('/tmp/x'),np.zeros((6,2),np.float32),np.zeros((6,1),np.float32),np.zeros(6),np.ones(6,bool),((0,3),(3,6)))
 x=np.arange(12,dtype=np.float32).reshape(6,2);f=_lag_features(x,r,np.array([3]),(0,1));assert np.all(f[0,2:]==0)
