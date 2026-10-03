import pytest, torch
from ssm_decode.mamba3_official import compatibility_attempt

def test_attempt_is_explicit_about_capability():
 r=compatibility_attempt('cuda:0' if torch.cuda.is_available() else 'cpu')
 assert r['status'] in {'supported','unsupported_or_unavailable'}
 if r['status']!='supported': assert r['exception_message']
 else:
  assert r['forward_shape']==[32,128,2]
  assert r['causal_future_perturb_max_abs'] < 1e-5
