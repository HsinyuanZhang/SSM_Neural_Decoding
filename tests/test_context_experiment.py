import numpy as np, torch
from ssm_decode.context_experiment import _predict_endpoints, _windows
from ssm_decode import debug_experiment as d
def test_cross_boundary_history_and_causality():
 class M(torch.nn.Module):
  config=type('C',(),{'output_size':1})()
  def forward(self,x): return x.cumsum(1).sum(-1,keepdim=True)
 m=M();x=np.arange(10,dtype='float32')[:,None]
 p=_predict_endpoints(m,x,[(0,5),(5,10)],np.array([5]),'cpu',4)
 assert p[0,0]==sum([2,3,4,5])

def test_future_does_not_change_endpoint_prediction():
 class M(torch.nn.Module):
  config=type('C',(),{'output_size':1})()
  def forward(self,x): return x.cumsum(1).sum(-1,keepdim=True)
 x=np.arange(12,dtype='float32')[:,None];a=_predict_endpoints(M(),x,[(0,12)],np.array([6]),'cpu',5)
 x[7:]+=1000;b=_predict_endpoints(M(),x,[(0,12)],np.array([6]),'cpu',5)
 assert np.array_equal(a,b)

def test_record_mask_excludes_other_trial_labels():
 class R:
  neural=np.zeros((10,2),np.float32);behavior=np.arange(10,dtype='float32')[:,None];eval_mask=np.ones(10,bool)
 r=R();w=_windows([r],[[(2,5)]],(np.zeros(2),np.ones(2),np.zeros(1),np.ones(1)),'cpu')[0]
 assert torch.equal(torch.nonzero(w[2]).flatten(),torch.tensor([2,3,4]))
 # Pre-boundary labels occur in the input context but are masked from loss.
 loss=(w[1][w[2]]**2).mean();r.behavior[:2]=999;w2=_windows([r],[[(2,5)]],(np.zeros(2),np.ones(2),np.zeros(1),np.ones(1)),'cpu')[0]
 assert torch.equal(w[2],w2[2])
 assert torch.equal(loss,(w2[1][w2[2]]**2).mean())

def test_target_indices_match_old_policy_and_are_disjoint():
 class R:
  neural=np.arange(1800,dtype='float32')[:,None];behavior=np.arange(1800,dtype='float32')[:,None]
  eval_mask=np.ones(1800,bool);trial_bounds=[(i*50,(i+1)*50) for i in range(36)]
 class M(torch.nn.Module):
  config=type('C',(),{'output_size':1})()
  def forward(self,x): return x.cumsum(1).sum(-1,keepdim=True)
 r=R();st=(np.zeros(1,np.float32),np.ones(1,np.float32),np.zeros(1,np.float32),np.ones(1,np.float32))
 old=d._target_predictions(M(),r,st,'cpu',8)
 saved=d._predict_endpoints;d._predict_endpoints=_predict_endpoints
 try: new=d._target_predictions(M(),r,st,'cpu',8)
 finally: d._predict_endpoints=saved
 for key in ('support_indices','legacy_indices','all_valid_indices'):
  assert np.array_equal(old[key],new[key])
 assert not np.intersect1d(new['support_indices'],new['legacy_indices']).size
