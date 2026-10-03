from pathlib import Path
import numpy as np
from ssm_decode.data import Recording, split_support_query
from ssm_decode.experiment import _predict_trials, evaluate_target
from ssm_decode.models import ModelConfig, build_model

def test_support_query_is_trial_disjoint():
 r=Recording('m1','held_in','s',Path('/tmp/s'),np.zeros((100,2),np.float32),np.zeros((100,1),np.float32),np.zeros(100),np.ones(100,bool),((0,25),(25,50),(50,75),(75,100)))
 support,query=split_support_query(r,support_trials=2,sequence=10)
 support_bins=set((support[:,None]+np.arange(10)).reshape(-1).tolist()); query_bins=set((query+9).tolist())
 assert not support_bins & query_bins

def test_budgets_share_common_query_tail():
 r=Recording('m1','held_in','s',Path('/tmp/s'),np.zeros((150,2),np.float32),np.zeros((150,1),np.float32),np.zeros(150),np.ones(150,bool),tuple((i,i+25) for i in range(0,150,25)))
 _,q5=split_support_query(r,support_trials=2,query_start_trials=4,sequence=10)
 _,q3=split_support_query(r,support_trials=3,query_start_trials=4,sequence=10)
 assert np.array_equal(q5,q3)

def test_batched_trial_predict_and_every_calibration_are_runnable():
 r=Recording('m2','held_in','s',Path('/tmp/s'),np.ones((80,3),np.float32),np.tile(np.arange(2,dtype=np.float32),(80,1)),np.zeros(80),np.ones(80,bool),((0,20),(20,40),(40,60),(60,80)))
 stats={'x_mean':np.zeros(3,np.float32),'x_std':np.ones(3,np.float32),'y_mean':np.zeros(2,np.float32),'y_std':np.ones(2,np.float32)}
 model=build_model(ModelConfig(3,2,width=4,kind='diag'))
 pred=_predict_trials(model,r,stats,'cpu',batch_trials=2)
 assert pred.shape==(80,2)
 for adaptation in ('zero','ridge','rls','delta'):
  row=evaluate_target(r,stats,pred,support_trials=2,query_start_trials=2,sequence=5,adaptation=adaptation)
  assert row['n_query_bins'] and row['support_query_overlap'] is False
