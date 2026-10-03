import torch
from ssm_decode.profile_models import LearnedProfileFrontend

def test_joint_permutation_invariant_and_masked():
 torch.manual_seed(2);f=LearnedProfileFrontend(5,7);x=torch.randn(3,4);p=torch.randn(4,5);m=torch.tensor([1,1,0,1],dtype=torch.bool)
 f.set_profile(p,m);a=f(x);perm=torch.tensor([2,0,3,1]);f.set_profile(p[perm],m[perm]);b=f(x[:,perm]);assert torch.allclose(a,b,atol=1e-6)
