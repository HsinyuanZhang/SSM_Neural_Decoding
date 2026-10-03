import torch, pytest
from ssm_decode.modern_models import ModernConfig,build_modern_model
@pytest.mark.parametrize('kind',["s4d","mamba2"])
def test_causal_chunk_and_grad(kind):
 torch.manual_seed(0);m=build_modern_model(ModernConfig(5,3,kind,width=32,layers=2,state_size=4,dropout=0)).train();x=torch.randn(2,9,5,requires_grad=True); y,s=m(x,return_state=True); a,st=m(x[:,:4],return_state=True);b,_=m(x[:,4:],st,return_state=True);assert y.shape==(2,9,3);assert torch.allclose(y,torch.cat([a,b],1),atol=1e-5);y.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
@pytest.mark.parametrize('kind',["s4d","mamba2"])
def test_arbitrary_length_and_causality(kind):
 m=build_modern_model(ModernConfig(4,2,kind,width=32,layers=1,state_size=4,dropout=0)).eval();x=torch.randn(1,7,4);z=x.clone();z[:,4:]+=10;assert m(torch.randn(1,1,4)).shape==(1,1,2);assert torch.allclose(m(x)[:,:4],m(z)[:,:4],atol=1e-5);assert sum(p.numel() for p in m.parameters())>1000

@pytest.mark.parametrize("kind", ["s4d", "mamba2"])
@pytest.mark.parametrize("length", [128, 256])
def test_training_lengths_match_streaming_under_strong_decay(kind, length):
 m=build_modern_model(ModernConfig(6,3,kind,width=128,layers=2,state_size=8,dropout=0)).train()
 if kind == "mamba2":
  for b in m.blocks: b.log_a.data.fill_(2.0)
 x=torch.randn(1,length,6,requires_grad=True); full,state=m(x,return_state=True); pieces=[]; s=None
 for chunk in x.split(31,dim=1): y,s=m(chunk,s,return_state=True); pieces.append(y)
 streamed=torch.cat(pieces,1); assert torch.allclose(full,streamed,rtol=2e-4,atol=2e-4)
 full.square().mean().backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
