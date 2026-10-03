"""Independent CPU checks for functional custom Mamba-3 ports."""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import ssm_decode.mamba3_research_adapters as ports

def _kernel(**k):
    q,kb,v,z=k['Q'],k['K'],k['V'],k['Z']; state=torch.zeros_like(v[:,0]); out=[]
    for t in range(v.shape[1]):
        state=.5*state+v[:,t]+q[:,t].mean(-1,keepdim=True)+kb[:,t].mean(-1,keepdim=True)
        out.append(torch.nn.functional.silu(z[:,t])*state)
    return torch.stack(out,1)

class Core(nn.Module):
 def __init__(self):
  super().__init__(); self.d_inner=4;self.d_state=2;self.num_bc_heads=1;self.nheads=1;self.headdim=4;self.num_rope_angles=1;self.mimo_rank=1;self.is_mimo=False;self.is_outproj_norm=False;self.A_floor=.01;self.chunk_size=8
  self.in_proj=nn.Linear(4,16,bias=False);self.out_proj=nn.Linear(4,4,bias=False);self.B_norm=nn.Identity();self.C_norm=nn.Identity();self.B_bias=nn.Parameter(torch.zeros(1,1,2));self.C_bias=nn.Parameter(torch.zeros(1,1,2));self.dt_bias=nn.Parameter(torch.zeros(1));self.D=nn.Parameter(torch.ones(1,4))
 def forward(self,u): return manual(self,u)
def manual(c,u):
 b,t,_=u.shape; z,x,k,q,dt,a,tr,ang=torch.split(c.in_proj(u),[4,4,2,2,1,1,1,1],-1);z=z.view(b,t,1,4);x=x.view(b,t,1,4);k=k.view(b,t,1,2);q=q.view(b,t,1,2);dt=torch.nn.functional.softplus(dt+c.dt_bias);a=-(a.clamp_min(0)+torch.reciprocal(1-a.clamp_max(0))).clamp_max(-c.A_floor)
 y=_kernel(Q=q,K=k,V=x,ADT=(a*dt).transpose(1,2),DT=dt.transpose(1,2),Trap=tr.transpose(1,2),Q_bias=c.C_bias.squeeze(1),K_bias=c.B_bias.squeeze(1),Angles=ang.unsqueeze(-2),D=c.D,Z=z,chunk_size=8,Input_States=None,return_final_states=False,cu_seqlens=None)
 return c.out_proj(y.flatten(-2))
class Block(nn.Module):
 def __init__(self): super().__init__();self.ssm=Core();self.norm1=nn.LayerNorm(4);self.norm2=nn.LayerNorm(4);self.ffn=nn.Linear(4,4);self.dropout=nn.Identity()
 def forward(self,x):
  x=x+self.dropout(self.ssm(self.norm1(x)))
  return x+self.dropout(self.ffn(self.norm2(x)))
class Model(nn.Module):
 def __init__(self,layers=2): super().__init__();self.config=SimpleNamespace(input_size=3);self.in_proj=nn.Linear(3,4);self.blocks=nn.ModuleList([Block() for _ in range(layers)]);self.final_norm=nn.LayerNorm(4);self.out_proj=nn.Linear(4,2)
 def forward(self,x):
  h=self.in_proj(x)
  for b in self.blocks:h=b(h)
  return self.out_proj(self.final_norm(h))
@pytest.fixture(autouse=True)
def kernel(monkeypatch): monkeypatch.setattr(ports,'_siso_kernel',_kernel)
@pytest.mark.parametrize('method',['state_offset','memba_causal'])
def test_install_zero_replay(method):
 torch.manual_seed(3);m=Model();x=torch.randn(2,6,3);base=m(x).detach();initial=copy.deepcopy(m.state_dict());r=ports.install_research_adapters(m,method,2)
 assert r['functional'] and torch.equal(m(x),base) and m.in_proj.A.dtype==x.dtype
 n=Model();n.load_state_dict(initial);ports.install_research_adapters(n,method,2);n.load_state_dict(m.state_dict());assert torch.equal(n(x),m(x))
def test_offset_gradient_formula_and_validation():
 m=Model(1);x=torch.randn(2,5,3);ports.install_research_adapters(m,'state_offset',2);a=m.blocks[0].ssm.research_adapter;c=torch.ones(1,2,1,2);a.U.data.fill_(1);a.V.data.fill_(2);assert torch.equal(a(c),torch.full((1,2,1,4),8.))
 before=m(x).detach();o=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],.1);m(x).square().mean().backward();assert a.U.grad.abs().sum()>0;o.step();assert not torch.equal(before,m(x))
 bad=Model();bad.blocks[0].ssm.is_mimo=True
 with pytest.raises(ValueError):ports.install_research_adapters(bad,'state_offset',2)
def test_membrane_causal_batch_reset():
 m=Model();x=torch.randn(2,6,3);ports.install_research_adapters(m,'memba_causal',2)
 for b in m.blocks:b.ssm.research_adapter.up.weight.data.fill_(.2)
 y=m(x);q=x.clone();q[:,4:]+=20;assert torch.equal(y[:,:4],m(q)[:,:4]);assert torch.equal(y[0],m(x[:1])[0])
 m.blocks[0].ssm.research_adapter.down.weight.data.add_(.1);assert not torch.equal(y,m(x))

def test_readout_angle_activation_and_quantization():
 c=torch.tensor([[[[1.,0.]],[[1.,0.]]]])
 bias=torch.zeros(1,2)
 angles=torch.full((1,2,1,1),.5)
 dt=torch.full((1,2,1),.25)
 q=ports.effective_readout(c,bias,angles,dt)
 increment=torch.tanh(torch.tensor(.5))*torch.pi*.25
 theta=torch.tensor([increment,2*increment]).to(torch.bfloat16).float()
 expected=torch.stack((torch.cos(theta),torch.sin(theta)),dim=-1).to(torch.bfloat16).float().reshape_as(q)
 assert torch.equal(q,expected)
