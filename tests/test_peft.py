import torch
from torch import nn
from ssm_decode.peft import configure_peft,merged_state_dict
class Core(nn.Module):
 def __init__(self):super().__init__();self.d_inner=4;self.d_state=2;self.num_bc_heads=1;self.mimo_rank=1;self.nheads=1;self.in_proj=nn.Linear(4,15,bias=False);self.out_proj=nn.Linear(4,4);self.dt_bias=nn.Parameter(torch.ones(1))
 def forward(self,x):
  q=self.in_proj(x); return self.out_proj(q[...,:4]+q[...,8:10].sum(-1,keepdim=True)+q[...,10:12].sum(-1,keepdim=True))
class Block(nn.Module):
 def __init__(self):super().__init__();self.ssm=Core();self.norm1=nn.LayerNorm(4)
 def forward(self,x):return self.ssm(self.norm1(x))
class Model(nn.Module):
 def __init__(self):super().__init__();self.in_proj=nn.Linear(3,4);self.blocks=nn.ModuleList([Block()]);self.final_norm=nn.LayerNorm(4);self.out_proj=nn.Linear(4,2)
 def forward(self,x):return self.out_proj(self.final_norm(self.blocks[0](self.in_proj(x))))
def test_lora_zero_merge_and_grad():
 torch.manual_seed(0);m=Model();x=torch.randn(2,3);base=m(x).detach();r=configure_peft(m,'bc_dt');assert torch.equal(base,m(x));bc=m.blocks[0].ssm.in_proj;opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.1,weight_decay=.1);m(x).sum().backward();assert bc.A.grad.abs().sum()>0;opt.step();delta=bc.delta();assert delta[8:10].abs().sum()>0 and delta[10:12].abs().sum()>0 and delta[:8].abs().sum()==0 and delta[12:].abs().sum()==0;export=merged_state_dict(m);assert 'blocks.0.ssm.in_proj.weight' in export and not any(k.endswith('.A') for k in export);assert r['trainable_count']>0
def test_io_is_exact_not_internal():
 m=Model();r=configure_peft(m,'io');assert any(n.startswith('in_proj') for n in r['trainable_paths']);assert not any('blocks.0.ssm.in_proj' in n for n in r['trainable_paths'])
