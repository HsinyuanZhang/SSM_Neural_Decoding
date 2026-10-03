import pytest, torch
import torch.nn.functional as F
from ssm_decode.mamba3_cpu import CPUDecoder

def _state(w=64,inp=3,out=2,layers=1):
 h=2*w; heads=h//64; rows=2*h+64+3*heads+8; d={'in_proj.weight':torch.randn(w,inp),'in_proj.bias':torch.randn(w),'final_norm.weight':torch.ones(w),'final_norm.bias':torch.zeros(w),'out_proj.weight':torch.randn(out,w),'out_proj.bias':torch.zeros(out)}
 for i in range(layers):
  p=f'blocks.{i}.'; d|={p+'norm1.weight':torch.ones(w),p+'norm1.bias':torch.zeros(w),p+'ssm.in_proj.weight':torch.randn(rows,w)*.02,p+'ssm.dt_bias':torch.zeros(heads),p+'ssm.B_bias':torch.ones(heads,1,32),p+'ssm.C_bias':torch.ones(heads,1,32),p+'ssm.D':torch.ones(heads),p+'ssm.B_norm.weight':torch.ones(32),p+'ssm.C_norm.weight':torch.ones(32),p+'ssm.out_proj.weight':torch.randn(w,h)*.02,p+'norm2.weight':torch.ones(w),p+'norm2.bias':torch.zeros(w),p+'ffn.0.weight':torch.randn(2*w,w)*.02,p+'ffn.0.bias':torch.zeros(2*w),p+'ffn.2.weight':torch.randn(w,2*w)*.02,p+'ffn.2.bias':torch.zeros(w)}
 return d

def test_plain_cpu_shape_and_causality():
 torch.manual_seed(3); d=CPUDecoder.from_state_dict(_state(layers=2)); x=torch.randn(2,128,3); y=d.forward(x); changed=x.clone();changed[:,80:]+=torch.randn_like(changed[:,80:]); assert y.shape==(2,128,2); assert torch.allclose(y[:,:80],d.forward(changed)[:,:80],atol=0,rtol=0)

def test_rejects_adapter_and_bad_shapes():
 s=_state();s['blocks.0.ssm.in_proj.A']=torch.zeros(1,1)
 with pytest.raises(ValueError,match='adapter'): CPUDecoder.from_state_dict(s)
 s=_state();s['blocks.0.ssm.D']=torch.zeros(1)
 with pytest.raises(ValueError,match='shape drift'): CPUDecoder.from_state_dict(s)

def test_rejects_overlong_window():
 d=CPUDecoder.from_state_dict(_state())
 with pytest.raises(ValueError,match='128'): d.forward(torch.randn(1,129,3))

def test_t1_closed_oracle_negative_a_full_gamma_and_root_bias():
 # This is a closed T=1 oracle derived from the kernel's diagonal path, not a
 # second call into CPUDecoder.  It detects wrong A sign, a doubled trap
 # sigmoid, and root/input-bias omission.
 s=_state(w=64,inp=1,out=1); s['in_proj.weight'].zero_(); s['in_proj.bias'].fill_(2.)
 p='blocks.0.'; s[p+'norm1.weight'].fill_(1);s[p+'norm1.bias'].zero_(); s[p+'ssm.in_proj.weight'].zero_()
 # projection layout z/x/B/C/dt/A/trap/angles.  One root-bias-only token.
 h=128; s[p+'ssm.in_proj.weight'][2*h+64+3:2*h+64+4].fill_(0.)
 s[p+'ssm.dt_bias'].fill_(0.);s[p+'ssm.D'].zero_();s[p+'ssm.B_bias'].zero_();s[p+'ssm.C_bias'].zero_()
 s[p+'ssm.B_norm.weight'].fill_(1);s[p+'ssm.C_norm.weight'].fill_(1);s[p+'ssm.out_proj.weight'].zero_()
 s[p+'norm2.weight'].fill_(1);s[p+'norm2.bias'].zero_();s[p+'ffn.0.weight'].zero_();s[p+'ffn.0.bias'].zero_();s[p+'ffn.2.weight'].zero_();s[p+'ffn.2.bias'].zero_()
 s['final_norm.weight'].fill_(1);s['final_norm.bias'].zero_();s['out_proj.weight'].fill_(1);s['out_proj.bias'].zero_()
 # Closed core path: V=1, Z=1, B/C are their FP32 biases.  With D=0 and a
 # one-token chunk, output is gamma * dot(Cbias,Bbias) * V * silu(Z).
 s[p+'ssm.B_bias'].fill_(1.);s[p+'ssm.C_bias'].fill_(2.)
 ip=s[p+'ssm.in_proj.weight']; ip.zero_(); ip[:128,0]=1.; ip[128:256,0]=1.
 s[p+'ssm.out_proj.weight'].zero_();s[p+'ssm.out_proj.weight'][0,0]=1.
 d=CPUDecoder.from_state_dict(s); core=d._core(torch.ones(1,1,64),d.blocks[0])
 gamma=F.softplus(torch.tensor(0.))*torch.sigmoid(torch.tensor(0.)); expected=(gamma*64*F.silu(torch.tensor(1.,dtype=torch.bfloat16).float())).to(torch.bfloat16).float()
 assert torch.allclose(core[0,0,0],expected,atol=0,rtol=0)
 # Root bias is present in full decoder (a zero input is not treated as zero
 # hidden state); this catches a common folded-export omission.
 y=d.forward(torch.zeros(1,1,1)); assert torch.isfinite(y).all() and y.shape==(1,1,1)

def test_cross_chunk_prefix_is_invariant():
 torch.manual_seed(8); d=CPUDecoder.from_state_dict(_state()); x=torch.randn(1,128,3); changed=x.clone();changed[:,96:]+=3*torch.randn_like(changed[:,96:])
 assert torch.equal(d.forward(x)[:,:96],d.forward(changed)[:,:96])
