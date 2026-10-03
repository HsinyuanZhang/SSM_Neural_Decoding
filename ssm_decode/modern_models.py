"""Pure-PyTorch causal S4D and Mamba-2/SSD reference backbones.

The Mamba block follows the discrete SSD variables in the official
state-spaces/mamba ``ssd_minimal.py``: per-token ``dt, B, C`` and diagonal
``A`` update a head/state/channel state.  It is a reference implementation,
not an official CUDA kernel or Mamba-3 implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math, torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModernConfig:
    input_size:int; output_size:int; kind:str="s4d"; width:int=128; layers:int=2; state_size:int=16; dropout:float=.1

class S4DBlock(nn.Module):
    def __init__(self,w,n,drop):
        super().__init__(); self.norm=nn.LayerNorm(w); self.inp=nn.Linear(w,w*2); self.log_dt=nn.Parameter(torch.full((w,),-2.)); self.log_a=nn.Parameter(torch.linspace(-3.,1.,n).repeat(w,1)); self.omega=nn.Parameter(torch.linspace(0.,math.pi,n).repeat(w,1)); self.B=nn.Parameter(torch.randn(w,n)*.1); self.C=nn.Parameter(torch.randn(w,n)*.1); self.D=nn.Parameter(torch.ones(w)); self.out=nn.Linear(w,w); self.ffnorm=nn.LayerNorm(w); self.ff=nn.Sequential(nn.Linear(w,2*w),nn.SiLU(),nn.Dropout(drop),nn.Linear(2*w,w)); self.drop=nn.Dropout(drop)
    def initial_state(self,b,device): return torch.zeros(b,self.log_dt.numel(),self.log_a.shape[1],device=device,dtype=torch.complex64)
    def step(self,x,s):
        u,g=self.inp(self.norm(x)).chunk(2,-1); dt=F.softplus(self.log_dt); lam=-torch.exp(self.log_a)+1j*self.omega
        a=torch.exp(lam*dt[:,None]); b=(a-1)/lam*self.B; s=s*a.unsqueeze(0)+u.unsqueeze(-1).to(s.dtype)*b.unsqueeze(0)
        y=(s*self.C.unsqueeze(0)).sum(-1).real+self.D*u; x=x+self.drop(self.out(F.silu(g)*y)); return x+self.drop(self.ff(self.ffnorm(x))),s
    def sequence(self, x, need_state=False):
        """Zero-state causal FFT convolution; step() remains the streaming oracle."""
        B,T,W=x.shape; u,g=self.inp(self.norm(x)).chunk(2,-1); dt=F.softplus(self.log_dt); lam=-torch.exp(self.log_a)+1j*self.omega; a=torch.exp(lam*dt[:,None]); b=(a-1)/lam*self.B
        powers=a.unsqueeze(-1).pow(torch.arange(T,device=x.device))
        k=(self.C.unsqueeze(-1)*b.unsqueeze(-1)*powers).sum(1).real; k[:,0]+=self.D
        n=2*T; yf=torch.fft.irfft(torch.fft.rfft(u.transpose(1,2),n)*torch.fft.rfft(k,n),n)[...,:T].transpose(1,2)
        h=x+self.drop(self.out(F.silu(g)*yf)); h=h+self.drop(self.ff(self.ffnorm(h)))
        if not need_state:
            return h, None
        # Closed form final state: sum_t u_t b a**(T-1-t), no step loop.
        s = torch.einsum("btw,wn,wnt->bwn", u, b, powers.flip(-1))
        return h, s

class Mamba2Block(nn.Module):
    def __init__(self,w,n,drop):
        super().__init__(); self.w=w; self.heads=max(1,min(8,w//16)); self.p=w//self.heads; self.w=self.heads*self.p; self.n=n; self.norm=nn.LayerNorm(w); self.inp=nn.Linear(w,2*self.w+self.heads+2*self.heads*n); self.conv=nn.Conv1d(self.w,self.w,3,groups=self.w,bias=True); self.log_a=nn.Parameter(torch.zeros(self.heads)); self.D=nn.Parameter(torch.ones(self.heads,self.p)); self.out=nn.Linear(self.w,w); self.ffnorm=nn.LayerNorm(w); self.ff=nn.Sequential(nn.Linear(w,2*w),nn.SiLU(),nn.Dropout(drop),nn.Linear(2*w,w)); self.drop=nn.Dropout(drop)
    def initial_state(self,b,device): return (torch.zeros(b,self.heads,self.n,self.p,device=device),torch.zeros(b,self.w,2,device=device))
    def step(self,x,state):
        s,buf=state; q=self.inp(self.norm(x)); u,z,dt,B,C=torch.split(q,[self.w,self.w,self.heads,self.heads*self.n,self.heads*self.n],-1)
        # Explicit local depthwise causal convolution, with state carried across chunks.
        local=F.conv1d(torch.cat([buf,u.unsqueeze(-1)],-1),self.conv.weight,self.conv.bias,groups=self.w).squeeze(-1); buf=torch.cat([buf[:,:,1:],u.unsqueeze(-1)],-1)
        u=F.silu(local).view(-1,self.heads,self.p); dt=F.softplus(dt).unsqueeze(-1).unsqueeze(-1); B=B.view(-1,self.heads,self.n,1); C=C.view(-1,self.heads,self.n,1)
        a=torch.exp(-torch.exp(self.log_a).view(1,-1,1,1)*dt); s=a*s+dt*B*u.unsqueeze(-2)
        y=(C*s).sum(-2)+self.D.unsqueeze(0)*u; y=y.reshape(x.shape[0],self.w)
        x=x+self.drop(self.out(y*F.silu(z))); return x+self.drop(self.ff(self.ffnorm(x))),(s,buf)
    def sequence(self, x, need_state=False):
        """Vectorized local convolution plus diagonal SSD associative scan."""
        B,T,_=x.shape; q=self.inp(self.norm(x)); u,z,dt,Bv,Cv=torch.split(q,[self.w,self.w,self.heads,self.heads*self.n,self.heads*self.n],-1); u_raw=u
        local=F.conv1d(u.transpose(1,2),self.conv.weight,self.conv.bias,padding=2,groups=self.w)[...,:T].transpose(1,2); u=F.silu(local).view(B,T,self.heads,self.p)
        dt=F.softplus(dt).unsqueeze(-1).unsqueeze(-1); Bv=Bv.view(B,T,self.heads,self.n,1); Cv=Cv.view(B,T,self.heads,self.n,1); a=torch.exp(-torch.exp(self.log_a).view(1,1,-1,1,1)*dt)
        inp = dt * Bv * u.unsqueeze(-2)
        # Parallel inclusive scan of affine maps (a, inp): composing current
        # after previous gives (a*a_prev, inp + a*inp_prev). This avoids the
        # underflow-prone cumprod/divide identity for long or strongly decayed
        # sequences while retaining O(log T) Python launches.
        aa, ss = a, inp
        shift = 1
        while shift < T:
            current_a = aa
            prior_a = torch.cat([torch.ones_like(aa[:, :shift]), aa[:, :-shift]], dim=1)
            prior_s = torch.cat([torch.zeros_like(ss[:, :shift]), ss[:, :-shift]], dim=1)
            valid = torch.arange(T, device=x.device)[None, :, None, None, None] >= shift
            aa = torch.where(valid, current_a * prior_a, current_a)
            ss = torch.where(valid, ss + current_a * prior_s, ss)
            shift *= 2
        y=(Cv*ss).sum(-2)+self.D.view(1,1,self.heads,self.p)*u; h=x+self.drop(self.out(y.reshape(B,T,self.w)*F.silu(z))); h=h+self.drop(self.ff(self.ffnorm(h)))
        if not need_state:
            return h, None
        history = F.pad(u_raw, (0, 0, max(0, 2 - T), 0))[:, -2:]
        return h, (ss[:, -1], history.transpose(1, 2))

class ModernDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();
        if cfg.kind not in {"s4d","mamba2"}: raise ValueError("kind must be s4d or mamba2")
        self.config=cfg; self.input=nn.Linear(cfg.input_size,cfg.width); cls=S4DBlock if cfg.kind=="s4d" else Mamba2Block; self.blocks=nn.ModuleList([cls(cfg.width,cfg.state_size,cfg.dropout) for _ in range(cfg.layers)]); self.final_norm=nn.LayerNorm(cfg.width); self.readout=nn.Linear(cfg.width,cfg.output_size)
    def initial_state(self,batch,device=None):
        d=device or self.readout.weight.device; return [b.initial_state(batch,d) for b in self.blocks]
    def step(self,x,state=None):
        h=self.input(x); state=self.initial_state(x.shape[0],x.device) if state is None else state; nxt=[]
        for b,s in zip(self.blocks,state): h,s=b.step(h,s); nxt.append(s)
        return self.readout(self.final_norm(h)),nxt
    def forward(self,x,state=None,return_state=False):
        if x.ndim!=3: raise ValueError("x must be [B,T,C]")
        # The zero-initial training route is vectorized. Any supplied state uses
        # the exact streaming oracle so chunk continuation has identical semantics.
        if state is None:
            h=self.input(x); final=[]
            for b in self.blocks:
                h,s=b.sequence(h, need_state=return_state); final.append(s)
            out=self.readout(self.final_norm(h)); return (out,final) if return_state else out
        ys=[]
        for t in range(x.shape[1]): y,state=self.step(x[:,t],state);ys.append(y)
        out=torch.stack(ys,1) if ys else x.new_empty(x.shape[0],0,self.config.output_size); return (out,state) if return_state else out

def build_modern_model(config:ModernConfig|dict): return ModernDecoder(ModernConfig(**config) if isinstance(config,dict) else config)
