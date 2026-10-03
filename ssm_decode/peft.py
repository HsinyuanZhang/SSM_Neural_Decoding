"""Auditable PEFT wrappers for the project OfficialMamba3Decoder only."""
from __future__ import annotations
import json
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=4, alpha=None, rows=None):
        if rank <= 0: raise ValueError("rank must be positive")
        if alpha is not None and alpha <= 0: raise ValueError("alpha must be positive")
        super().__init__(); self.base=base; self.rank=rank; self.scale=(rank if alpha is None else alpha)/rank
        for p in base.parameters(): p.requires_grad=False
        selected=list(range(base.out_features)) if rows is None else list(rows)
        if not selected or len(set(selected)) != len(selected) or min(selected)<0 or max(selected)>=base.out_features: raise ValueError("invalid or duplicate selected rows")
        self.register_buffer("rows",torch.tensor(selected,device=base.weight.device,dtype=torch.long))
        self.A=nn.Parameter(base.weight.new_zeros(len(selected),rank)); self.B=nn.Parameter(base.weight.new_empty(rank,base.in_features)); nn.init.kaiming_uniform_(self.B,a=5**.5)
        self.merged=False
    def delta(self):
        out=self.base.weight.new_zeros(self.base.out_features,self.base.in_features);out.index_copy_(0,self.rows,(self.A@self.B)*self.scale);return out
    def forward(self,x): return self.base(x) if self.merged else self.base(x)+torch.nn.functional.linear(x,self.delta())
    def merge(self):
        if not self.merged:
            self.base.weight.data.add_(self.delta().data);self.merged=True
    def unmerge(self):
        if self.merged:
            self.base.weight.data.sub_(self.delta().data);self.merged=False

def _parent(model,path):
    bits=path.split('.'); obj=model
    for b in bits[:-1]: obj=obj[int(b)] if b.isdigit() else getattr(obj,b)
    return obj,bits[-1]

def _replace(model,path,wrapper):
    p,n=_parent(model,path); setattr(p,n,wrapper)

def _core_slices(core):
    inner=core.d_inner; bc=core.d_state*core.num_bc_heads*core.mimo_rank; h=core.nheads
    angle=getattr(core,"num_rope_angles",0); sizes=[inner,inner,bc,bc,h,h,h,angle]; names=["z","x","B","C","dt","A","trap","angle"];start=0;out={}
    for name,size in zip(names,sizes):out[name]=range(start,start+size);start+=size
    projection=core.in_proj.base if isinstance(core.in_proj,LoRALinear) else core.in_proj
    if start != projection.out_features: raise ValueError("official Mamba3 in_proj layout does not match live rows")
    return out

def exact_groups(model):
    groups={"top_io":["in_proj","out_proj","final_norm"],"core_io":[],"bc_rows":[],"dt_bias":[]}
    for i,b in enumerate(model.blocks):
        prefix=f"blocks.{i}.ssm"; groups["core_io"] += [prefix+".in_proj",prefix+".out_proj"];groups["bc_rows"] += [prefix+".in_proj:B",prefix+".in_proj:C"];groups["dt_bias"] += [prefix+".dt_bias"]
    return groups

def configure_peft(model,method,rank=4,alpha=None,**kwargs):
    if method not in {"none","io","lora","bc_lora","bc_dt","full"}: raise ValueError("unknown PEFT method")
    for p in model.parameters(): p.requires_grad=False
    groups=exact_groups(model); paths=[]
    if method=="full":
        for p in model.parameters():p.requires_grad=True
    elif method=="io":
        names=["in_proj","out_proj","final_norm"]+(["blocks.0.norm1"] if kwargs.get("norm1") else [])
        for n,p in model.named_parameters():
            if n.rsplit('.',1)[0] in names or n.startswith("final_norm."):p.requires_grad=True
    elif method in {"lora","bc_lora","bc_dt"}:
        targets=["in_proj","out_proj"]+([p for p in groups["core_io"] if not p.endswith(".in_proj")] if method in {"bc_lora","bc_dt"} else groups["core_io"])
        for path in targets:
            mod=dict(model.named_modules())[path]
            if isinstance(mod,nn.Linear): _replace(model,path,LoRALinear(mod,rank,alpha));paths.append(path)
        if method in {"bc_lora","bc_dt"}:
            for i,b in enumerate(model.blocks):
                path=f"blocks.{i}.ssm.in_proj"; mod=dict(model.named_modules())[path]; slices=_core_slices(b.ssm)
                rows=list(slices["B"])+list(slices["C"]); _replace(model,path,LoRALinear(mod,rank,alpha,rows));paths.append(path+":BC")
        if method=="bc_dt":
            for b in model.blocks:b.ssm.dt_bias.requires_grad=True
    layouts={f"blocks.{i}.ssm":{k:[v.start,v.stop] for k,v in _core_slices(b.ssm).items()} for i,b in enumerate(model.blocks)} if method in {"bc_lora","bc_dt"} else {}
    named=list(model.named_parameters(remove_duplicate=False)); ids=[id(p) for _,p in named];
    if len(ids)!=len(set(ids)): raise ValueError("parameter alias registration detected")
    intervals=[]
    for name,p in named:
        start=p.untyped_storage().data_ptr()+p.storage_offset()*p.element_size(); end=start+p.numel()*p.element_size()
        for other,lo,hi in intervals:
            if start < hi and lo < end: raise ValueError(f"parameter storage overlap: {name} and {other}")
        intervals.append((name,start,end))
    receipt={"schema":"mamba3_peft_v1","method":method,"rank":rank,"alpha":rank if alpha is None else alpha,"groups":groups,"lora_paths":paths,"bc_layouts":layouts,"trainable_paths":[n for n,p in named if p.requires_grad],"custom_bc_lora_not_sdlora":method in {"bc_lora","bc_dt"},"dt_bias_policy":"additive raw dt_bias; requires runner no-weight-decay group" if method=="bc_dt" else None}
    receipt["trainable_count"]=sum(p.numel() for p in model.parameters() if p.requires_grad);json.dumps(receipt);return receipt

def merged_state_dict(model):
    state={}
    for name,module in model.named_modules():
        if isinstance(module,LoRALinear):
            prefix=(name+".") if name else "";state[prefix+"weight"]=(module.base.weight+module.delta()).detach().clone()
            if module.base.bias is not None:state[prefix+"bias"]=module.base.bias.detach().clone()
    for key,value in model.state_dict().items():
        if ".base." not in key and not key.endswith(".A") and not key.endswith(".B") and not key.endswith(".rows"):
            state.setdefault(key,value.detach().clone())
    return state
