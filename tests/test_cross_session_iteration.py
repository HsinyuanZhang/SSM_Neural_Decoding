import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from ssm_decode.data import Recording
import ssm_decode.cross_session_iteration as ci


class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ssm = nn.Linear(width, width)
    def forward(self, x):
        return x + torch.tanh(self.ssm(x))


class Toy(nn.Module):
    def __init__(self, inputs, outputs, width=4, layers=1, **kwargs):
        super().__init__()
        self.config = SimpleNamespace(input_size=inputs, output_size=outputs)
        self.in_proj = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([ToyBlock(width) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, outputs)
    def forward(self, x):
        z = self.in_proj(x)
        for block in self.blocks:
            z = block(z)
        return self.out_proj(self.final_norm(z))


def record(query_shift=0):
    rng = np.random.default_rng(44)
    x = rng.normal(size=(34*60, 3)).astype(np.float32)
    y = (x[:, :2]*0.4).astype(np.float32)
    y[33*60:] += query_shift
    return Recording('m1','held_in','toy',Path('unused.nwb'),x,y,np.zeros(len(x),bool),
                     np.ones(len(x),bool),tuple((i*60,(i+1)*60) for i in range(34)))


def make_fixture(monkeypatch, tmp_path):
    torch.manual_seed(8)
    source = Toy(3,2)
    checkpoint = tmp_path/'source.pt'
    torch.save({'state_dict':source.state_dict(),'args':{'width':4,'layers':1,'state_size':2,'dropout':0.}},checkpoint)
    np.savez(tmp_path/'normalizer.npz',x_mean=np.zeros(3,np.float32),x_std=np.ones(3,np.float32),
             y_mean=np.zeros(2,np.float32),y_std=np.ones(2,np.float32))
    current = {'record': record()}
    monkeypatch.setattr(ci,'build_mamba3_official',Toy)
    monkeypatch.setattr(ci,'_target',lambda *a,**k:(current['record'],{}))
    monkeypatch.setattr(ci.d,'_official_pin',lambda *a:None)
    monkeypatch.setattr(ci,'_data_hash',lambda target:'toy-data-hash')
    return checkpoint,current


def args(checkpoint, output, **updates):
    a = ci.parser().parse_args(['--task','m1','--pretrained',str(checkpoint),'--output',str(output),
       '--method','io','--device','cpu','--steps','3','--max-steps','6','--val-interval','1',
       '--context','50','--batch-size','2','--eval-batch-size','32','--lr','0.01','--defer-query'])
    for k,v in updates.items():
        setattr(a,k,v)
    return a


def test_interleaved_partition():
    bounds = [(i*50,(i+1)*50) for i in range(40)]
    tr,val,_ = ci.support_partition(bounds)
    assert len(tr)==26 and len(val)==7
    assert [a//50 for a,b in val] == list(ci.INTERLEAVED)


def test_recording_context_has_no_other_partition_labels():
    r = record()
    stats=(np.zeros(3,np.float32),np.ones(3,np.float32),np.zeros(2,np.float32),np.ones(2,np.float32))
    (x,y,m), = ci.partition_windows(r,[(300,360)],stats,'cpu','recording_causal_fixed_window')
    assert torch.equal(x[:300],torch.from_numpy(r.neural[:300]))
    assert torch.isnan(y[:300]).all() and torch.isnan(y[360:]).all()
    assert not m[:300].any() and not m[360:].any() and m[300:360].all()
    xx,yy,mm = ci.d._right_aligned([(x,y,m)],[(0,305)],128)
    assert torch.isnan(yy[0,:122]).all() and mm[0,-6:].all()


def test_query_label_perturbation_leaves_fitting_and_selection_exact(monkeypatch,tmp_path):
    checkpoint,current=make_fixture(monkeypatch,tmp_path)
    first=ci.run(args(checkpoint,tmp_path/'first'))
    current['record']=record(query_shift=999)
    second=ci.run(args(checkpoint,tmp_path/'second'))
    p=torch.load(first/'best.pt',weights_only=False)
    q=torch.load(second/'best.pt',weights_only=False)
    assert p['best_step']==q['best_step']
    assert all(torch.equal(p['state_dict'][k],q['state_dict'][k]) for k in p['state_dict'])
    assert not (first/'query_evaluation').exists()
    assert json.loads((first/'fit_result.json').read_text())['query_evaluated'] is False
    restored,_=ci.load_saved_model(first/'best.pt','cpu')
    reference=Toy(3,2);reference.load_state_dict(p['state_dict']);reference.eval()
    probe=torch.randn(2,50,3)
    assert torch.equal(restored(probe),reference(probe))
    current['record']=record()
    ci.evaluate_checkpoint(first/'best.pt',tmp_path/'eval_first','cpu')
    current['record']=record(query_shift=999)
    ci.evaluate_checkpoint(second/'best.pt',tmp_path/'eval_second','cpu')
    a=json.loads((tmp_path/'eval_first'/'metrics.json').read_text())
    b=json.loads((tmp_path/'eval_second'/'metrics.json').read_text())
    assert a['final']!=b['final']


def test_budget_extension_only_uses_prefix_validation(monkeypatch,tmp_path):
    checkpoint,_=make_fixture(monkeypatch,tmp_path)
    scores=iter([0.,0.1,0.2,0.3,0.4,0.5,0.6])
    monkeypatch.setattr(ci.d,'_val',lambda *a,**k:next(scores))
    path=ci.run(args(checkpoint,tmp_path/'extended'))
    result=json.loads((path/'fit_result.json').read_text())
    assert result['completed_steps']==6 and result['best_step']==6
    assert result['convergence_satisfied'] is False
    assert result['extensions']==[{'old_budget':3,'new_budget':6,'best_step':3,
                                   'reason':'best_step_at_least_80pct_budget'}]


def test_full_stages_preserve_optimizer_state_and_respect_no_decay(monkeypatch,tmp_path):
    checkpoint,_=make_fixture(monkeypatch,tmp_path)
    model=Toy(3,2,layers=3)
    ci.configure_adaptation(model,'full')
    model.blocks[-1].ssm.bias._no_weight_decay=True
    opt=torch.optim.AdamW(ci._groups([p for p in model.parameters() if p.requires_grad],.001,.01))
    model(torch.randn(2,3,3)).square().mean().backward();opt.step()
    original_state=opt.state[model.in_proj.weight]
    assert ci._unfreeze(model,opt,200,.001,.01)
    assert opt.state[model.in_proj.weight] is original_state
    name_groups=ci._optimizer_receipt(model,opt,200)['groups']
    assert all(not p.requires_grad for b in model.blocks[:-1] for p in b.parameters())
    bias_group=next(g for g in name_groups if 'blocks.2.ssm.bias' in g['names'])
    assert bias_group['weight_decay']==0 and bias_group['lr']==.0005
    assert ci._unfreeze(model,opt,400,.001,.01)
    assert all(p.requires_grad for b in model.blocks for p in b.parameters())
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))
    assert opt.state[model.in_proj.weight] is original_state
