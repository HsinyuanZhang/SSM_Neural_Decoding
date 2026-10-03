import pytest
import torch
from torch import nn

import ssm_decode.mamba3_cpu_lora as runtime
from ssm_decode.mamba3_cpu import CPUDecoder as PlainCPUDecoder
from ssm_decode.mamba3_cpu_lora import CPUDecoder
from ssm_decode.peft import LoRALinear


def _state(w=64, inp=3, out=2, layers=1):
    h = 2*w; heads = h//64; rows = 2*h + 64 + 3*heads + 8
    d = {'in_proj.weight':torch.randn(w,inp), 'in_proj.bias':torch.randn(w),
         'final_norm.weight':torch.ones(w), 'final_norm.bias':torch.zeros(w),
         'out_proj.weight':torch.randn(out,w), 'out_proj.bias':torch.zeros(out)}
    for i in range(layers):
        p = f'blocks.{i}.'
        d |= {p+'norm1.weight':torch.ones(w),p+'norm1.bias':torch.zeros(w),
              p+'ssm.in_proj.weight':torch.randn(rows,w)*.02,p+'ssm.dt_bias':torch.zeros(heads),
              p+'ssm.B_bias':torch.ones(heads,1,32),p+'ssm.C_bias':torch.ones(heads,1,32),p+'ssm.D':torch.ones(heads),
              p+'ssm.B_norm.weight':torch.ones(32),p+'ssm.C_norm.weight':torch.ones(32),p+'ssm.out_proj.weight':torch.randn(w,h)*.02,
              p+'norm2.weight':torch.ones(w),p+'norm2.bias':torch.zeros(w),p+'ffn.0.weight':torch.randn(2*w,w)*.02,
              p+'ffn.0.bias':torch.zeros(2*w),p+'ffn.2.weight':torch.randn(w,2*w)*.02,p+'ffn.2.bias':torch.zeros(w)}
    return d


def _linear_paths(layers):
    result = ['in_proj', 'out_proj']
    for i in range(layers):
        p = f'blocks.{i}.'
        result += [p+'ssm.in_proj', p+'ssm.out_proj', p+'ffn.0', p+'ffn.2']
    return result


def _runtime_line(decoder, path):
    if path == 'in_proj': return decoder.in_proj
    if path == 'out_proj': return decoder.out_proj
    pieces = path.split('.')
    block = decoder.blocks[int(pieces[1])]
    if path.endswith('ssm.in_proj'): return block.ip
    if path.endswith('ssm.out_proj'): return block.op
    if path.endswith('ffn.0'): return block.ff1
    if path.endswith('ffn.2'): return block.ff2
    raise AssertionError(path)


def _unmerged(state, *, rank=2, alpha=2, selected_rows=None):
    """Convert every CPU linear path to literal peft.LoRALinear state_dict keys."""
    result = {key:value.clone() for key,value in state.items()}
    wrappers = {}
    layers = len({key.split('.')[1] for key in state if key.startswith('blocks.')})
    for path in _linear_paths(layers):
        weight = result.pop(path+'.weight')
        bias = result.pop(path+'.bias', None)
        base = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        with torch.no_grad():
            base.weight.copy_(weight)
            if bias is not None:
                base.bias.copy_(bias)
        rows = None if selected_rows is None else selected_rows.get(path)
        wrapper = LoRALinear(base, rank=rank, alpha=alpha, rows=rows)
        if wrapper.base.bias is not None:
            # The state format must retain a PEFT trainable base.bias value.
            wrapper.base.bias.requires_grad_(True)
            with torch.no_grad(): wrapper.base.bias.add_(.125)
        with torch.no_grad():
            wrapper.A.copy_(torch.randn_like(wrapper.A) * .02)
            wrapper.B.copy_(torch.randn_like(wrapper.B) * .02)
        result.update({path+'.'+key:value.detach().clone() for key,value in wrapper.state_dict().items()})
        wrappers[path] = wrapper.eval()
    return result, wrappers


def _merged_state(plain, wrappers):
    result = {key:value.clone() for key,value in plain.items()}
    for path, wrapper in wrappers.items():
        result[path+'.weight'] = (wrapper.base.weight + wrapper.delta()).detach().clone()
        if wrapper.base.bias is not None:
            result[path+'.bias'] = wrapper.base.bias.detach().clone()
    return result


def test_plain_state_matches_existing_runtime_without_regression():
    torch.manual_seed(11)
    state = _state(layers=2)
    x = torch.randn(2, 37, 3)
    assert torch.equal(CPUDecoder.from_state_dict(state).forward(x), PlainCPUDecoder.from_state_dict(state).forward(x))


def test_unmerged_lora_matches_peft_for_every_linear_path_and_full_forward(monkeypatch):
    torch.manual_seed(12)
    plain = _state(layers=1)
    # Sparse rows exercise the serialised `rows` mapping in addition to full paths.
    selected = {'blocks.0.ffn.0': [0, 17, 91], 'in_proj': [0, 9, 63]}
    unmerged, wrappers = _unmerged(plain, rank=2, alpha=2, selected_rows=selected)
    decoder = CPUDecoder.from_state_dict(unmerged)
    assert all(wrapper.scale == 1 for wrapper in wrappers.values())
    assert all(_runtime_line(decoder, path).rank == 2 for path in wrappers)

    for path, wrapper in wrappers.items():
        x = torch.randn(2, 5, wrapper.base.in_features)
        assert torch.allclose(runtime._linear(x, _runtime_line(decoder, path)), wrapper(x), atol=1e-6, rtol=1e-6), path

    # The CPU executor must issue two linear projections, retaining base.bias on
    # the base call and never folding the matrices before the call.
    calls, original = [], runtime.F.linear
    def spy(x, weight, bias=None):
        calls.append((weight, bias))
        return original(x, weight, bias)
    monkeypatch.setattr(runtime.F, 'linear', spy)
    root = _runtime_line(decoder, 'in_proj')
    _ = runtime._linear(torch.randn(1, 3, root.weight.shape[1]), root)
    assert len(calls) == 2
    assert torch.equal(calls[0][0], root.weight) and calls[0][1] is not None
    assert torch.equal(calls[1][0], root.delta) and calls[1][1] is None

    # A separately materialized test oracle is allowed here; production runtime
    # keeps the state unmerged as proven above.
    merged = CPUDecoder.from_state_dict(_merged_state(plain, wrappers))
    x = torch.randn(2, 19, 3)
    assert torch.allclose(decoder.forward(x), merged.forward(x), atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d.pop('in_proj.B'), 'incomplete unmerged'),
    (lambda d: d.__setitem__('in_proj.rows', torch.tensor([0., 1.])), 'rows'),
    (lambda d: d.__setitem__('in_proj.rows', torch.tensor([0, 0], dtype=torch.int64)), 'rows'),
    (lambda d: d.__setitem__('in_proj.rows', torch.tensor([0, 64], dtype=torch.int64)), 'rows'),
    (lambda d: d.__setitem__('in_proj.A', torch.zeros(64, 3)), 'rank/geometry'),
    (lambda d: d.__setitem__('in_proj.A', torch.zeros(64, 2, dtype=torch.int64)), 'floating'),
    (lambda d: d.__setitem__('in_proj.B', torch.empty(2, 3, device='meta')), 'CPU tensor'),
    (lambda d: d.__setitem__('in_proj.base.weight', torch.full((64, 3), float('nan'))), 'nonfinite'),
    (lambda d: d.__setitem__('in_proj.weight', torch.zeros(64, 3)), 'mixed plain'),
    (lambda d: d.__setitem__('blocks.0.ssm.in_proj.evil', torch.tensor(1.)), 'unsupported state keys'),
])
def test_rejects_malformed_or_malicious_unmerged_lora_state(mutate, message):
    torch.manual_seed(13)
    state, _ = _unmerged(_state())
    mutate(state)
    with pytest.raises(ValueError, match=message):
        CPUDecoder.from_state_dict(state)


def test_rejects_alpha_scale_drift_and_plain_unsupported_dtypes():
    torch.manual_seed(14)
    state, _ = _unmerged(_state(), rank=2, alpha=2)
    with pytest.raises(ValueError, match='alpha/rank'):
        CPUDecoder.from_state_dict(state, lora_alpha_over_rank=.5)
    plain = _state()
    plain['in_proj.weight'] = torch.ones(64, 3, dtype=torch.int64)
    with pytest.raises(ValueError, match='floating'):
        CPUDecoder.from_state_dict(plain)
    plain = _state()
    plain['state_offset.evil'] = torch.tensor(1.)
    with pytest.raises(ValueError, match='unsupported adapter'):
        CPUDecoder.from_state_dict(plain)
