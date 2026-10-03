"""Check real-source initialization, adapter gradients, and causal predictions."""
import argparse
import copy
import json
from pathlib import Path

import torch
from ssm_decode import peft_experiment as pe
from ssm_decode.sparse_state_tuning import begin_dense_selection, selection_from_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='m2')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(902)
    matrix = json.loads(Path('configs/peft_round1.json').read_text())
    source = torch.load(matrix['pretrained_by_task'][args.task], map_location='cpu', weights_only=False)
    cfg = source['args']
    inp, out = (64, 16) if args.task == 'm1' else (96, 2)
    def model():
        m = pe.build_mamba3_official(inp, out, width=cfg['width'], layers=cfg['layers'], state_size=cfg['state_size'], dropout=cfg['dropout']).cuda().eval()
        m.load_state_dict(source['state_dict'])
        return m
    original = model()
    x = torch.randn(2, 128, inp, device='cuda')
    with torch.no_grad():
        baseline = original(x)
    del original
    records = []
    for method in matrix['methods']:
        m = model()
        selections = None
        if method == 'sparse_sdt_m3':
            begin_dense_selection(m)
            optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.01)
            m(x).square().mean().backward()
            optimizer.step()
            selections = selection_from_warmup(m, 8)['selections']
        receipt = pe._configure_method(m, method, 4, 4., selections)
        with torch.no_grad():
            prediction = m(x)
        error = float((prediction-baseline).abs().max())
        if error != 0:
            raise AssertionError(f'{method}: startup max error {error}')
        gradient_names = []
        if method != 'none':
            optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.001)
            optimizer.zero_grad(set_to_none=True)
            m(x).square().mean().backward()
            gradient_names = [name for name,p in m.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()>0)]
            if not gradient_names:
                raise AssertionError(f'{method}: no nonzero gradients')
            if method in {'state_offset','memba_causal'}:
                for i in range(cfg['layers']):
                    path = f'blocks.{i}.ssm.research_adapter.'
                    if not any(n.startswith(path) for n in gradient_names):
                        raise AssertionError(f'{method}: inactive layer {i}')
            optimizer.step()
        changed = x.clone()
        changed[:,64:] += 3*torch.randn_like(changed[:,64:])
        with torch.no_grad():
            a, b = m(x), m(changed)
        causal_error = float((a[:,:64]-b[:,:64]).abs().max())
        if causal_error > 5e-5:
            raise AssertionError(f'{method}: future perturbation error {causal_error}')
        records.append({'method':method,'startup_max_abs':error,'future_perturb_max_abs':causal_error,
                        'trainable_params':receipt['trainable_count'],'nonzero_gradient_names':gradient_names})
        print(method, error, causal_error, flush=True)
        del m
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'task':args.task,'device':torch.cuda.get_device_name(0),'status':'passed','records':records},indent=2)+'\n')


if __name__ == '__main__':
    main()
