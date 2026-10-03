"""Train on support labels and select settings before query evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from . import debug_experiment as d
from .data import DEFAULT_ROOT, load_recording, source_target_plan
from .input_adaptation import configure_adaptation
from .mamba3_official import build_mamba3_official
from .session_normalization import fit_support_statistics, normalize_neural, strict_past_ema_normalize

INTERLEAVED = (4, 9, 14, 19, 24, 29, 32)
KEYS = ('x_mean', 'x_std', 'y_mean', 'y_std')


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_hashes():
    names = ('cross_session_iteration.py', 'session_normalization.py', 'input_adaptation.py',
             'debug_experiment.py', 'calibration.py', 'data.py', 'mamba3_official.py',
             'peft.py', 'models.py', 'modern_models.py')
    return {name: _file_hash(Path(__file__).parent/name) for name in names}


def _runtime_identity():
    official = d._official_pin('mamba3_official')
    return {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'official_expected': None if official is None else official['expected_official_commit'],
            'official_actual': None if official is None else official['actual_official_commit'],
            'triton': None if official is None or official['triton'] is None else official['triton']['version']}


def _data_hash(target):
    return _file_hash(target.path)


def _parameter_hashes(model):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def support_partition(bounds, split='interleaved', support_trials=33):
    usable = [tuple(b) for b in bounds if b[1]-b[0] >= 50]
    if support_trials < 5 or len(usable) <= support_trials:
        raise ValueError('support requires at least five trials and a separate query')
    support = usable[:support_trials]
    count = math.ceil(support_trials/5)
    if split == 'interleaved':
        indices = set(range(4, support_trials, 5))
        if len(indices) < count:
            indices.add(support_trials-1)
    elif split == 'blocked':
        indices = set(range(support_trials-count, support_trials))
    else:
        raise ValueError('unknown validation split')
    return ([b for i, b in enumerate(support) if i not in indices],
            [b for i, b in enumerate(support) if i in indices], support)


def query_behavior_nan(behavior, query_bounds=None, support_bounds=()):
    masked = np.full_like(behavior, np.nan)
    for a, b in support_bounds:
        masked[a:b] = behavior[a:b]
    return masked


def partition_windows(target, bounds, stats, device, policy):
    if policy == 'trial_causal_fixed_window':
        return d._windows([target], [bounds], stats, device)
    if policy != 'recording_causal_fixed_window':
        raise ValueError('unknown context policy')
    x = normalize_neural(target.neural, stats)
    y = np.full_like(target.behavior, np.nan)
    mask = np.zeros(len(x), dtype=bool)
    for a, b in bounds:
        y[a:b] = (target.behavior[a:b]-stats[2])/stats[3]
        mask[a:b] = target.eval_mask[a:b]
    return [(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
             torch.from_numpy(mask).to(device))]


def _base_model(cfg, inputs, outputs, device):
    return build_mamba3_official(inputs, outputs, width=cfg['width'], layers=cfg['layers'],
                                state_size=cfg['state_size'], dropout=cfg['dropout']).to(device)


def load_saved_model(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload['iteration_replay']
    model = _base_model(meta['base_cfg'], meta['input_size'], meta['output_size'], device)
    configure_adaptation(model, **meta['adaptation'])
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval(), payload


def _groups(parameters, lr, decay):
    parameters = list(parameters)
    groups = []
    for no_decay in (False, True):
        selected = [p for p in parameters if bool(getattr(p, '_no_weight_decay', False)) == no_decay]
        if selected:
            groups.append({'params': selected, 'lr': lr, 'weight_decay': 0.0 if no_decay else decay})
    return groups


def _optimizer_receipt(model, optimizer, step):
    names = {id(p): n for n, p in model.named_parameters()}
    return {'step': step, 'groups': [{'names': [names[id(p)] for p in g['params']],
            'count': sum(p.numel() for p in g['params']), 'lr': g['lr'],
            'weight_decay': g['weight_decay']} for g in optimizer.param_groups]}


def _frozen(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}


def _check_frozen(model, snapshot, step):
    changed = [n for n, p in model.named_parameters() if n in snapshot
               and not torch.equal(p.detach().cpu(), snapshot[n])]
    if changed:
        raise AssertionError(f'frozen tensors changed at step {step}: {changed[:4]}')
    return {'step': step, 'status': 'passed', 'count': len(snapshot)}


def _unfreeze(model, optimizer, step, lr, decay):
    blocks = list(model.blocks)
    wanted = list(blocks[-1].parameters()) if blocks and step == 200 else []
    if step == 400:
        wanted = [p for b in blocks for p in b.parameters()]
    added = [p for p in wanted if not p.requires_grad]
    for p in added:
        p.requires_grad_(True)
    for group in _groups(added, lr*0.5, decay):
        optimizer.add_param_group(group)
    return bool(added)


def _target(task, root, session=None):
    plan = source_target_plan(task, root=Path(root))
    selected = session or plan['cross_session_local_dev']['target_session']
    return load_recording(task, 'held_in', selected, root=Path(root)), plan


def _cache_predictions(model, target, stats, args, device, ema_half_life=None):
    usable = [b for b in target.trial_bounds if b[1]-b[0] >= 50]
    support, query = usable[:args.support_trials], usable[args.query_cutoff_trials:]
    if not query or max(b for a, b in support) > query[0][0]:
        raise ValueError('support and query must be temporally disjoint')
    support_indices = np.concatenate([np.arange(a, b) for a, b in support])
    allvalid = np.concatenate([np.arange(a, b) for a, b in query])
    legacy = np.concatenate([np.arange(a+49, b) for a, b in query])
    support_indices = support_indices[target.eval_mask[support_indices]]
    allvalid, legacy = allvalid[target.eval_mask[allvalid]], legacy[target.eval_mask[legacy]]
    neural = normalize_neural(target.neural, stats)
    ema_receipt = None
    if ema_half_life is not None:
        if args.normalization != 'support':
            raise ValueError('EMA requires support initialization')
        neural, ema_receipt = strict_past_ema_normalize(target.neural, support, half_life=ema_half_life)
    needed = np.unique(np.r_[support_indices, allvalid, legacy])
    bounds = ((0, len(neural)),) if args.policy == 'recording_causal_fixed_window' else target.trial_bounds
    prediction = d._predict_endpoints(model, neural, bounds, needed, device, args.context, batch=args.eval_batch_size)
    lookup = {int(v): i for i, v in enumerate(needed)}
    def take(indices):
        return prediction[[lookup[int(i)] for i in indices]]
    return {'support_indices': support_indices, 'support_prediction': take(support_indices),
            'support_truth_normalized': (target.behavior[support_indices]-stats[2])/stats[3],
            'legacy_indices': legacy, 'legacy_prediction': take(legacy),
            'legacy_truth_physical': target.behavior[legacy].copy(),
            'all_valid_indices': allvalid, 'all_valid_prediction': take(allvalid),
            'all_valid_truth_physical': target.behavior[allvalid].copy()}, ema_receipt


def evaluate_checkpoint(checkpoint, output, device='cuda:0', *, ema_half_life=None):
    torch.set_num_threads(1)
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model, payload = load_saved_model(checkpoint, device)
    args = argparse.Namespace(**payload['args'])
    if payload['code_hashes'] != _code_hashes():
        raise RuntimeError('replay implementation bytes differ from fitted bytes')
    stats = tuple(np.asarray(payload['normalizer'][k], dtype=np.float32) for k in KEYS)
    target, plan = _target(args.task, args.data_root, args.target_session)
    if _data_hash(target) != payload['target_data_sha256'] or _runtime_identity() != payload['runtime_identity']:
        raise RuntimeError('replay data bytes or official runtime differ from fitted provenance')
    cache, ema = _cache_predictions(model, target, stats, args, device, ema_half_life)
    zero, ridge = d._score_target(cache, stats, 'zero'), d._score_target(cache, stats, 'ridge')
    np.savez(out/'predictions.npz', legacy_indices=cache['legacy_indices'], allvalid_indices=cache['all_valid_indices'],
             truth_legacy=cache['legacy_truth_physical'], truth_allvalid=cache['all_valid_truth_physical'],
             support_indices=cache['support_indices'], support_truth_normalized=cache['support_truth_normalized'],
             support_prediction=cache['support_prediction'], zero_legacy=zero['legacy']['prediction_physical'],
             ridge_legacy=ridge['legacy']['prediction_physical'], zero_allvalid=zero['all_valid']['prediction_physical'],
             ridge_allvalid=ridge['all_valid']['prediction_physical'])
    for result in (zero, ridge):
        for cohort in ('legacy', 'all_valid'):
            for field in ('indices', 'truth_physical', 'prediction_physical'):
                result[cohort].pop(field)
    result = {'status': 'completed', 'task': args.task, 'method': args.method, 'seed': args.seed,
              'lr': args.lr, 'rank': args.rank, 'lora_scope': args.lora_scope, 'normalization': args.normalization,
              'validation_split': args.validation_split, 'policy': args.policy, 'support_trials': args.support_trials,
              'query_cutoff_trials': args.query_cutoff_trials, 'checkpoint': str(checkpoint),
              'checkpoint_sha256': _file_hash(checkpoint), 'fit_result': str(Path(checkpoint).parent/'fit_result.json'),
              'best_step': payload['best_step'], 'receipt': payload['receipt'], 'final': {'zero': zero, 'ridge': ridge},
              'ema': ema, 'code_hashes': payload['code_hashes'], 'datamanifest': plan, 'target_session': target.session,
              'held_out_accessed': False, 'query_used_for_selection': False,
              'cohort_index_hashes': {k: hashlib.sha256(cache[k].tobytes()).hexdigest()
                                      for k in ('support_indices', 'legacy_indices', 'all_valid_indices')},
              'cohort_truth_hashes': {k: hashlib.sha256(cache[k].tobytes()).hexdigest()
                                     for k in ('support_truth_normalized', 'legacy_truth_physical', 'all_valid_truth_physical')},
              'runtime_identity': payload['runtime_identity'], 'target_data_sha256': payload['target_data_sha256']}
    _write(out/'metrics.json', result)
    return out


def run(args):
    torch.set_num_threads(1)
    if not 0 <= args.steps <= args.max_steps or args.val_interval < 1:
        raise ValueError('invalid training budget')
    if args.query_cutoff_trials < args.support_trials:
        raise ValueError('query cutoff cannot precede support cutoff')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    d._seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    hashes = _code_hashes()
    _write(out/'code_hashes_start.json', hashes)
    source_hash = _file_hash(args.pretrained)
    source_normalizer_hash = _file_hash(Path(args.pretrained).parent/'normalizer.npz')
    source = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = source['args']
    with np.load(Path(args.pretrained).parent/'normalizer.npz') as z:
        source_stats = tuple(z[k].astype(np.float32) for k in KEYS)
    target, plan = _target(args.task, args.data_root, args.target_session)
    target_data_hash, runtime = _data_hash(target), _runtime_identity()
    train_bounds, val_bounds, support = support_partition(target.trial_bounds, args.validation_split, args.support_trials)
    query = [b for b in target.trial_bounds if b[1]-b[0] >= 50][args.query_cutoff_trials:]
    if not query or max(b for a, b in support) > query[0][0]:
        raise ValueError('support and query are not temporally disjoint')
    fit_target = replace(target, behavior=query_behavior_nan(target.behavior, query, support))
    assert all(np.isnan(fit_target.behavior[a:b]).all() for a, b in query)
    stats = fit_support_statistics(target.neural, support, source_stats) if args.normalization == 'support' else source_stats
    np.savez(out/'normalizer.npz', **dict(zip(KEYS, stats)))
    model = _base_model(cfg, target.neural.shape[1], target.behavior.shape[1], device)
    model.load_state_dict(source['state_dict'], strict=True)
    base_count = sum(p.numel() for p in model.parameters())
    adaptation = {'method': args.method, 'rank': args.rank, 'alpha': args.alpha,
                  'lora_scope': args.lora_scope, 'train_input_bias': args.train_input_bias}
    receipt = configure_adaptation(model, **adaptation)
    ui_hash = None
    if args.ui_checkpoint:
        if args.method != 'full':
            raise ValueError('UI initialization requires full fine-tuning')
        ui = torch.load(args.ui_checkpoint, map_location=device, weights_only=False)
        for key in ('task','seed','normalization','validation_split','support_trials','query_cutoff_trials','policy','context','target_session'):
            if ui['args'][key] != getattr(args, key):
                raise ValueError(f'UI contract differs: {key}')
        if ui['args']['method'] != 'io' or any(not np.array_equal(ui['normalizer'][k], s) for k, s in zip(KEYS, stats)):
            raise ValueError('UI weights or statistics differ')
        model.load_state_dict(ui['state_dict'], strict=True)
        ui_hash = _file_hash(args.ui_checkpoint)
    train = partition_windows(fit_target, train_bounds, stats, device, args.policy)
    valid = partition_windows(fit_target, val_bounds, stats, device, args.policy)
    eligible = d._eligible(train)
    selection, selection_hash = d._validation_selection(valid, args.context, 10**9)
    d._seed(args.seed)
    initial_hashes, frozen = _parameter_hashes(model), _frozen(model)
    audits, stages, logs, extensions = [], [], [], []
    replay = {'base_cfg': cfg, 'input_size': target.neural.shape[1], 'output_size': target.behavior.shape[1], 'adaptation': adaptation}
    def save(path, step):
        torch.save({'state_dict': model.state_dict(), 'best_step': step, 'args': vars(args), 'iteration_replay': replay,
                    'normalizer': dict(zip(KEYS, stats)), 'receipt': receipt, 'code_hashes': hashes,
                    'target_data_sha256': target_data_hash, 'runtime_identity': runtime}, path)
    started = time.perf_counter()
    best = d._val(model, valid, args.context, selection, args.eval_batch_size, stats)
    if not np.isfinite(best):
        raise FloatingPointError('nonfinite initial validation')
    best_step = step = 0
    save(out/'best.pt', 0)
    logs.append({'step': 0, 'prefix_val_r2': best})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(_groups([p for p in model.parameters() if p.requires_grad], args.lr, args.weight_decay)) if trainable else None
    if optimizer:
        stages.append(_optimizer_receipt(model, optimizer, 0))
    budget = args.steps if optimizer else 0
    while step < budget:
        step += 1
        if args.method == 'full' and step in (200, 400):
            audits.append(_check_frozen(model, frozen, step))
            if _unfreeze(model, optimizer, step, args.lr, args.weight_decay):
                stages.append(_optimizer_receipt(model, optimizer, step))
                frozen = _frozen(model)
        model.train()
        x, y, mask = d._sample_batch(train, eligible, args.context, args.batch_size)
        prediction, truth = d._forward(model, x)[:, args.context//2:], y[:, args.context//2:]
        good = mask[:, args.context//2:] & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(prediction).all():
            raise FloatingPointError('invalid training batch')
        optimizer.zero_grad(set_to_none=True)
        loss = (prediction[good]-truth[good]).square().mean()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        optimizer.step()
        event = {'step': step, 'loss': float(loss.detach()), 'gradient_norm': grad}
        if step % args.val_interval == 0 or step == budget:
            score = d._val(model, valid, args.context, selection, args.eval_batch_size, stats)
            if not np.isfinite(score):
                raise FloatingPointError('nonfinite validation')
            event['prefix_val_r2'] = score
            if score > best:
                best, best_step = score, step
                save(out/'best.pt', step)
        logs.append(event)
        if step == budget and best_step >= 0.8*budget and budget < args.max_steps:
            updated = min(args.max_steps, budget*2)
            extensions.append({'old_budget': budget, 'new_budget': updated, 'best_step': best_step,
                               'reason': 'best_step_at_least_80pct_budget'})
            budget = updated
    audits.append(_check_frozen(model, frozen, step))
    save(out/'last.pt', step)
    last_hashes = _parameter_hashes(model)
    selected = torch.load(out/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['state_dict'])
    best_hashes = _parameter_hashes(model)
    if hashes != _code_hashes():
        raise RuntimeError('implementation changed during fit')
    if source_hash != _file_hash(args.pretrained) or source_normalizer_hash != _file_hash(Path(args.pretrained).parent/'normalizer.npz'):
        raise RuntimeError('source checkpoint or normalizer changed during fit')
    if target_data_hash != _data_hash(target):
        raise RuntimeError('target data changed during fit')
    manifest = {'args': vars(args), 'plan': plan, 'target_session': target.session,
        'source_checkpoint_sha256': source_hash, 'source_normalizer_sha256': source_normalizer_hash,
        'ui_checkpoint_sha256': ui_hash, 'code_hashes': hashes, 'normalizer_sha256': _file_hash(out/'normalizer.npz'), 'receipt': receipt,
        'train_bounds': train_bounds, 'val_bounds': val_bounds, 'support_bounds': support, 'query_bounds': query,
        'prefix_statistics_scope': 'all support neural only' if args.normalization == 'support' else 'source train only',
        'prefix_before_query_verified': True, 'query_labels_masked': True, 'query_used_for_selection': False,
        'prefix_val_indices_sha256': selection_hash, 'optimizer_stages': stages, 'frozen_audits': audits,
        'initial_parameter_hashes': initial_hashes, 'best_parameter_hashes': best_hashes, 'last_parameter_hashes': last_hashes,
        'extensions': extensions, 'official': d._official_pin('mamba3_official'),
        'target_data_sha256': target_data_hash, 'runtime_identity': runtime,
        'changed_parameter_names': {'best': [n for n in initial_hashes if initial_hashes[n] != best_hashes[n]],
                                    'last': [n for n in initial_hashes if initial_hashes[n] != last_hashes[n]]},
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda, 'held_out_accessed': False}
    _write(out/'manifest.json', manifest)
    _write(out/'train_log.json', logs)
    result = {'status': 'completed', 'task': args.task, 'method': args.method, 'seed': args.seed, 'lr': args.lr,
        'best_prefix_val_r2': best, 'best_step': best_step, 'completed_steps': step,
        'convergence_satisfied': not optimizer or best_step < 0.8*budget,
        'selection_admissible': not optimizer or best_step < 0.8*budget, 'final_budget': budget, 'extensions': extensions,
        'receipt': receipt, 'base_parameters': base_count, 'initial_trainable_params': trainable,
        'peak_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()), 'fit_seconds': time.perf_counter()-started,
        'code_hashes': hashes, 'best_checkpoint_sha256': _file_hash(out/'best.pt'), 'last_checkpoint_sha256': _file_hash(out/'last.pt'),
        'source_checkpoint_sha256': source_hash, 'source_normalizer_sha256': source_normalizer_hash,
        'target_data_sha256': target_data_hash,
        'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
        'query_evaluated': False, 'query_used_for_selection': False}
    _write(out/'fit_result.json', result)
    if not args.defer_query:
        evaluate_checkpoint(out/'best.pt', out/'query_evaluation', device)
    return out


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['m1','m2'])
    p.add_argument('--pretrained')
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=['none','io','lora','affine','affine_lora','offset_rotated','offset_original','full'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--max-steps', type=int, default=8000)
    p.add_argument('--val-interval', type=int, default=100)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--support-trials', type=int, default=33)
    p.add_argument('--query-cutoff-trials', type=int, default=33)
    p.add_argument('--validation-split', choices=['blocked','interleaved'], default='interleaved')
    p.add_argument('--normalization', choices=['source','support'], default='support')
    p.add_argument('--policy', choices=['trial_causal_fixed_window','recording_causal_fixed_window'], default='trial_causal_fixed_window')
    p.add_argument('--lora-scope', choices=['all','input','root','core'], default='all')
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--alpha', type=float, default=4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--train-input-bias', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--defer-query', action='store_true')
    p.add_argument('--target-session')
    p.add_argument('--ui-checkpoint')
    p.add_argument('--data-root', default=str(DEFAULT_ROOT))
    p.add_argument('--evaluate-checkpoint')
    p.add_argument('--ema-half-life', type=float)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.evaluate_checkpoint:
        print(evaluate_checkpoint(args.evaluate_checkpoint, args.output, args.device, ema_half_life=args.ema_half_life))
    else:
        if not all((args.task, args.method, args.pretrained)):
            raise ValueError('training requires task, method, and pretrained')
        print(run(args))


if __name__ == '__main__':
    main()
