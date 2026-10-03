"""Train a session-aware SSM from source recordings without target labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import debug_experiment as d
from .cross_session_iteration import _code_hashes, _runtime_identity, _groups
from .data import DEFAULT_ROOT, load_recording, source_target_plan, sha256_file
from .mamba3_official import build_mamba3_official
from .session_frontend import SessionFrontendBank, drift_augment, fold_session_input


class SessionAwareDecoder(nn.Module):
    def __init__(self, base, bank):
        super().__init__()
        self.base, self.bank = base, bank
        self.config = base.config

    def forward(self, x, session_index):
        z = self.bank.add_embedding(self.base.in_proj(self.bank.forward_input(x, session_index)), session_index)
        for block in self.base.blocks:
            z = block(z)
        return self.base.out_proj(self.base.final_norm(z))


def source_statistics(records, splits):
    """Use source training neural bins and eval-valid training labels only."""
    moments = []
    labels = []
    for record, (train, _) in zip(records, splits):
        neural = np.concatenate([record.neural[a:b] for a, b in train]).astype(np.float64)
        mean, std = neural.mean(0), neural.std(0)
        std[std < 1e-6] = 1
        moments.append((mean.astype(np.float32), std.astype(np.float32)))
        for a, b in train:
            labels.append(record.behavior[a:b][record.eval_mask[a:b]])
    labels = np.concatenate(labels).astype(np.float64)
    if not len(labels):
        raise ValueError('source training labels have no eval-valid bins')
    mean, std = labels.mean(0), labels.std(0)
    std[std < 1e-6] = 1
    return [(xm, xs, mean.astype(np.float32), std.astype(np.float32)) for xm, xs in moments]


def continuous_source_windows(records, bounds_by_record, statistics, device):
    windows = []
    for record, bounds, stats in zip(records, bounds_by_record, statistics):
        x = ((record.neural-stats[0])/stats[1]).astype(np.float32)
        y = np.full_like(record.behavior, np.nan)
        mask = np.zeros(len(x), dtype=bool)
        for a, b in bounds:
            y[a:b] = (record.behavior[a:b]-stats[2])/stats[3]
            mask[a:b] = record.eval_mask[a:b]
        windows.append(tuple(torch.from_numpy(value).to(device) for value in (x, y, mask)))
    return windows


def sample_balanced_batch(windows, eligible, context, batch_size):
    if any(not values for values in eligible):
        raise ValueError('every source session must have a valid training endpoint')
    sessions = np.random.randint(len(windows), size=batch_size)
    picks = [(int(session), random.choice(eligible[int(session)])) for session in sessions]
    x, y, mask = d._right_aligned(windows, picks, context)
    return x, y, mask, torch.as_tensor(sessions, device=x.device)


def validate_source(model, windows, selections, context, batch, stats):
    model.eval()
    predictions, truths = [], []
    with torch.inference_mode():
        for session, selection in enumerate(selections):
            for lo in range(0, len(selection), batch):
                x, y, mask = d._right_aligned([windows[session]], selection[lo:lo+batch], context)
                p = model(x, session)[:, -1]
                good = mask[:, -1] & torch.isfinite(y[:, -1]).all(-1)
                if not torch.isfinite(p).all():
                    raise FloatingPointError('source validation prediction is not finite')
                predictions.append(p[good])
                truths.append(y[:, -1][good])
        scale = torch.as_tensor(stats[3], device=predictions[0].device)
        offset = torch.as_tensor(stats[2], device=scale.device)
        return d._r2(torch.cat(truths)*scale+offset, torch.cat(predictions)*scale+offset)


def code_hashes():
    hashes = _code_hashes()
    for name in ('session_pretraining.py', 'session_frontend.py'):
        hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return hashes


def run(args):
    if args.steps < 1 or args.val_interval < 1 or args.context < 2:
        raise ValueError('invalid source training budget or context')
    torch.set_num_threads(1)
    d._seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    hashes = code_hashes()
    snapshot = out/'code_snapshot'
    snapshot.mkdir()
    for name in hashes:
        (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    plan = source_target_plan(args.task, root=Path(args.data_root))
    source_ids = plan['source_held_in_sessions']
    records = [load_recording(args.task, 'held_in', session, root=Path(args.data_root)) for session in source_ids]
    source_hashes = {record.session: sha256_file(record.path) for record in records}
    splits = [d._trial_split(record) for record in records]
    if any(not train or not valid for train, valid in splits):
        raise ValueError('every source session requires train and validation trials')
    statistics = source_statistics(records, splits)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    train = continuous_source_windows(records, [split[0] for split in splits], statistics, device)
    valid = continuous_source_windows(records, [split[1] for split in splits], statistics, device)
    eligible = d._eligible(train)
    selections = [d._validation_selection([window], args.context, args.max_val_endpoints)[0] for window in valid]
    inputs, outputs = records[0].neural.shape[1], records[0].behavior.shape[1]
    base = build_mamba3_official(inputs, outputs, args.width, args.layers, args.state_size, args.dropout).to(device)
    model = SessionAwareDecoder(base, SessionFrontendBank(len(records), inputs, args.width, device=device))
    optimizer = torch.optim.AdamW(_groups(list(model.parameters()), args.lr, args.weight_decay))
    runtime = _runtime_identity()
    best_score, best_step = -float('inf'), 0
    log = []
    started = time.perf_counter()
    for step in range(1, args.steps+1):
        model.train()
        x, y, mask, sessions = sample_balanced_batch(train, eligible, args.context, args.batch_size)
        x = drift_augment(x, args.gain_sigma, args.offset_sigma, args.channel_dropout)
        prediction = model(x, sessions)[:, args.context//2:]
        truth = y[:, args.context//2:]
        good = mask[:, args.context//2:] & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(prediction).all():
            raise FloatingPointError('source training batch is invalid')
        optimizer.zero_grad(set_to_none=True)
        loss = (prediction[good]-truth[good]).square().mean()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True))
        optimizer.step()
        if step == 1 or step % args.val_interval == 0 or step == args.steps:
            score = validate_source(model, valid, selections, args.context, args.eval_batch_size, statistics[0])
            if not np.isfinite(score):
                raise FloatingPointError('source validation score is not finite')
            event = {'step': step, 'loss': float(loss.detach()), 'gradient_norm': gradient,
                     'source_val_r2': score, 'elapsed_seconds': time.perf_counter()-started}
            log.append(event)
            with (out/'train_log.jsonl').open('a') as stream:
                stream.write(json.dumps(event)+'\n')
            if score > best_score:
                best_score, best_step = score, step
                torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'best_step': step,
                            'source_val_r2': score, 'source_statistics': statistics, 'source_sessions': source_ids,
                            'code_hashes': hashes, 'source_data_sha256': source_hashes, 'runtime_identity': runtime}, out/'session_bank_best.pt')
    torch.save({'state_dict': model.state_dict(), 'args': vars(args)}, out/'session_bank_last.pt')
    selected = torch.load(out/'session_bank_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['state_dict'])
    model.eval()
    folded = build_mamba3_official(inputs, outputs, args.width, args.layers, args.state_size, args.dropout).to(device)
    folded.load_state_dict(base.state_dict())
    folded.in_proj = fold_session_input(base.in_proj, model.bank)
    # Verify every source fold against the trained session-aware model.
    fold_errors, input_errors, fold_pass = {}, {}, True
    with torch.inference_mode():
        for session in range(len(records)):
            x = train[session][0][:args.context].unsqueeze(0)
            folded.in_proj = fold_session_input(base.in_proj, model.bank, session)
            folded.eval()
            before = model.bank.add_embedding(base.in_proj(model.bank.forward_input(x,session)),session)
            after = folded.in_proj(x)
            input_errors[source_ids[session]] = float((before-after).abs().max())
            expected, actual = model(x,session), folded(x)
            fold_errors[source_ids[session]] = float((expected-actual).abs().max())
            fold_pass = fold_pass and bool(torch.allclose(before,after,atol=1e-5,rtol=1e-5))
            fold_pass = fold_pass and bool(torch.allclose(expected,actual,atol=1e-2,rtol=1e-2))
    folded.in_proj = fold_session_input(base.in_proj, model.bank)
    fallback_stats = (np.mean([stats[0] for stats in statistics], axis=0),
                      np.mean([stats[1] for stats in statistics], axis=0), statistics[0][2], statistics[0][3])
    np.savez(out/'normalizer.npz', **dict(zip(('x_mean','x_std','y_mean','y_std'), fallback_stats)))
    if hashes != code_hashes() or source_hashes != {record.session: sha256_file(record.path) for record in records}:
        raise RuntimeError('source implementation or data changed during training')
    if not fold_pass:
        raise RuntimeError('session fold differs from session-aware inference')
    receipt = {'status': 'completed', 'args': vars(args), 'plan': plan, 'source_sessions': source_ids,
               'source_train_bounds': [split[0] for split in splits], 'source_val_bounds': [split[1] for split in splits],
               'source_statistics_scope': 'source training neural; eval-valid training labels',
               'policy': 'recording_causal_fixed_window', 'continuous_context': True,
               'label_partitions_masked': True, 'session_balanced_sampling': True,
               'best_step': best_step, 'best_source_val_r2': best_score, 'fit_seconds': time.perf_counter()-started,
               'base_parameters': sum(p.numel() for p in base.parameters()),
               'session_frontend_parameters': sum(p.numel() for p in model.bank.parameters()),
               'target_frontend_initialization': 'mean of source gain, bias, and latent embedding folded into in_proj',
               'source_fold_max_abs': fold_errors, 'code_hashes': hashes, 'source_data_sha256': source_hashes,
               'source_fold_input_max_abs': input_errors,
               'fold_input_tolerance': {'atol':1e-5,'rtol':1e-5},
               'fold_bf16_output_tolerance': {'atol':1e-2,'rtol':1e-2},
               'normalizer_sha256': hashlib.sha256((out/'normalizer.npz').read_bytes()).hexdigest(),
               'session_bank_best_sha256': hashlib.sha256((out/'session_bank_best.pt').read_bytes()).hexdigest(),
               'train_log_sha256': hashlib.sha256((out/'train_log.jsonl').read_bytes()).hexdigest(),
               'runtime_identity': runtime, 'target_recording_loaded': False, 'query_used_for_selection': False,
               'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None}
    torch.save({'state_dict': folded.state_dict(), 'args': vars(args), 'best_step': best_step,
                'best_source_val_r2': best_score, 'session_pretraining_receipt': receipt}, out/'best.pt')
    receipt['checkpoint_sha256'] = hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['m1','m2'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=str(DEFAULT_ROOT))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--kind', default='mamba3_official', choices=['mamba3_official'])
    for name, default in [('width',128),('layers',2),('state-size',32),('context',128),('batch-size',32),
                          ('eval-batch-size',256),('steps',2000),('val-interval',100),('max-val-endpoints',1024),('seed',0)]:
        parser.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr',1e-3),('weight-decay',1e-3),('dropout',.2),('gain-sigma',.3),('offset-sigma',.1),('channel-dropout',.05)]:
        parser.add_argument('--'+name, type=float, default=default)
    print(run(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
