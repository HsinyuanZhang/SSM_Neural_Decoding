"""Compare unit and session maps with an unchanged Mamba-3 backbone.

Fit every declared model before the separate query-scoring phase.
Use raw calibration trials. Keep query labels out of all fit functions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import debug_experiment as d
from .cross_session_iteration import _code_hashes, _runtime_identity, _groups, _parameter_hashes
from .data import load_recording, source_target_plan, sha256_file, APST_SRC
from .mamba3_official import build_mamba3_official
from .official_calibration import load_public_calibration
from .session_frontend import drift_augment
from .session_pretraining import source_statistics, continuous_source_windows, sample_balanced_batch
from .unit_session_frontend import UnitSessionBank, UnitSessionDecoder, fold_unit_session

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('x_mean', 'x_std', 'y_mean', 'y_std')


class FixedSessionView(torch.nn.Module):
    def __init__(self, model, session):
        super().__init__()
        self.model, self.session = model, session

    def forward(self, x):
        return self.model(x, self.session)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def code_hashes():
    hashes = _code_hashes()
    for name in ('unit_session_experiment.py', 'unit_session_frontend.py', 'session_frontend.py',
                 'session_pretraining.py', 'official_calibration.py'):
        hashes[name] = sha256_file(Path(__file__).with_name(name))
    return hashes


def r2(truth, prediction):
    truth, prediction = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('score arrays must have equal shapes and finite values')
    return float(1-np.square(truth-prediction).sum()/max(np.square(truth-truth.mean(0)).sum(), 1e-12))


def raw_support_partition(calibration):
    """Reserve every fifth raw trial, plus the last trial if necessary."""
    bounds = list(calibration.trial_bounds)
    n = len(bounds)
    if n not in (10, 33) or calibration.receipt['raw_nwb_trial_ids_first_n'] != list(range(n)):
        raise ValueError('calibration must contain the first 10 or 33 raw trials')
    validation = set(range(4, n, 5))
    if len(validation) < int(np.ceil(n/5)):
        validation.add(n-1)
    return ([b for i, b in enumerate(bounds) if i not in validation],
            [b for i, b in enumerate(bounds) if i in validation], sorted(validation))


def target_statistics(calibration, source_stats):
    """Use prefix neural bins and keep source output statistics fixed."""
    neural = np.asarray(calibration.neural, np.float64)
    mean, std = neural.mean(0), neural.std(0)
    std[std < 1e-6] = 1
    return (mean.astype(np.float32), std.astype(np.float32),
            np.asarray(source_stats[2], np.float32), np.asarray(source_stats[3], np.float32))


def target_windows(calibration, bounds, statistics, device):
    x = ((calibration.neural-statistics[0])/statistics[1]).astype(np.float32)
    y = np.full_like(calibration.behavior, np.nan)
    mask = np.zeros(len(x), dtype=bool)
    for a, b in bounds:
        y[a:b] = (calibration.behavior[a:b]-statistics[2])/statistics[3]
        mask[a:b] = calibration.eval_mask[a:b]
    return [(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(mask).to(device))]


def effective_maps(model):
    """Return input and normalized-output maps. These maps define the penalty."""
    b, base = model.bank, model.base
    weight = base.in_proj.weight
    a = weight.unsqueeze(0)*b.gain.unsqueeze(1)
    if b.unit_delta is not None:
        a = a+b.unit_delta
    c = torch.einsum('dc,sc->sd', weight, b.bias)+b.embedding
    if base.in_proj.bias is not None:
        c = c+base.in_proj.bias
    scale = b.readout_log_gain.exp() if b.session_readout else b.gain.new_ones(b.num_sessions, b.output_size)
    v = base.out_proj.weight.unsqueeze(0)*scale.unsqueeze(-1)
    offset = b.readout_bias if b.session_readout else b.gain.new_zeros(b.num_sessions, b.output_size)
    if base.out_proj.bias is not None:
        offset = offset+scale*base.out_proj.bias
    return dict(input_weight=a, input_bias=c, output_weight=v, output_bias=offset)


def mapping_penalty(maps, anchors=None):
    """Penalize effective map deviations, independent of parameter decomposition."""
    terms = []
    for name, value in maps.items():
        center = value.mean(0, keepdim=True) if anchors is None else anchors[name]
        # Sum over latent rows for the read-in matrix. Average over units/sessions.
        error = (value-center).square()
        terms.append(error.sum(-2).mean() if name == 'input_weight' else error.mean())
    return torch.stack(terms).sum()


def map_receipt(model, output):
    values = {n: v.detach().cpu().numpy() for n, v in effective_maps(model).items()}
    path = output/'effective_maps.npz'
    np.savez(path, **values)
    return dict(effective_maps_sha256=sha256_file(path), effective_map_norms={
        n: float(np.linalg.norm(v.astype(np.float64))) for n, v in values.items()})


def fold_replay(model, windows, session, cfg, stats):
    """Check every validation endpoint. Retain an unmerged model on failure."""
    folded = fold_unit_session(model.base, model.bank, session).eval()
    selections = d._validation_selection(windows, cfg['context'], 10**9)[0]
    truth, unmerged, merged = [], [], []
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(selections), cfg['eval_batch_size']):
            x, y, mask = d._right_aligned(windows, selections[lo:lo+cfg['eval_batch_size']], cfg['context'])
            p = model(x, session)[:, -1].float().cpu().numpy()
            q = folded(x)[:, -1].float().cpu().numpy()
            truth.append(y[:, -1].float().cpu().numpy())
            unmerged.append(p); merged.append(q)
    truth, unmerged, merged = [np.concatenate(v) for v in (truth, unmerged, merged)]
    # Validate all points independently from score similarity.
    point_pass = bool(np.allclose(unmerged, merged, atol=.01, rtol=.01))
    y = truth*stats[3]+stats[2]
    p, q = unmerged*stats[3]+stats[2], merged*stats[3]+stats[2]
    pscore, qscore = r2(y, p), r2(y, q)
    return folded, dict(validation_endpoints=len(y), unmerged_r2=pscore, folded_r2=qscore,
        r2_delta=qscore-pscore, normalized_max_abs=float(np.max(np.abs(unmerged-merged))),
        point_allclose_pass=point_pass, score_pass=abs(qscore-pscore) <= .001,
        fold_accepted=point_pass and abs(qscore-pscore) <= .001,
        normalized_point_tolerance=dict(atol=.01, rtol=.01), r2_tolerance=.001)


def make_model(task_cfg, variant, inputs, outputs, sessions, cfg, device):
    base = build_mamba3_official(inputs, outputs, task_cfg['width'], task_cfg['layers'],
                                cfg['state_size'], cfg['dropout']).to(device)
    bank = UnitSessionBank(sessions, inputs, task_cfg['width'], outputs,
                           unit_residual=variant['unit_residual'], session_readout=variant['session_readout'],
                           device=device)
    return UnitSessionDecoder(base, bank)


def train_loop(model, sample, validate, cfg, lr, penalty, output, stage, checkpoint_metadata):
    """Select a checkpoint on validation. Extend a late maximum before scoring."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(_groups(trainable, lr, cfg['weight_decay'])) if trainable else None
    initial_hashes = _parameter_hashes(model)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    anchors = {n: v.detach().clone() for n, v in effective_maps(model).items()}
    step, best_step, budget = 0, 0, cfg['steps'] if optimizer else 0
    best = validate()
    if not np.isfinite(best):
        raise FloatingPointError('initial validation is not finite')
    logs = [{'step': 0, 'val_r2': best}]
    extensions = []
    def save(name, selected_step):
        torch.save(dict(state_dict=model.state_dict(), best_step=selected_step,
                        metadata=checkpoint_metadata), output/name)
    save('best.pt', 0)
    started = time.perf_counter()
    while step < budget:
        step += 1
        model.train()
        x, truth, mask, session = sample()
        prediction = model(x, session)[:, cfg['context']//2:]
        truth, mask = truth[:, cfg['context']//2:], mask[:, cfg['context']//2:]
        good = mask & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(prediction).all():
            raise FloatingPointError('training batch is not finite or has no labels')
        optimizer.zero_grad(set_to_none=True)
        data_loss = (prediction[good]-truth[good]).square().mean()
        extra = penalty(model, anchors)
        loss = data_loss+extra
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True))
        optimizer.step()
        event = {'step': step, 'data_loss': float(data_loss.detach()),
                 'penalty': float(extra.detach()), 'gradient_norm': grad}
        if step % cfg['val_interval'] == 0 or step == budget:
            score = validate()
            if not np.isfinite(score):
                raise FloatingPointError('validation is not finite')
            event['val_r2'] = score
            if score > best:
                best, best_step = score, step
                save('best.pt', step)
            write(output/'progress.json', dict(stage=stage, step=step, budget=budget,
                  best_step=best_step, best_val_r2=best, elapsed_seconds=time.perf_counter()-started))
        logs.append(event)
        if step == budget and best_step >= .8*budget and budget < cfg['max_steps']:
            updated = min(cfg['max_steps'], budget*2)
            extensions.append(dict(old=budget, new=updated, best_step=best_step))
            budget = updated
    save('last.pt', step)
    changed_frozen = [n for n, p in model.named_parameters() if n in frozen and not torch.equal(p, frozen[n])]
    if changed_frozen:
        raise RuntimeError('a frozen backbone parameter changed')
    model.load_state_dict(torch.load(output/'best.pt', map_location=model.bank.gain.device, weights_only=False)['state_dict'])
    model.eval()
    chosen_hashes = _parameter_hashes(model)
    write(output/'train_log.json', logs)
    result = dict(best_step=best_step, best_val_r2=best, completed_steps=step, final_budget=budget,
                  converged=not optimizer or best_step < .8*budget, extensions=extensions,
                  fit_seconds=time.perf_counter()-started,
                  trainable_parameters=sum(p.numel() for p in trainable),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_names=[n for n, p in model.named_parameters() if p.requires_grad],
                  frozen_parameter_audit_pass=True, initial_parameter_hashes=initial_hashes,
                  selected_parameter_hashes=chosen_hashes,
                  best_checkpoint_sha256=sha256_file(output/'best.pt'),
                  train_log_sha256=sha256_file(output/'train_log.json'),
                  peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    result.update(map_receipt(model, output))
    return result


def source_fit(task, task_cfg, variant, cfg, root, contract):
    out = root/'source'/variant['name']; out.mkdir(parents=True, exist_ok=False)
    d._seed(cfg['source_seed']); torch.cuda.reset_peak_memory_stats()
    plan = source_target_plan(task, root=Path(cfg['data_root']))
    records = [load_recording(task, 'held_in', sid, root=Path(cfg['data_root'])) for sid in plan['source_held_in_sessions']]
    splits = [d._trial_split(r) for r in records]
    statistics = source_statistics(records, splits)
    train = continuous_source_windows(records, [v[0] for v in splits], statistics, 'cuda:0')
    valid = continuous_source_windows(records, [v[1] for v in splits], statistics, 'cuda:0')
    eligible = d._eligible(train)
    selections = [d._validation_selection([w], cfg['context'], cfg['max_source_val_endpoints_per_session'])[0] for w in valid]
    inputs, outputs = records[0].neural.shape[1], records[0].behavior.shape[1]
    model = make_model(task_cfg, variant, inputs, outputs, len(records), cfg, 'cuda:0')
    initial_base_hashes = _parameter_hashes(model.base)
    def sample():
        x, y, mask, session = sample_balanced_batch(train, eligible, cfg['context'], cfg['batch_size'])
        x = drift_augment(x, cfg['gain_sigma'], cfg['offset_sigma'], cfg['channel_dropout'])
        return x, y, mask, session
    def validate():
        # A session has one vote. Long recordings do not dominate selection.
        scores = [d._val(FixedSessionView(model, session), [window], cfg['context'], selected,
            cfg['eval_batch_size'], statistics[session])
            for session, (window, selected) in enumerate(zip(valid, selections))]
        return float(np.mean(scores))
    def penalty(m, unused):
        return cfg['source_unit_penalty']*mapping_penalty(effective_maps(m))
    metadata = dict(task=task, task_cfg=task_cfg, variant=variant, cfg=cfg, inputs=inputs,
                    outputs=outputs, sessions=len(records), contract_sha256=sha256_file(root/'contract.json'))
    result = train_loop(model, sample, validate, cfg, cfg['source_lr'], penalty, out, 'source', metadata)
    folds = {}
    for session, record in enumerate(records):
        folded, receipt = fold_replay(model, [valid[session]], session, cfg, statistics[session])
        folds[record.session] = receipt
        del folded
    result.update(task=task, variant=variant, source_sessions=[r.session for r in records],
        source_train_bounds=[v[0] for v in splits], source_val_bounds=[v[1] for v in splits],
        source_data_sha256={r.session: sha256_file(r.path) for r in records},
        initial_base_parameter_hashes=initial_base_hashes,
        source_statistics=[dict(zip(KEYS, [a.tolist() for a in s])) for s in statistics],
        source_validation_selections_sha256=hashlib.sha256(json.dumps(selections).encode()).hexdigest(),
        target_recording_loaded=False, query_labels_used=False, source_fold_replays=folds,
        source_selection_metric='macro mean physical R2 across source sessions',
        selected_full_validation_r2_by_session={n: r['unmerged_r2'] for n, r in folds.items()})
    write(out/'fit_result.json', result)
    del model, train, valid
    torch.cuda.empty_cache()
    return result


def restore_source(source_dir, device='cuda:0'):
    checkpoint = torch.load(source_dir/'best.pt', map_location=device, weights_only=False)
    meta = checkpoint['metadata']
    model = make_model(meta['task_cfg'], meta['variant'], meta['inputs'], meta['outputs'], meta['sessions'], meta['cfg'], device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, meta


def target_fit(task, variant, cfg, root, method, seed, lr):
    name = f'{method}_seed{seed}_lr{lr:g}'
    out = root/'adapt'/variant['name']/name; out.mkdir(parents=True, exist_ok=False)
    source_dir = root/'source'/variant['name']
    source, meta = restore_source(source_dir)
    base = copy.deepcopy(source.base)
    model = UnitSessionDecoder(base, source.bank.target_bank())
    del source
    for p in model.base.parameters(): p.requires_grad_(False)
    for p in model.bank.parameters(): p.requires_grad_(method != 'none')
    d._seed(seed); torch.cuda.reset_peak_memory_stats()
    target_session = read(root/'contract.json')['plan']['cross_session_local_dev']['target_session']
    calibration = load_public_calibration(task, 'held_in', target_session, Path(cfg['data_root']))
    calibration_binding = read(root/'contract.json')['calibration_binding']
    if (str(calibration.path) != calibration_binding['path']
            or calibration.receipt != calibration_binding['receipt']):
        raise RuntimeError('calibration identity changed between candidates')
    if calibration.receipt['calibration_trials'] != cfg['tasks'][task]['raw_support_trials']:
        raise RuntimeError('raw calibration budget differs')
    source_result = read(source_dir/'fit_result.json')
    source_stats = [np.asarray(source_result['source_statistics'][0][k], np.float32) for k in KEYS]
    statistics = target_statistics(calibration, source_stats)
    np.savez(out/'normalizer.npz', **dict(zip(KEYS, statistics)))
    train_bounds, val_bounds, val_ids = raw_support_partition(calibration)
    train, valid = (target_windows(calibration, bounds, statistics, 'cuda:0') for bounds in (train_bounds, val_bounds))
    eligible = d._eligible(train)
    selection, selection_sha = d._validation_selection(valid, cfg['context'], 10**9)
    def sample():
        x, y, mask = d._sample_batch(train, eligible, cfg['context'], cfg['batch_size'])
        return x, y, mask, 0
    def validate():
        return d._val(model, valid, cfg['context'], selection, cfg['eval_batch_size'], statistics)
    def penalty(m, anchors):
        return cfg['target_anchor_penalty']*mapping_penalty(effective_maps(m), anchors)
    metadata = dict(meta, sessions=1, method=method, seed=seed, lr=lr,
                    source_checkpoint_sha256=sha256_file(source_dir/'best.pt'))
    result = train_loop(model, sample, validate, cfg, lr, penalty, out, 'target', metadata)
    prefix = target_windows(calibration, list(calibration.trial_bounds), statistics, 'cuda:0')
    folded, fold_receipt = fold_replay(model, prefix, 0, cfg, statistics)
    folded_val = fold_receipt['folded_r2']
    delta = fold_receipt['r2_delta']
    torch.save(dict(state_dict=folded.state_dict(), metadata=metadata), out/'folded.pt')
    result.update(task=task, variant=variant['name'], method=method, seed=seed, lr=lr,
        calibration_receipt=calibration.receipt, train_bounds=train_bounds, val_bounds=val_bounds,
        validation_raw_trial_indices=val_ids, prefix_val_indices_sha256=selection_sha,
        normalization_scope='all legal prefix neural including leading neural-only bins; fixed source-train outputs',
        normalizer_sha256=sha256_file(out/'normalizer.npz'),
        source_checkpoint_sha256=sha256_file(source_dir/'best.pt'),
        folded_checkpoint_sha256=sha256_file(out/'folded.pt'), fold_prefix_r2=folded_val,
        fold_prefix_r2_delta=delta, fold_prefix_r2_pass=abs(delta) <= .001,
        fold_replay=fold_receipt, deployment_fold_accepted=fold_receipt['fold_accepted'],
        query_labels_used=False, source_initializer='arithmetic mean of effective source input and normalized-output maps',
        backbone_frozen=True, representation='session-specific channel read-in; no biological unit alignment')
    if sha256_file(calibration.path) != calibration_binding['receipt']['rawfile_sha256']:
        raise RuntimeError('calibration data changed during fit')
    write(out/'fit_result.json', result)
    del model, folded, train, valid
    torch.cuda.empty_cache()
    return result


def verify_contract(root, config_path):
    frozen = read(root/'contract.json')
    if (frozen['config_sha256'] != sha256_file(config_path) or frozen['code_hashes'] != code_hashes()
            or frozen['runtime'] != _runtime_identity()):
        raise RuntimeError('frozen configuration, implementation, or runtime changed')
    for row in frozen['plan']['held_in']:
        if sha256_file(row['path']) != row['sha256']:
            raise RuntimeError('held-in data file changed')
    calibration = frozen['calibration_binding']
    if (sha256_file(calibration['path']) != calibration['receipt']['rawfile_sha256']
            or frozen['reader_sha256'] != sha256_file(APST_SRC/'apst/data/load.py')):
        raise RuntimeError('calibration data or official reader changed')
    return frozen


def fit_task(task, cfg, config_path, root):
    root.mkdir(parents=True, exist_ok=False)
    plan = source_target_plan(task, root=Path(cfg['data_root']))
    calibration = load_public_calibration(task, 'held_in', plan['cross_session_local_dev']['target_session'], Path(cfg['data_root']))
    write(root/'contract.json', dict(schema='unit_session_contract_v1', cfg=cfg, task=task,
        config_sha256=sha256_file(config_path), code_hashes=code_hashes(), runtime=_runtime_identity(), plan=plan,
        calibration_binding=dict(path=str(calibration.path), receipt=calibration.receipt),
        reader_sha256=sha256_file(APST_SRC/'apst/data/load.py'),
        raw_validation_trial_indices=raw_support_partition(calibration)[2],
        query_labels_used_for_selection=False, source_target_disjoint=True,
        representation='session-specific channel read-in; no biological unit alignment'))
    for name in code_hashes():
        destination = root/'code_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(__file__).with_name(name).read_bytes())
    rows = []
    for variant in cfg['variants']:
        verify_contract(root, config_path)
        source = source_fit(task, cfg['tasks'][task], variant, cfg, root, read(root/'contract.json'))
        if not source['converged']:
            raise RuntimeError('source maximum remains late at the maximum budget; no query scoring')
        rows.append(target_fit(task, variant, cfg, root, 'none', 0, 0))
        for lr in cfg['adapt_lrs']:
            for seed in cfg['adapt_seeds']:
                verify_contract(root, config_path)
                rows.append(target_fit(task, variant, cfg, root, 'ui', seed, lr))
    selected = []
    decisions = {}
    for variant in cfg['variants']:
        name = variant['name']
        selected += [r for r in rows if r['variant'] == name and r['method'] == 'none']
        candidates = []
        for lr in cfg['adapt_lrs']:
            family = [r for r in rows if r['variant'] == name and r['method'] == 'ui' and r['lr'] == lr]
            admissible = len(family) == len(cfg['adapt_seeds']) and all(r['converged'] for r in family)
            candidates.append(dict(lr=lr, admissible=admissible,
                mean_prefix_val_r2=float(np.mean([r['best_val_r2'] for r in family]))))
        available = [c for c in candidates if c['admissible']]
        if not available:
            decisions[name] = dict(candidates=candidates, selected_lr=None, excluded=True)
            continue
        best = max(available, key=lambda c: (c['mean_prefix_val_r2'], -c['lr']))
        decisions[name] = dict(candidates=candidates, selected_lr=best['lr'], excluded=False)
        selected += [r for r in rows if r['variant'] == name and r['method'] == 'ui' and r['lr'] == best['lr']]
    verify_contract(root, config_path)
    # All artifacts are hashed before any query-scoring process starts.
    artifacts = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob('*') if p.is_file()
                 and 'code_snapshot' not in p.parts and p.name != 'fits_complete.json'}
    write(root/'fits_complete.json', dict(selected=selected, decisions=decisions,
        source_fits=len(cfg['variants']), target_fits=len(rows), artifacts=artifacts,
        all_fits_completed_before_query=True, query_metrics_read=False))


def score_task(task, cfg, config_path, root, expected_completion_sha256):
    verify_contract(root, config_path)
    if sha256_file(root/'fits_complete.json') != expected_completion_sha256:
        raise RuntimeError('fit-completion manifest changed before query scoring')
    completion = read(root/'fits_complete.json')
    for name, digest in completion['artifacts'].items():
        if sha256_file(root/name) != digest:
            raise RuntimeError('fit artifact changed before scoring')
    plan = read(root/'contract.json')['plan']
    record = load_recording(task, 'held_in', plan['cross_session_local_dev']['target_session'], root=Path(cfg['data_root']))
    usable = [b for b in record.trial_bounds if b[1]-b[0] >= 50]
    query = usable[cfg['query_cutoff_usable_trials']:]
    indices = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in query])
    indices = indices[record.eval_mask[indices]]
    if len(indices) != cfg['tasks'][task]['expected_query_bins']:
        raise RuntimeError('query cohort count changed')
    truth = record.behavior[indices].copy()
    task_cfg = cfg['tasks'][task]
    if (hashlib.sha256(indices.tobytes()).hexdigest() != task_cfg['expected_indices_sha256']
            or hashlib.sha256(truth.tobytes()).hexdigest() != task_cfg['expected_truth_sha256']):
        raise RuntimeError('canonical query indices or truth changed')
    rows = []
    for selected in completion['selected']:
        variant = next(v for v in cfg['variants'] if v['name'] == selected['variant'])
        fit_name = f"{selected['method']}_seed{selected['seed']}_lr{selected['lr']:g}"
        fit = root/'adapt'/variant['name']/fit_name
        if sha256_file(fit/'best.pt') != selected['best_checkpoint_sha256']:
            raise RuntimeError('selected checkpoint changed')
        checkpoint = torch.load(fit/'best.pt', map_location='cuda:0', weights_only=False)
        meta = checkpoint['metadata']
        model = make_model(meta['task_cfg'], variant, meta['inputs'], meta['outputs'], 1, cfg, 'cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
        with np.load(fit/'normalizer.npz') as archive:
            stats = tuple(archive[k] for k in KEYS)
        prefix_end = selected['calibration_receipt']['prefixend']
        if indices.min() < prefix_end:
            raise RuntimeError('query overlaps legal calibration')
        x = ((record.neural-stats[0])/stats[1]).astype(np.float32)
        start = time.perf_counter()
        normalized = d._predict_endpoints(model, x, [(0, len(x))], indices, 'cuda:0', cfg['context'], batch=cfg['eval_batch_size'])
        prediction = (normalized*stats[3]+stats[2]).astype(np.float32)
        archive_path = root/'scores'/variant['name']/(fit_name+'.npz'); archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists(): raise RuntimeError('query archive already exists')
        np.savez(archive_path, indices=indices, truth=truth, prediction=prediction)
        rows.append(dict(task=task, variant=variant['name'], method=selected['method'], seed=selected['seed'], lr=selected['lr'],
            r2=r2(truth, prediction), query_bins=len(indices), best_step=selected['best_step'],
            best_prefix_val_r2=selected['best_val_r2'], trainable_parameters=selected['trainable_parameters'],
            query_seconds=time.perf_counter()-start, prediction_archive=str(archive_path),
            prediction_archive_sha256=sha256_file(archive_path), best_checkpoint_sha256=selected['best_checkpoint_sha256'],
            indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
            folded_prefix_r2_delta=selected['fold_prefix_r2_delta']))
        del model
    verify_contract(root, config_path)
    write(root/'summary.json', dict(schema='unit_session_local_results_v1', task=task, rows=rows,
          all_fits_completed_before_query=True, query_used_for_checkpoint_or_lr_selection=False,
          architecture_comparison_uses_local_query=True, official_heldout_result=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', choices=['m1', 'm2'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['fit', 'score'], required=True)
    parser.add_argument('--expected-fit-completion-sha256')
    args = parser.parse_args(argv)
    cfg = read(args.config)
    torch.set_num_threads(1); torch.cuda.set_device(0)
    if cfg['policy'] != 'recording_causal_fixed_window': raise RuntimeError('unsupported context policy')
    if args.phase == 'fit':
        fit_task(args.task, cfg, args.config.resolve(), args.output_root.resolve()/args.task)
    else:
        if not args.expected_fit_completion_sha256:
            raise RuntimeError('query scoring requires an externally bound fit-completion digest')
        score_task(args.task, cfg, args.config.resolve(), args.output_root.resolve()/args.task,
                   args.expected_fit_completion_sha256)


if __name__ == '__main__':
    main()
