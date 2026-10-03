"""Run the frozen source capacity matrix and recording-context controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/session_pretraining_iteration_b.json'
sys.path[:0] = [str(ROOT/'.tools/mamba_deps'), str(ROOT)]


def current_runtime():
    from ssm_decode.cross_session_iteration import _runtime_identity
    return _runtime_identity()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2)+'\n')


def provenance(cfg, stage_a):
    matrix = read(stage_a/'matrix.json')
    names = matrix['code_files']+['ssm_decode/session_pretraining.py', 'ssm_decode/session_frontend.py']
    source_data = {}
    for task in cfg['tasks']:
        manifest = read(stage_a/task/'none/seed0/lr0/manifest.json')
        rows = {row['session']: row for row in manifest['plan']['held_in']}
        source_data[task] = {session: {'path': rows[session]['path'], 'sha256': sha(rows[session]['path'])}
                             for session in manifest['plan']['source_held_in_sessions']}
    return {'config_sha256': sha(CONFIG), 'launcher_sha256': sha(__file__),
            'adapt_launcher_sha256': sha(ROOT/'scripts/run_cross_session_iteration.py'),
            'runtime_identity': current_runtime(), 'source_dataset_inputs': source_data,
            'stage_a_provenance': read(stage_a/'provenance.json'),
            'stage_a_matrix_sha256': sha(stage_a/'matrix.json'),
            'code_sha256': {name: sha(ROOT/name) for name in names}}


def prepare(output, stage_a):
    cfg = read(CONFIG)
    source_count = len(cfg['tasks'])*len(cfg['capacities'])
    adapt_count = len(cfg['tasks'])*len(cfg['adapt_profiles'])*(1+(len(cfg['adapt_methods'])-1)*len(cfg['adapt_selection']['seeds']))
    if (cfg['counts'] != {'source_fits': source_count, 'adapt_fits': adapt_count} or
            cfg['adapt_selection']['none_seeds'] != [0]):
        raise RuntimeError('declared counts or none seeds differ from executable enumeration')
    if any(cfg['shared_args'].get(key) != value for key,value in {
            'mode':'cross-session','source_only':True,'window_policy':'recording_causal_fixed_window'}.items()):
        raise RuntimeError('source-only recording protocol differs from the implemented protocol')
    receipt = provenance(cfg, stage_a)
    if output.exists():
        if read(output/'provenance.json') != receipt or read(output/'matrix.json') != cfg:
            raise RuntimeError('source iteration provenance differs from the frozen root')
    else:
        output.mkdir(parents=True)
        write(output/'matrix.json', cfg)
        write(output/'provenance.json', receipt)
        for name in receipt['code_sha256']:
            destination = output/'code_snapshot'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        shutil.copyfile(__file__, output/'code_snapshot/run_session_pretraining_iteration.py')
    return cfg, receipt


def verify_live(cfg, frozen, stage_a):
    if provenance(cfg, stage_a) != frozen:
        raise RuntimeError('source iteration code or input provenance changed')


def child_env(output, gpu):
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=f"{ROOT/'.tools/mamba_deps'}:{ROOT}",
               TRITON_CACHE_DIR=str(output/'.triton_cache'/f'gpu{gpu}'))
    return env


def source_path(output, task, capacity):
    return output/'source'/task/capacity['name']


def source_is_current(path, cfg, frozen, task, capacity):
    required = ('manifest.json','best.pt','normalizer.npz','session_bank_best.pt','train_log.jsonl')
    if any(not (path/name).is_file() for name in required):
        return False
    manifest = read(path/'manifest.json')
    expected = {key: value for key, value in cfg['shared_args'].items() if key not in ('mode','source_only','window_policy')}
    expected.update(task=task, width=capacity['width'], layers=capacity['layers'], device='cuda:0', kind='mamba3_official')
    if any(manifest.get('args', {}).get(key) != value for key, value in expected.items()):
        return False
    if manifest.get('status') != 'completed' or manifest.get('target_recording_loaded') is not False:
        return False
    hashes = {Path(name).name: value for name, value in frozen['code_sha256'].items()}
    if manifest.get('code_hashes') != hashes:
        return False
    if any(manifest.get(key) != sha(path/name) for key, name in (
            ('checkpoint_sha256','best.pt'), ('normalizer_sha256','normalizer.npz'),
            ('session_bank_best_sha256','session_bank_best.pt'), ('train_log_sha256','train_log.jsonl'))):
        return False
    import torch
    payload = torch.load(path/'best.pt', map_location='cpu', weights_only=False)
    embedded = payload.get('session_pretraining_receipt', {})
    expected_receipt = {key:value for key,value in manifest.items() if key != 'checkpoint_sha256'}
    # Checkpoint uses tuples for bounds; JSON uses arrays.
    if json.loads(json.dumps(embedded)) != expected_receipt:
        return False
    if payload.get('args') != manifest['args'] or manifest['runtime_identity'] != current_runtime():
        return False
    bank_payload = torch.load(path/'session_bank_best.pt',map_location='cpu',weights_only=False)
    for key in ('args','code_hashes','source_data_sha256','runtime_identity','source_sessions'):
        if bank_payload.get(key) != manifest.get(key):
            return False
    events = [json.loads(line) for line in (path/'train_log.jsonl').read_text().splitlines()]
    chosen = max(events,key=lambda row:(row['source_val_r2'],-row['step']))
    if (manifest.get('best_step') != chosen['step'] or manifest.get('best_source_val_r2') != chosen['source_val_r2'] or
            payload.get('best_step') != chosen['step'] or payload.get('best_source_val_r2') != chosen['source_val_r2'] or
            bank_payload.get('best_step') != chosen['step'] or bank_payload.get('source_val_r2') != chosen['source_val_r2']):
        return False
    bank_state, plain_state = bank_payload['state_dict'], payload['state_dict']
    gain = bank_state['bank.gain'].mean(0)
    bias = bank_state['bank.bias'].mean(0)
    embedding = bank_state['bank.embedding'].mean(0)
    weight = bank_state['base.in_proj.weight']
    expected_weight = weight*gain.unsqueeze(0)
    expected_bias = weight@bias+bank_state['base.in_proj.bias']+embedding
    if (not torch.allclose(plain_state['in_proj.weight'],expected_weight,atol=1e-6,rtol=1e-6) or
            not torch.allclose(plain_state['in_proj.bias'],expected_bias,atol=1e-6,rtol=1e-6)):
        return False
    for name,value in plain_state.items():
        if name not in ('in_proj.weight','in_proj.bias') and not torch.equal(value,bank_state['base.'+name]):
            return False
    expected_source = {session: row['sha256'] for session,row in frozen['source_dataset_inputs'][task].items()}
    if (manifest['source_data_sha256'] != expected_source or manifest['source_sessions'] != list(expected_source) or
            manifest.get('label_partitions_masked') is not True or manifest.get('continuous_context') is not True or
            manifest.get('query_used_for_selection') is not False):
        return False
    rows = {row['session']: row for row in manifest['plan']['held_in']}
    return (all(sha(rows[session]['path']) == digest for session, digest in manifest['source_data_sha256'].items()) and
            manifest['runtime_identity']['official_actual'] == manifest['runtime_identity']['official_expected'])


def run_sources(cfg, frozen, stage_a, output):
    def queue(task):
        for capacity in cfg['capacities']:
            verify_live(cfg, frozen, stage_a)
            path = source_path(output, task, capacity)
            if source_is_current(path, cfg, frozen, task, capacity):
                continue
            if path.exists():
                raise RuntimeError(f'stale or incomplete source fit exists: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'ssm_decode.session_pretraining', '--task', task,
                       '--output', str(path), '--device', 'cuda:0', '--width', str(capacity['width']),
                       '--layers', str(capacity['layers'])]
            for key, value in cfg['shared_args'].items():
                if key not in ('mode','source_only','window_policy'):
                    command += ['--'+key.replace('_','-'), str(value)]
            with (path.parent/(path.name+'.log')).open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, env=child_env(output,cfg['tasks'][task]['physical_gpu']),
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode or not source_is_current(path,cfg,frozen,task,capacity):
                raise RuntimeError(f'source fit failed provenance checks: {path}')
        small, wide = [read(source_path(output,task,capacity)/'manifest.json') for capacity in cfg['capacities']]
        for key in ('source_sessions','source_train_bounds','source_val_bounds','source_data_sha256','runtime_identity'):
            if small[key] != wide[key]:
                raise RuntimeError(f'capacity controls differ in {key}: {task}')
    parallel(cfg['tasks'], queue)


def adaptation_matrix(cfg, stage_a, output, task, profile):
    original = read(stage_a/'matrix.json')
    selections = read(stage_a/'selections.json')
    # The A launcher reconstructs selections from prefix fits before it evaluates.
    if selections.get('query_metrics_read') is not False or selections['provenance'] != read(stage_a/'provenance.json'):
        raise RuntimeError('A selection receipt does not match its frozen provenance')
    for task_row in selections['tasks'].values():
        for method in task_row['methods'].values():
            for fit in method['selected']['fits']:
                if not (Path(fit['fit'])/'query_evaluation/metrics.json').is_file():
                    raise RuntimeError('A selected evaluation artifacts are incomplete')
    matrix = dict(original)
    matrix.update(name=f'session_iteration_b_{task}_{profile}', tasks=[task],
                  physical_gpu_by_task={task:str(cfg['tasks'][task]['physical_gpu'])},
                  policy=cfg['adapt_selection']['policy'], seeds=cfg['adapt_selection']['seeds'])
    matrix['methods'] = {method:[selections['tasks'][task]['methods'][method]['selected']['lr']]
                         for method in cfg['adapt_methods']}
    source = original['pretrained_by_task'][task]
    if profile != 'old':
        capacity = next(row for row in cfg['capacities'] if row['name'] == profile)
        source = str(source_path(output,task,capacity)/'best.pt')
    matrix['pretrained_by_task'] = {task:source}
    matrix['lr_transfer_basis'] = 'Stage A prefix validation; query scores are never read'
    matrix['parent_stage_a_provenance'] = selections['provenance']
    if profile != 'old':
        source_dir = Path(source).parent
        matrix['source_pretraining_provenance'] = {
            'manifest_sha256': sha(source_dir/'manifest.json'),
            'bank_best_sha256': sha(source_dir/'session_bank_best.pt'),
            'receipt': read(source_dir/'manifest.json')}
    return matrix


def run_adaptations(cfg, frozen, stage_a, output):
    # Authenticate A selections independently before transferring their LRs.
    sys.path.insert(0,str(ROOT))
    from scripts.run_cross_session_iteration import _select
    if read(stage_a/'selections.json') != _select(read(stage_a/'matrix.json'),stage_a,read(stage_a/'provenance.json'),persist=False):
        raise RuntimeError('A selections differ from independent prefix recomputation')
    def queue(task):
        for profile in cfg['adapt_profiles']:
            verify_live(cfg,frozen,stage_a)
            if profile != 'old':
                capacity = next(row for row in cfg['capacities'] if row['name'] == profile)
                if not source_is_current(source_path(output,task,capacity),cfg,frozen,task,capacity):
                    raise RuntimeError(f'source checkpoint is not current: {task}/{profile}')
            matrix = adaptation_matrix(cfg,stage_a,output,task,profile)
            matrix_path = output/'adapt_matrices'/f'{task}_{profile}.json'
            matrix_path.parent.mkdir(exist_ok=True)
            if matrix_path.exists() and read(matrix_path) != matrix:
                raise RuntimeError('existing adaptation matrix differs')
            write(matrix_path,matrix)
            destination = output/'adapt'/profile/task
            for manifest_file in destination.glob('*/*/seed*/lr*/manifest.json'):
                if read(manifest_file).get('runtime_identity') != current_runtime():
                    raise RuntimeError('existing adaptation runtime differs from current runtime')
            with (matrix_path.with_suffix('.log')).open('w') as stream:
                result = subprocess.run([sys.executable,str(ROOT/'scripts/run_cross_session_iteration.py'),
                        '--matrix',str(matrix_path),'--output-root',str(destination),'--phase','all'],
                        cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'adaptation failed: {destination}')
            for artifact in list(destination.glob('*/*/seed*/lr*/manifest.json'))+list(destination.glob('*/*/seed*/lr*/query_evaluation/metrics.json')):
                if read(artifact).get('runtime_identity') != current_runtime():
                    raise RuntimeError('completed adaptation runtime differs from current runtime')
    parallel(cfg['tasks'],queue)


def parallel(tasks, function):
    errors = []
    def queue(task):
        try:
            function(task)
        except Exception as error:
            errors.append((task,str(error)))
    threads = [threading.Thread(target=queue,args=(task,)) for task in tasks]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if errors:
        raise RuntimeError(str(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-a-root',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--phase',choices=['source','adapt','all'],default='all')
    args = parser.parse_args(argv)
    stage_a, output = args.stage_a_root.resolve(), args.output_root.resolve()
    cfg, frozen = prepare(output,stage_a)
    if args.phase in ('source','all'): run_sources(cfg,frozen,stage_a,output)
    if args.phase in ('adapt','all'): run_adaptations(cfg,frozen,stage_a,output)


if __name__ == '__main__':
    main()
