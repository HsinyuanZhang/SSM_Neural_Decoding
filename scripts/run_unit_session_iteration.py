"""Run both GPU queues. Start query scoring only after both fit queues finish."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_task(task, cfg, config, output, phase, completion_sha=None):
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES=str(cfg['tasks'][task]['physical_gpu']),
               PYTHONPATH=f"{ROOT/'.tools/mamba_deps'}:{ROOT}",
               TRITON_CACHE_DIR=str(output/'.triton_cache'/task))
    with (output/f'{task}_{phase}.log').open('w') as stream:
        command = [sys.executable, '-m', 'ssm_decode.unit_session_experiment',
                   '--config', str(config), '--task', task, '--output-root', str(output), '--phase', phase]
        if phase == 'score':
            command += ['--expected-fit-completion-sha256', completion_sha]
        subprocess.run(command,
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/unit_session_iteration_20261004.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['fit', 'score'], default='fit')
    parser.add_argument('--expected-fit-seal-sha256')
    args = parser.parse_args()
    config, output = args.config.resolve(), args.output_root.resolve()
    cfg = json.loads(config.read_text())
    binding = dict(config=cfg, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                   launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.phase == 'fit':
        output.mkdir(parents=True, exist_ok=False)
        (output/'matrix.json').write_text(json.dumps(binding, indent=2)+'\n')
    elif json.loads((output/'matrix.json').read_text()) != binding:
        raise RuntimeError('score configuration or launcher differs from the frozen fit root')
    phases = [args.phase]
    expected_seal = args.expected_fit_seal_sha256
    for phase in phases:
        if phase == 'score' and any(not (output/task/'fits_complete.json').is_file() for task in cfg['tasks']):
            raise RuntimeError('every fit queue must finish before query scoring')
        seal = None
        if phase == 'score':
            if not expected_seal or digest(output/'fit_seal.json') != expected_seal:
                raise RuntimeError('query scoring requires the independently recorded fit-seal digest')
            seal = json.loads((output/'fit_seal.json').read_text())
            if (not isinstance(seal, dict) or seal.get('schema') != 'unit_session_fit_seal_v1'
                    or set(seal) != {'schema', 'tasks'} or not isinstance(seal.get('tasks'), dict)
                    or set(seal['tasks']) != set(cfg['tasks'])):
                raise RuntimeError('fit seal must bind every declared task exactly once')
            for task, binding in seal['tasks'].items():
                if (not isinstance(binding, dict) or set(binding) != {'completion_sha256', 'contract_sha256'}
                        or any(not isinstance(v, str) or re.fullmatch(r'[0-9a-f]{64}', v) is None
                               for v in binding.values())):
                    raise RuntimeError('fit seal task binding is malformed')
                if (digest(output/task/'fits_complete.json') != binding['completion_sha256']
                        or digest(output/task/'contract.json') != binding['contract_sha256']):
                    raise RuntimeError('a task fit-completion or contract changed')
                completion = json.loads((output/task/'fits_complete.json').read_text())
                if (completion.get('all_fits_completed_before_query') is not True
                        or completion.get('query_metrics_read') is not False
                        or not isinstance(completion.get('artifacts'), dict)):
                    raise RuntimeError('task completion does not prove a query-free fit phase')
                for name, value in completion['artifacts'].items():
                    path = (output/task/name).resolve()
                    if (output/task).resolve() not in path.parents or digest(path) != value:
                        raise RuntimeError('a sealed fit artifact changed')
            # Validate all data and implementation contracts before either task starts.
            sys.path.insert(0, str(ROOT))
            from ssm_decode.unit_session_experiment import verify_contract
            for task in cfg['tasks']:
                verify_contract(output/task, config)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_task, task, cfg, config, output, phase,
                       None if seal is None else seal['tasks'][task]['completion_sha256']) for task in cfg['tasks']]
            for future in futures: future.result()
        if phase == 'fit':
            seal = dict(schema='unit_session_fit_seal_v1', tasks={task: dict(
                completion_sha256=digest(output/task/'fits_complete.json'),
                contract_sha256=digest(output/task/'contract.json')) for task in cfg['tasks']})
            (output/'fit_seal.json').write_text(json.dumps(seal, indent=2)+'\n')
            expected_seal = digest(output/'fit_seal.json')
            print('fit_seal_sha256='+expected_seal, flush=True)


if __name__ == '__main__':
    main()
