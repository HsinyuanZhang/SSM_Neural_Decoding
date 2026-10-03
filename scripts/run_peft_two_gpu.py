"""Launch reproducible PEFT queues with one visible CUDA device per child."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix_bytes = matrix_path.read_bytes()
    cfg = json.loads(matrix_bytes)
    output = Path(args.output_root)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'matrix.json').write_bytes(matrix_bytes)
    launcher_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {'matrix_path':str(matrix_path),
                  'matrix_sha256':hashlib.sha256(matrix_bytes).hexdigest(),
                  'launcher_sha256':launcher_hash}
    jobs = []
    for method in cfg['methods']:
        for seed in ([0] if method == 'none' else cfg['seeds']):
            for task, physical_gpu in [('m1', '0'), ('m2', '1')]:
                dest = output / task / f'{method}_seed{seed}'
                cmd = [sys.executable, '-m', 'ssm_decode.peft_experiment', '--task', task,
                       '--device', 'cuda:0', '--output', str(dest), '--pretrained', cfg['pretrained_by_task'][task],
                       '--method', method, '--steps', str(cfg['steps']), '--context', str(cfg['context']),
                       '--batch-size', str(cfg['batch_size']), '--rank', str(cfg['rank']), '--alpha', str(cfg['alpha']),
                       '--seed', str(seed), '--policy', cfg['policy']]
                jobs.append((task, physical_gpu, dest, cmd))
    (output/'launch_manifest.json').write_text(json.dumps({**provenance,'jobs':[{'task':t,'physical_gpu':g,'output':str(o),'command':c} for t,g,o,c in jobs]},indent=2)+'\n')
    errors=[]
    def queue(task):
        for _,gpu,dest,cmd in [x for x in jobs if x[0] == task]:
            env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['CUDA_VISIBLE_DEVICES']=gpu
            env['PYTHONPATH']=f"{ROOT/'.tools'/'mamba_deps'}:{ROOT}"
            env['TRITON_CACHE_DIR']=str(output/'.triton_cache'/task)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with (dest.parent/(dest.name+'.log')).open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode: errors.append((task,dest.name,result.returncode)); return
            print(f'{task} {dest.name} completed',flush=True)
    threads=[threading.Thread(target=queue,args=(task,)) for task in ('m1','m2')]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != launcher_hash:
        errors.append(('launcher_changed',None,None))
    if errors: raise SystemExit(str(errors))
if __name__ == '__main__': main()
