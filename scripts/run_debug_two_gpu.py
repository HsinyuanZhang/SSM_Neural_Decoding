"""Reproducible two-GPU queues for ``ssm_decode.debug_experiment``.

Each queue is serial (M1 on GPU0, M2 on GPU1). A failure stops that queue;
commands and stdout/stderr locations are recorded before launch.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, threading, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _tag(spec, context, mode):
    return f"{mode}_{spec['kind']}_w{spec.get('width',128)}_l{spec.get('layers',2)}_n{spec.get('state_size',16)}_ctx{context}_seed{spec.get('seed',0)}"

def _command(task, gpu, out, spec, defaults, mode, context):
    cmd=[sys.executable,'-m','ssm_decode.debug_experiment','--task',task,'--device',gpu,'--output',str(out),'--kind',spec['kind'],'--mode',mode,'--width',str(spec.get('width',128)),'--layers',str(spec.get('layers',2)),'--state-size',str(spec.get('state_size',16)),'--steps',str(spec.get('steps',defaults['steps'])),'--context',str(context),'--batch-size',str(spec.get('batch_size',defaults['batch_size'])),'--seed',str(spec.get('seed',defaults.get('seed',0)))]
    if spec.get('warmstart'): cmd += ['--warmstart',spec['warmstart'],'--adapt-parameters',spec.get('adapt_parameters','all')]
    return cmd

def _matrix_jobs(matrix, output):
    defaults={'steps':matrix['steps'],'batch_size':matrix['batch_size'],'seed':matrix.get('seed',0)}; jobs=[]
    for mode,key in [('cross_session','cross_session'),('within_session_prefix','within_session_prefix_diagnostic'),('within_session_split','within_session_split_diagnostic'),('within_session_prefix','warmstart_io')]:
        for spec in matrix.get(key,[]):
            for context in spec.get('contexts',[matrix['context']]):
                for task,gpu in [('m1','cuda:0'),('m2','cuda:1')]:
                    spec=dict(spec)
                    if 'warmstart_by_task' in spec: spec['warmstart']=spec['warmstart_by_task'][task]
                    out=output/task/_tag(spec,context,mode)
                    jobs.append({'task':task,'gpu':gpu,'mode':mode,'context':context,'spec':spec,'output':str(out),'command':_command(task,gpu,out,spec,defaults,mode,context)})
    return jobs

def _run_queue(jobs, stop):
    for job in jobs:
        if stop.is_set(): return
        out=Path(job['output'])
        if out.exists(): raise FileExistsError(f"refusing to overwrite existing job path: {out}")
        out.parent.mkdir(parents=True,exist_ok=True)
        (out.parent/(out.name+'.command.json')).write_text(json.dumps(job,indent=2)+'\n')
        with (out.parent/(out.name+'.log')).open('w') as log:
            env=os.environ.copy()
            if job['spec']['kind'] in {'mamba2_official','mamba3_official'}:
                env['PYTHONPATH']=f"{ROOT / '.tools' / 'mamba_deps'}:{ROOT}" + (f":{env['PYTHONPATH']}" if env.get('PYTHONPATH') else '')
            done=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,text=True,env=env)
        if done.returncode: raise RuntimeError(f"failed job ({done.returncode}): {' '.join(job['command'])}")

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--matrix',help='JSON experiment matrix; queues M1/GPU0 and M2/GPU1');p.add_argument('--kind',default='s4d',help='single-kind mode if --matrix is omitted');p.add_argument('--steps',type=int,default=1000);p.add_argument('--context',type=int,default=128);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--mode',default='cross_session',choices=['cross_session','within_session_prefix']);a=p.parse_args();output=Path(a.output_root)
    if output.exists(): raise FileExistsError(f"refusing to overwrite existing output root: {output}")
    output.mkdir(parents=True);(output/'m1').mkdir();(output/'m2').mkdir()
    if a.matrix:
        matrix=json.loads(Path(a.matrix).read_text());jobs=_matrix_jobs(matrix,output)
    else:
        spec={'kind':a.kind};defaults={'steps':a.steps,'batch_size':a.batch_size,'seed':0};jobs=[]
        for task,gpu in [('m1','cuda:0'),('m2','cuda:1')]:
            out=output/task/_tag(spec,a.context,a.mode);jobs.append({'task':task,'gpu':gpu,'mode':a.mode,'context':a.context,'spec':spec,'output':str(out),'command':_command(task,gpu,out,spec,defaults,a.mode,a.context)})
    (output/'launch_manifest.json').write_text(json.dumps({'matrix':a.matrix,'jobs':jobs},indent=2)+'\n'); failures=[]; stop=threading.Event()
    def worker(task):
        try: _run_queue([j for j in jobs if j['task']==task],stop)
        except Exception as exc: failures.append((task,str(exc)));stop.set()
    threads=[threading.Thread(target=worker,args=(task,),daemon=False) for task in ('m1','m2')]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures: raise SystemExit('; '.join(f'{task}: {reason}' for task,reason in failures))

if __name__=='__main__': main()
