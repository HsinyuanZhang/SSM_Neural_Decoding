"""Launch independent M1/M2 pilot runs on two GPUs."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--steps',type=int,default=1200);a=p.parse_args(argv)
 a.output_root.mkdir(parents=True,exist_ok=False); jobs=[]
 for task,gpu in [('m1','cuda:0'),('m2','cuda:1')]:
  cmd=[sys.executable,'-m','ssm_decode.experiment','--task',task,'--device',gpu,'--output',str(a.output_root/task),'--steps',str(a.steps)]
  jobs.append(subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[1]))
 codes=[j.wait() for j in jobs]
 if any(codes): raise SystemExit(f'pilot failures: {codes}')
if __name__=='__main__': main()
