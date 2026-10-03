#!/usr/bin/env python3
import argparse
import json
import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from ssm_decode.falcon_decoder import SSMFalconDecoder

parser=argparse.ArgumentParser(description='CPU FALCON SSM decoder')
parser.add_argument('--evaluation',choices=('local','remote'),required=True)
parser.add_argument('--model-path',default='/payload')
parser.add_argument('--split',choices=('m1','m2'),required=True)
parser.add_argument('--phase',choices=('minival','test'),default='test')
parser.add_argument('--batch-size',type=int,default=None)
args=parser.parse_args()
default={'m1':4,'m2':7}[args.split]; batch=default if args.batch_size is None else args.batch_size
torch.set_num_threads(2)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
task=getattr(FalconTask,args.split)
decoder=SSMFalconDecoder(FalconConfig(task=task),args.model_path,batch)
metrics = FalconEvaluator(eval_remote=args.evaluation=='remote',split=args.split,dataloader_workers=0).evaluate(decoder,phase=args.phase)
if metrics is not None:
    print(json.dumps({"local_metrics": metrics}, sort_keys=True), flush=True)
