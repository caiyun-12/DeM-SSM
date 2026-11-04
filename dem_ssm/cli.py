"""Command line interface for reproducible DeM-SSM research reconstruction."""
from __future__ import annotations
import argparse
from copy import deepcopy
import glob
import json
from pathlib import Path
import pandas as pd
from .config import load_config,variant_config
from .metrics import from_predictions,save_report
from .splits import make_fixed,validate_split,save_json
from .data import read_manifest
from .statistics import paired_subject_bootstrap
from .runner import train_one,evaluate_one,train_loso
from .manifest import build_manifest
from .toy import generate_toy
from .audit import audit_manifest
from .utils import write_json


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _add_config(sub):
    sub.add_argument('--config',default='configs/paper.yaml')
    sub.add_argument('--variant',default=None,choices=['baseline','a','b','c','d','e','dem_ssm'])
    sub.add_argument('--seed',type=int,default=None)
    sub.add_argument('--device',default=None,choices=['auto','cpu','cuda'])


def _resolve_config(args):
    config=load_config(args.config)
    if getattr(args,'variant',None):
        config=variant_config(config,args.variant)
    if getattr(args,'seed',None) is not None:
        config['train']['seed']=args.seed
    if getattr(args,'device',None) is not None:
        config['train']['device']=args.device
    return config


def get_parser():
    p=argparse.ArgumentParser(prog='python -m dem_ssm.cli',
                              description='Independent DeM-SSM code reconstruction; NOT original training artifacts.')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('toy-data',help='Generate tiny synthetic videos for smoke testing only')
    q.add_argument('--out',default='data/toy');q.add_argument('--image-size',type=int,default=32)
    q.add_argument('--frames',type=int,default=3)
    q=sub.add_parser('build-manifest',help='Create nonoverlapping 64-frame clips from verified video index')
    q.add_argument('--index',required=True);q.add_argument('--out',required=True)
    q.add_argument('--frames',type=int,default=64);q.add_argument('--dataset',default='')
    q.add_argument('--skip-invalid',action='store_true')
    q=sub.add_parser('split',help='Generate subject-disjoint split; not original subject identities')
    q.add_argument('--manifest',required=True);q.add_argument('--dataset',required=True,choices=['uta_rldd','drozy'])
    q.add_argument('--out',required=True);q.add_argument('--seed',type=int,default=2026)
    q=sub.add_parser('audit',help='Audit manifest/split for overlap and class coverage')
    q.add_argument('--manifest',required=True);q.add_argument('--split')
    q.add_argument('--check-videos',action='store_true');q.add_argument('--out')
    q=sub.add_parser('train',help='Train/restart, checkpoint best & last, export predictions')
    q.add_argument('--manifest',required=True);q.add_argument('--split',required=True);q.add_argument('--out',required=True)
    q.add_argument('--resume',action='store_true');q.add_argument('--init-checkpoint')
    _add_config(q)
    q=sub.add_parser('evaluate',help='Evaluate trained checkpoint and recompute real prediction metrics')
    q.add_argument('--checkpoint',required=True);q.add_argument('--manifest',required=True)
    q.add_argument('--split',required=True);q.add_argument('--out',required=True)
    q.add_argument('--partition',choices=['train','val','test'],default='test')
    q=sub.add_parser('loso',help='Train DROZY LOSO folds (14 subjects, 11/2 train/validation)')
    q.add_argument('--manifest',required=True);q.add_argument('--out',required=True)
    q.add_argument('--only-subject');q.add_argument('--resume',action='store_true')
    _add_config(q)
    q=sub.add_parser('metrics',help='Calculate metrics from an actual predictions CSV')
    q.add_argument('--predictions',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('paired-bootstrap',help='Real paired predictions; subject-cluster bootstrap')
    q.add_argument('--model-a',required=True);q.add_argument('--model-b',required=True)
    q.add_argument('--metric',default='roc_auc_ovr');q.add_argument('--replicates',type=int,default=2000)
    q.add_argument('--seed',type=int,default=0);q.add_argument('--out',required=True)
    q=sub.add_parser('aggregate-seeds',help='Aggregate only actual independent-run predictions')
    q.add_argument('--predictions',nargs='+',required=True,help='Explicit prediction CSV paths')
    q.add_argument('--out',required=True)
    return p


def main(argv=None):
    args=get_parser().parse_args(argv)
    if args.command=='toy-data':
        result=generate_toy(args.out,args.image_size,args.frames)
    elif args.command=='build-manifest':
        result=build_manifest(args.index,args.out,args.frames,args.dataset,not args.skip_invalid)
    elif args.command=='split':
        df=read_manifest(args.manifest)
        split=make_fixed(df,args.dataset,args.seed)
        validate_split(df,split)
        save_json(args.out,split);result=split
    elif args.command=='audit':
        result=audit_manifest(args.manifest,_json(args.split) if args.split else None,args.check_videos)
        if args.out:write_json(args.out,result)
    elif args.command=='train':
        cfg=_resolve_config(args)
        result=train_one(args.manifest,_json(args.split),cfg,args.out,
                         resume=args.resume,init_checkpoint=args.init_checkpoint)
    elif args.command=='evaluate':
        result=evaluate_one(args.checkpoint,args.manifest,_json(args.split),args.out,
                            partition=args.partition)
    elif args.command=='loso':
        result=train_loso(args.manifest,_resolve_config(args),args.out,args.only_subject,args.resume)
    elif args.command=='metrics':
        pred=pd.read_csv(args.predictions,dtype={'clip_id':str,'subject_id':str})
        result=from_predictions(pred)
        save_report(args.out,result)
    elif args.command=='paired-bootstrap':
        a=pd.read_csv(args.model_a,dtype={'clip_id':str,'subject_id':str})
        b=pd.read_csv(args.model_b,dtype={'clip_id':str,'subject_id':str})
        result=paired_subject_bootstrap(a,b,metric=args.metric,n_boot=args.replicates,seed=args.seed)
        write_json(args.out,result)
    elif args.command=='aggregate-seeds':
        rows=[]
        base_ids=None
        for path in args.predictions:
            df=pd.read_csv(path,dtype={'clip_id':str,'subject_id':str})
            m=from_predictions(df)['overall']
            signature=df[['clip_id','subject_id','true_label']].sort_values('clip_id').reset_index(drop=True)
            if base_ids is not None and not base_ids.equals(signature):
                raise ValueError('Different prediction files do not share identical clip/subject/label records')
            base_ids=signature
            rows.append({'path':str(path),**{k:v for k,v in m.items() if not isinstance(v,(list,dict))}})
        stats=pd.DataFrame(rows)
        stats.to_csv(args.out,index=False)
        result={'n_runs':len(rows),'per_run':str(args.out),
                'summary':{col:{'mean':float(stats[col].dropna().mean()),
                                'sd_sample':float(stats[col].dropna().std(ddof=1)) if stats[col].notna().sum()>1 else None,
                                'n_valid':int(stats[col].notna().sum())}
                           for col in stats.select_dtypes(include='number') if col not in ('n_clips','n_correct')}}
        write_json(Path(args.out).with_suffix('.json'),result)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))


if __name__=='__main__':
    main()
