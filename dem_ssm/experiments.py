"""Controlled variant/seed runners. No historical results are read or injected."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .config import load_config,variant_config
from .runner import train_one
from .metrics import from_predictions
from .utils import file_sha256,load_safe_checkpoint
import json

DEFAULT_VARIANTS=('baseline','a','b','c','d','e','dem_ssm')
PAPER_SEEDS=(0,42,1234,2026,3407)


def run_suite(manifest,split_path,config_path,out,variants=DEFAULT_VARIANTS,seeds=PAPER_SEEDS,
              resume=False) -> list[dict]:
    split=json.loads(Path(split_path).read_text(encoding='utf-8'))
    cfg=load_config(config_path)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for variant in variants:
        for seed in seeds:
            work=out/variant/f'seed_{seed}'
            work.mkdir(parents=True,exist_ok=True)
            config=variant_config(cfg,variant)
            config['train']['seed']=int(seed)
            pred_file=work/'predictions_test.csv'
            if resume and pred_file.is_file() and (work/'best.pt').is_file():
                old=load_safe_checkpoint(work/'best.pt')
                if old.get('manifest_sha256')!=file_sha256(manifest) or old.get('split')!=split or old.get('config')!=config:
                    raise ValueError(f'Refusing stale run reuse: {work}')
                # Metrics are recomputed from the saved, actual prediction output.
                report=from_predictions(pd.read_csv(pred_file,dtype={'clip_id':str,'subject_id':str}))
            else:
                report=train_one(manifest,split,config,work,
                                 resume=resume and (work/'last.pt').is_file())
            rows={'variant':variant,'seed':int(seed),'prediction_file':str(pred_file),
                  **{key:val for key,val in report['overall'].items() if not isinstance(val,(dict,list))}}
            runs.append(rows)
            pd.DataFrame(runs).to_csv(out/'run_metrics.csv',index=False)
    data=pd.DataFrame(runs)
    summary=[]
    for variant,group in data.groupby('variant',sort=False):
        row={'variant':variant,'n_runs':len(group)}
        for key in ('accuracy','macro_precision_all3','macro_recall_all3','macro_f1_all3',
                    'roc_auc_ovr','pr_auc_ovr'):
            x=pd.to_numeric(group[key],errors='coerce').dropna()
            row[f'{key}_mean']=float(x.mean()) if len(x) else None
            row[f'{key}_sd']=float(x.std(ddof=1)) if len(x)>1 else None
            row[f'{key}_n_valid']=len(x)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    return summary


def main():
    p=argparse.ArgumentParser(description='Train Table-5 seven ablations or multi-seed full models')
    p.add_argument('--manifest',required=True);p.add_argument('--split',required=True)
    p.add_argument('--config',default='configs/paper.yaml');p.add_argument('--out',required=True)
    p.add_argument('--variants',nargs='+',default=list(DEFAULT_VARIANTS))
    p.add_argument('--seeds',nargs='+',type=int,default=list(PAPER_SEEDS))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for name in args.variants:
        if name not in DEFAULT_VARIANTS:
            p.error(f'Variant {name} not in {DEFAULT_VARIANTS}')
    print(json.dumps(run_suite(args.manifest,args.split,args.config,args.out,
                               variants=args.variants,seeds=args.seeds,resume=args.resume),
                     indent=2),flush=True)


if __name__=='__main__':
    main()
