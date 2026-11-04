"""Training, evaluation and LOSO. Predictions are never synthesized."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import ClipDataset, read_manifest
from .metrics import PROBS, from_predictions, save_report
from .model import DeMSSM
from .splits import split_data, validate_split, make_loso
from .utils import set_seed, get_device, write_json, atomic_save_torch, load_safe_checkpoint, file_sha256


def build_model(config: dict) -> DeMSSM:
    return DeMSSM(**config['model'])


def make_loader(part: pd.DataFrame, config: dict, training: bool, seed: int = 0) -> DataLoader:
    cfg, tc = config['model'],config['train']
    ds = ClipDataset(part, image_size=cfg['image_size'], n_frames=cfg['frames'],
                     cache_dir=tc['cache_dir'], landmark_mode=tc['landmark_mode'],
                     detector=tc['detector'], training=training,
                     crop_margin=tc.get('crop_margin',1.2))
    n_workers = int(tc.get('num_workers',0))
    if n_workers and tc['landmark_mode'] == 'mediapipe':
        raise ValueError('MediaPipe FaceMesh is process-dependent; use num_workers=0 or precomputed landmarks')
    generator = torch.Generator().manual_seed(seed)
    def worker_init(worker_id):
        base = torch.initial_seed() % 2**32
        np.random.seed(base); random.seed(base)
    return DataLoader(ds, batch_size=int(tc['batch_size']),shuffle=training,
                      num_workers=n_workers,worker_init_fn=worker_init if n_workers else None,
                      generator=generator,pin_memory=False,drop_last=False)


def _autocast_context(device, use_amp: bool):
    return torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type=='cuda'))


def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            amp: bool = False, seed: int | None = None) -> pd.DataFrame:
    model.eval()
    records=[]
    with torch.inference_mode():
        for batch in loader:
            video = batch['video'].to(device)
            lm = batch['landmarks'].to(device)
            with _autocast_context(device,amp):
                logits,_ = model(video,lm)
            prob = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for i,p in enumerate(prob):
                records.append({'clip_id':str(batch['clip_id'][i]),
                                'subject_id':str(batch['subject_id'][i]),
                                'true_label':int(batch['label'][i]),
                                'pred_label':int(np.argmax(p)),
                                **{k:float(v) for k,v in zip(PROBS,p)},
                                'seed':seed})
    return pd.DataFrame.from_records(records)


def eval_loss(model: nn.Module, loader: DataLoader, device, amp=False) -> float:
    model.eval(); total=0.0; seen=0
    with torch.inference_mode():
        for batch in loader:
            video=batch['video'].to(device)
            lm=batch['landmarks'].to(device)
            label=batch['label'].to(device)
            with _autocast_context(device,amp):
                logits,_=model(video,lm)
                loss=nn.functional.cross_entropy(logits,label)
            n=len(label)
            total+=float(loss)*n;seen+=n
    return total/seen


def train_one(manifest: str | Path, split: dict, config: dict, output: str | Path,
              resume: bool = False, init_checkpoint: str | None = None,
              do_test: bool = True) -> dict:
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cfg=deepcopy(config)
    tc=cfg['train']; seed=int(tc.get('seed',0))
    set_seed(seed)
    device=get_device(str(tc.get('device','auto')))
    manifest=Path(manifest).resolve()
    df=read_manifest(manifest)
    validate_split(df,split)
    parts=split_data(df,split)
    # Explicit split audit: includes train/val/test subjects and class coverage.
    split_audit={'manifest_sha256':file_sha256(manifest),
                 'manifest':str(manifest),'split':split,
                 'partitions':{p:{'n_clips':len(f),'n_subjects':f.subject_id.nunique(),
                                  'class_counts':{str(i):int((f.label==i).sum()) for i in range(3)}}
                               for p,f in parts.items()}}
    write_json(output/'split_audit.json',split_audit)
    write_json(output/'run_config.json',cfg)
    if any(n==0 for n in split_audit['partitions']['train']['class_counts'].values()):
        warnings.warn('Train partition lacks one or more classes; results may be uninformative.')
    model=build_model(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=float(tc['lr']), weight_decay=float(tc['weight_decay']))
    epochs=int(tc['epochs'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,epochs))
    use_amp=bool(tc.get('amp',True)) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    start_epoch=0; best=float('inf'); best_epoch=-1; patience_counter=0; history=[]
    if resume:
        cp=output/'last.pt'
        if not cp.is_file():
            raise FileNotFoundError(f'--resume specified but {cp} does not exist')
        saved=load_safe_checkpoint(cp)
        if saved['manifest_sha256']!=split_audit['manifest_sha256'] or saved['split']!=split:
            raise ValueError('Refusing resume: manifest hash or split changed')
        if saved['config'] != cfg:
            raise ValueError('Refusing resume: config changed (use original run config)')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        scaler.load_state_dict(saved['scaler_state'])
        start_epoch=int(saved['epoch'])+1
        best=float(saved['best_val_loss']); best_epoch=int(saved['best_epoch'])
        patience_counter=int(saved['patience_counter'])
        history=list(saved.get('history',[]))
    elif init_checkpoint:
        other=load_safe_checkpoint(init_checkpoint)
        state=other.get('model_state',other.get('state_dict',other))
        result=model.load_state_dict(state,strict=False)
        write_json(output/'initialization_audit.json',
                   {'source_checkpoint':str(init_checkpoint),'missing_keys':list(result.missing_keys),
                    'unexpected_keys':list(result.unexpected_keys),
                    'warning':'Non-exact state keys are not an official pretrained-weight match'})
    min_update=int(tc.get('grad_accum',1))
    if min_update<1:
        raise ValueError('grad_accum must be >=1')
    for epoch in range(start_epoch,epochs):
        set_seed(seed + epoch * 7919)
        train_loader=make_loader(parts['train'],cfg,training=True,seed=seed+epoch)
        valid_loader=make_loader(parts['val'],cfg,training=False,seed=seed)
        model.train();optimizer.zero_grad(set_to_none=True)
        train_sum=0.0;total=0
        for step,batch in enumerate(train_loader):
            x=batch['video'].to(device)
            xy=batch['landmarks'].to(device)
            label=batch['label'].to(device)
            with _autocast_context(device,use_amp):
                logits,offset_penalty=model(x,xy)
                loss=nn.functional.cross_entropy(logits,label)+float(tc['offset_lambda'])*offset_penalty
            scaler.scale(loss/min_update).backward()
            if (step+1)%min_update==0 or (step+1)==len(train_loader):
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            train_sum+=float(loss.detach())*len(label);total+=len(label)
        val_loss=eval_loss(model,valid_loader,device,amp=use_amp)
        scheduler.step()
        history.append({'epoch':epoch,'train_loss':train_sum/max(total,1),'val_loss':val_loss,
                        'lr':float(optimizer.param_groups[0]['lr'])})
        improved=val_loss<best-1e-8
        if improved:
            best=val_loss;best_epoch=epoch;patience_counter=0
        else:
            patience_counter+=1
        state={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
               'scheduler_state':scheduler.state_dict(),'scaler_state':scaler.state_dict(),
               'epoch':epoch,'best_val_loss':best,'best_epoch':best_epoch,
               'patience_counter':patience_counter,'history':history,
               'config':cfg,'split':split,'manifest_sha256':split_audit['manifest_sha256'],
               'code_origin':'independent_reconstruction_not_original_author_code'}
        atomic_save_torch(output/'last.pt',state)
        if improved:
            atomic_save_torch(output/'best.pt',state)
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        print(f'epoch {epoch+1}/{epochs} train_loss={train_sum/max(total,1):.5f} '
              f'val_loss={val_loss:.5f} best_epoch={best_epoch+1}',flush=True)
        if patience_counter>=int(tc['patience']):
            print('Early stopping on validation loss.',flush=True)
            break
    if do_test:
        return evaluate_one(output/'best.pt',manifest,split,output,
                            partition='test',allow_new_manifest=False)
    return {'best_val_loss':best,'best_epoch':best_epoch,'output':str(output)}


def evaluate_one(checkpoint: str | Path,manifest: str | Path,split: dict,
                 output: str | Path, partition: str = 'test',
                 allow_new_manifest: bool = False) -> dict:
    if partition not in ('train','val','test'):
        raise ValueError('partition must be train/val/test')
    saved=load_safe_checkpoint(checkpoint)
    cfg=saved['config']; device=get_device(str(cfg['train'].get('device','auto')))
    manifest=Path(manifest).resolve()
    if not allow_new_manifest and saved.get('manifest_sha256')!=file_sha256(manifest):
        raise ValueError('Manifest checksum differs from the saved training manifest')
    if not allow_new_manifest and saved['split']!=split:
        raise ValueError('Split differs from training checkpoint')
    df=read_manifest(manifest);parts=split_data(df,split)
    model=build_model(cfg).to(device)
    model.load_state_dict(saved['model_state'],strict=True)
    loader=make_loader(parts[partition],cfg,training=False)
    pred=predict(model,loader,device,amp=bool(cfg['train'].get('amp',True)),
                 seed=int(cfg['train'].get('seed',0)))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pred_file=output/f'predictions_{partition}.csv'
    pred.to_csv(pred_file,index=False,float_format='%.9g')
    report=from_predictions(pred)
    report['provenance']={'checkpoint':str(checkpoint),'manifest':str(manifest),
                          'partition':partition,'predictions':str(pred_file),
                          'note':'New metrics computed from this run; no historical manuscript results imported'}
    save_report(output/f'metrics_{partition}.json',report)
    print(f'Evaluated {len(pred)} real manifest clips; wrote {pred_file}',flush=True)
    return report


def train_loso(manifest: str | Path,config: dict,output: str | Path,
               only_subject: str | None = None, resume: bool = False) -> dict:
    df=read_manifest(manifest)
    subjects=sorted(df.subject_id.unique().tolist())
    if len(subjects)!=14:
        raise ValueError('Full DROZY LOSO requires 14 distinct participants; no invented subjects')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    selected=[only_subject] if only_subject else subjects
    if any(s not in subjects for s in selected):
        raise ValueError('Unknown LOSO subject')
    all_preds=[];fold_stats=[]
    for subject in selected:
        split=make_loso(df,subject)
        work=output/f'fold_{subject}'
        prediction=work/'predictions_test.csv'
        if prediction.is_file() and not resume:
            raise FileExistsError(f'Existing LOSO fold {work}; use a different output or --resume')
        if resume and (work/'best.pt').is_file() and prediction.is_file():
            archived=load_safe_checkpoint(work/'best.pt')
            if archived.get('manifest_sha256')!=file_sha256(manifest) or archived.get('split')!=split:
                raise ValueError(f'Existing LOSO fold {subject} has changed manifest or subject split')
            frame=pd.read_csv(prediction, dtype={'subject_id':str,'clip_id':str})
            expected=set(df.loc[df.subject_id==subject,'clip_id'])
            if set(frame.clip_id)!=expected:
                raise ValueError(f'Existing fold {subject} has wrong prediction clip IDs')
        else:
            cfg=deepcopy(config)
            cfg['train']['seed']=int(config['train']['seed'])
            train_one(manifest,split,cfg,work,resume=resume and (work/'last.pt').is_file())
            frame=pd.read_csv(prediction,dtype={'subject_id':str,'clip_id':str})
        all_preds.append(frame)
        report=from_predictions(frame)
        row={'test_subject':subject,'n_clips':len(frame),**report['overall']}
        row.pop('confusion_matrix',None)
        fold_stats.append(row)
    preds=pd.concat(all_preds,ignore_index=True)
    if preds.clip_id.duplicated().any():
        raise ValueError('Duplicate LOSO clip_id: data leakage or reruns collide')
    preds.to_csv(output/'loso_oof_predictions.csv',index=False)
    pd.DataFrame(fold_stats).to_json(output/'loso_folds.json',orient='records',indent=2)
    report=from_predictions(preds)
    report['warning']='Pooled OOF metrics differ from 9-fold conditional table in manuscript. Fold-level ROC-AUC requires all three true classes.'
    save_report(output/'loso_pooled_oof.json',report)
    return {'n_evaluated_subjects':len(selected),'n_oof_clips':len(preds),
            'pooled_report':str(output/'loso_pooled_oof.json')}
