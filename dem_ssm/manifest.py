"""Build clip manifests from user-verified video-to-label metadata (never guesses labels)."""
from __future__ import annotations
from pathlib import Path
import cv2
import pandas as pd
from .data import read_manifest


def kss_class(value: int) -> int:
    """Paper bins: KSS 1..3 -> 0, 4..6 -> 1, 7..9 -> 2."""
    v = int(value)
    if not 1 <= v <= 9:
        raise ValueError('KSS must be in [1,9]')
    return 0 if v <= 3 else (1 if v <= 6 else 2)


def build_manifest(index_path: str | Path, output_path: str | Path, clip_length: int = 64,
                   dataset: str = '', strict: bool = True) -> dict:
    """Input CSV: dataset,subject_id,session_id,video_path,(label OR kss_score).

    Output manifest uses absolute video paths, start-frame offsets, and explicit
    session-level KSS inheritance only when KSS is provided. Short trailing
    segments are dropped, never padded or assigned invented labels.
    """
    index_path = Path(index_path).resolve()
    meta = pd.read_csv(index_path, dtype={'subject_id':str,'session_id':str})
    needed = {'subject_id','video_path'}
    if needed - set(meta.columns):
        raise ValueError(f'Video index is missing: {needed - set(meta.columns)}')
    if 'label' not in meta and 'kss_score' not in meta:
        raise ValueError('Must supply verified label or KSS score; labels are not inferred from filenames.')
    if clip_length < 1:
        raise ValueError('clip_length must be positive')
    rows, audit, errors = [], [], []
    for idx, r in meta.iterrows():
        kind = str(r.get('dataset', dataset)).strip().lower()
        if not kind or kind == 'nan':
            raise ValueError('dataset required in --dataset or index dataset column')
        subject = str(r.subject_id).strip()
        if not subject or subject.lower() == 'nan':
            raise ValueError(f'Row {idx} has missing subject_id')
        session = str(r.get('session_id', f'{idx}'))
        if session.lower() == 'nan':
            session = str(idx)
        if pd.notna(r.get('label', float('nan'))):
            label = int(r['label'])
            if label not in (0,1,2):
                raise ValueError(f'Row {idx}: label must be 0,1,2')
        else:
            label = kss_class(r['kss_score'])
        vid = Path(str(r.video_path)).expanduser()
        vid = (vid if vid.is_absolute() else index_path.parent / vid).resolve()
        if not vid.is_file():
            errors.append(f'Missing video at row {idx}: {vid}')
            continue
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            errors.append(f'Cannot decode video at row {idx}: {vid}')
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if n_frames <= 0:
            errors.append(f'No frame count at row {idx}: {vid}')
            continue
        full = n_frames // clip_length
        audit.append({'video_path':str(vid),'dataset':kind,'subject_id':subject,
                      'session_id':session,'kss_score':r.get('kss_score',float('nan')),
                      'label':label,'n_source_frames':n_frames,'fps':fps,
                      'n_full_clips':full,'dropped_tail_frames':n_frames % clip_length})
        for i in range(full):
            row = {'clip_id':f'{kind}_{subject}_{session}_{idx:04d}_{i:06d}',
                   'dataset':kind,'subject_id':subject,'session_id':session,
                   'video_path':str(vid),'label':label,
                   'start_frame':i * clip_length,'num_frames':clip_length}
            if pd.notna(r.get('landmarks_path', float('nan'))):
                p = Path(str(r.landmarks_path)).expanduser()
                row['landmarks_path'] = str((p if p.is_absolute() else index_path.parent / p).resolve())
            rows.append(row)
    if errors and strict:
        raise ValueError('Manifest failed strict checks:\n' + '\n'.join(errors[:20]))
    if not rows:
        raise ValueError('No complete clips generated')
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.clip_id.duplicated().any():
        raise ValueError('Duplicate derived clip IDs')
    df.to_csv(output_path, index=False)
    # Metadata audit separates sampling information from the model-ready manifest.
    pd.DataFrame(audit).to_csv(output_path.with_name(output_path.stem + '_sessions.csv'),index=False)
    pd.DataFrame({'error':errors}).to_csv(output_path.with_name(output_path.stem + '_errors.csv'),index=False)
    read_manifest(output_path)
    return {'clips':len(df),'subjects':df.subject_id.nunique(),
            'sessions':len(audit),'errors':len(errors),'manifest':str(output_path)}
