"""Dataset, split and prediction provenance checks."""
from __future__ import annotations
from pathlib import Path
import cv2
import pandas as pd
from .data import read_manifest
from .splits import split_data
from .utils import write_json


def audit_manifest(manifest: str | Path, split: dict | None = None,
                   check_video_bounds: bool = False) -> dict:
    df=read_manifest(manifest)
    conflicts=[]
    # Session-level KSS inheritance requires all clips from a session share label.
    if 'session_id' in df.columns:
        tally=df.groupby(['dataset','subject_id','session_id']).label.nunique()
        conflicts=[{'dataset':str(idx[0]),'subject_id':str(idx[1]),'session_id':str(idx[2])}
                   for idx in tally[tally>1].index]
    result={'n_clips':len(df),'n_subjects':df.subject_id.nunique(),
            'n_video_paths':df.video_path.nunique(),
            'label_counts':{str(k):int((df.label==k).sum()) for k in range(3)},
            'n_session_label_conflicts':len(conflicts),'session_conflicts':conflicts,
            'overlapping_clip_intervals':[],
            'split_audit':None,'out_of_bounds':[]}
    # Distinct clips from same source must never overlap frames, even if their IDs differ.
    for path,part in df.groupby('video_path'):
        items=part.sort_values('start_frame')
        ends=items.start_frame.to_numpy() + items.num_frames.to_numpy()
        starts=items.start_frame.to_numpy()
        for i in range(1,len(items)):
            if starts[i] < ends[i-1]:
                result['overlapping_clip_intervals'].append({
                    'video_path':path,'clip_a':str(items.iloc[i-1].clip_id),
                    'clip_b':str(items.iloc[i].clip_id)})
    if split is not None:
        parts=split_data(df,split)
        result['split_audit']={k:{'subjects':sorted(x.subject_id.unique().tolist()),
                                  'clips':len(x),'class_counts':{str(i):int((x.label==i).sum()) for i in range(3)}}
                               for k,x in parts.items()}
    if check_video_bounds:
        sizes={}
        for filename in sorted(df.video_path.unique()):
            cap=cv2.VideoCapture(filename)
            if not cap.isOpened():
                result['out_of_bounds'].append({'video':filename,'error':'cannot open'})
                continue
            sizes[filename]=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        for row in df.itertuples():
            if row.video_path in sizes and row.start_frame+row.num_frames>sizes[row.video_path]:
                result['out_of_bounds'].append({'clip_id':row.clip_id,'video':row.video_path})
    result['pass']=(not conflicts and not result['overlapping_clip_intervals'] and not result['out_of_bounds'])
    return result
