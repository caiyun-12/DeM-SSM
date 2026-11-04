"""Tiny labeled, synthetic video fixtures. NEVER use for manuscript results."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .manifest import build_manifest
from .utils import write_json


def generate_toy(folder: str | Path, image_size: int = 32,
                 n_frames: int = 3, n_subjects: int = 5, seed: int = 7) -> dict:
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    sessions=[]
    for s in range(n_subjects):
        for cls in range(3):
            file=folder/f'toy_S{s:02d}_C{cls}.avi'
            if not file.is_file():
                fourcc=cv2.VideoWriter_fourcc(*'MJPG')
                writer=cv2.VideoWriter(str(file),fourcc,10.0,(image_size,image_size))
                if not writer.isOpened():
                    raise RuntimeError('OpenCV MJPG video writer unavailable')
                for step in range(n_frames):
                    # Class pattern is intentionally trivial: tests software, not science.
                    im=np.full((image_size,image_size,3),cls*75+10,dtype=np.uint8)
                    noise=rng.integers(0,16,size=im.shape,dtype=np.uint8)
                    im=np.clip(im.astype(np.int16)+noise,0,255).astype(np.uint8)
                    cv2.circle(im,(image_size//2,image_size//2),max(1,image_size//7),
                               (int(s*20+20),int(step*30+20),int(cls*50+30)),-1)
                    writer.write(im)
                writer.release()
            sessions.append({'dataset':'toy','subject_id':f'S{s:02d}',
                             'session_id':f'C{cls}','video_path':str(file),'label':cls})
    index=folder/'video_index.csv';pd.DataFrame(sessions).to_csv(index,index=False)
    result=build_manifest(index,folder/'manifest.csv',n_frames)
    split={'train':[f'S{s:02d}' for s in range(n_subjects-2)],
           'val':[f'S{n_subjects-2:02d}'],'test':[f'S{n_subjects-1:02d}'],
           'description':'synthetic-only smoke-test partition'}
    write_json(folder/'split.json',split)
    write_json(folder/'NOT_REAL_DATA.json',
               {'warning':'Generated synthetic shapes used solely for software testing; not UTA-RLDD/DROZY.',
                'n_subjects':n_subjects,'seed':seed})
    return {**result,'split':str(folder/'split.json')}
