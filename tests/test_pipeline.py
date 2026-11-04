import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dem_ssm.toy import generate_toy
from dem_ssm.data import read_manifest,ClipDataset
from dem_ssm.audit import audit_manifest
from dem_ssm.runner import train_one, evaluate_one
from dem_ssm.config import load_config
from dem_ssm.statistics import paired_subject_bootstrap


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    path=tmp_path_factory.mktemp('demssm-toy')
    generate_toy(path, image_size=32,n_frames=3,n_subjects=5)
    return path


def test_manifest_cache_and_repair(toy):
    manifest=read_manifest(toy/'manifest.csv')
    ds=ClipDataset(manifest.iloc[:1],image_size=32,n_frames=3,
                   cache_dir=toy/'cache',landmark_mode='static',detector='none')
    x=ds[0]
    assert x['video'].shape==(3,3,32,32)
    assert x['landmarks'].shape==(3,5,2)
    cache=list((toy/'cache').rglob('*.npz'))
    assert len(cache)==1
    # Cache corruption must re-decode rather than silently return stale data.
    cache[0].write_bytes(b'broken')
    repaired=ds[0]
    assert repaired['video'].shape==x['video'].shape
    assert len(list((toy/'cache').rglob('*.npz')))==1


def test_toy_partition_and_end_to_end(toy,tmp_path):
    split=json.loads((toy/'split.json').read_text())
    audit=audit_manifest(toy/'manifest.csv',split,check_video_bounds=True)
    assert audit['pass']
    cfg=load_config('configs/smoke.yaml')
    cfg['train']['cache_dir']=str(tmp_path/'cached')
    out=tmp_path/'run'
    result=train_one(toy/'manifest.csv',split,cfg,out)
    assert (out/'best.pt').is_file()
    assert (out/'last.pt').is_file()
    assert (out/'predictions_test.csv').is_file()
    assert result['overall']['n_clips']==3
    repeated=evaluate_one(out/'best.pt',toy/'manifest.csv',split,out/'repeat')
    assert repeated['overall']['n_correct']==result['overall']['n_correct']
    # Split change must be rejected even if model weights otherwise load.
    altered={**split,'train':split['val'],'val':split['train']}
    with pytest.raises(ValueError):
        evaluate_one(out/'best.pt',toy/'manifest.csv',altered,out/'illegal')


def test_bootstrap_genuine_matched_data():
    rows=[]
    for subject in range(6):
        for cls in (0,1,2):
            p=np.full(3,.1);p[cls]=.8
            rows.append({'clip_id':f'{subject}_{cls}','subject_id':str(subject),
                         'true_label':cls,'prob_0':p[0],'prob_1':p[1],'prob_2':p[2]})
    a=pd.DataFrame(rows);b=a.copy()
    out=paired_subject_bootstrap(a,b,metric='accuracy',n_boot=60,seed=0)
    assert out['point_delta_a_minus_b']==0
    assert out['n_subjects']==6
    assert out['ci95_percentile']==[0.,0.]
