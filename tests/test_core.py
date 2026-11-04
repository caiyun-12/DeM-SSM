from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from dem_ssm.config import load_config,variant_config
from dem_ssm.manifest import kss_class
from dem_ssm.metrics import from_predictions,summarize
from dem_ssm.model import DeMSSM
from dem_ssm.splits import validate_split,make_loso
from dem_ssm.statistics import paired_subject_bootstrap


@pytest.fixture
def smoke():
    return load_config('configs/smoke.yaml')


def test_forward_backward_and_attention(smoke):
    torch.manual_seed(0)
    model=DeMSSM(**smoke['model'])
    video=torch.randn(2,3,3,32,32)
    anchor=torch.tensor([[[[.36,.4],[.64,.4],[.5,.54],[.42,.7],[.58,.7]]]*3]*2)
    detail=model(video,anchor,True)
    assert detail['logits'].shape==(2,3)
    assert torch.allclose(detail['temporal_pool_weights'].sum(1),torch.ones(2),atol=1e-5)
    (detail['logits'].sum()+.05*detail['offset_l2_px']).backward()
    assert torch.isfinite(model.classifier[-1].weight.grad).all()


@pytest.mark.parametrize('variant', ['baseline','a','b','c','d','e','dem_ssm'])
def test_all_variants(smoke,variant):
    config=variant_config(smoke,variant)
    model=DeMSSM(**config['model'])
    x=torch.rand(1,2,3,32,32)
    lm=torch.rand(1,2,5,2)
    logits,regularizer=model(x,lm)
    assert logits.shape==(1,3)
    assert torch.isfinite(logits).all()
    assert regularizer>=0


def test_kss_binning():
    assert [kss_class(k) for k in range(1,10)] == [0,0,0,1,1,1,2,2,2]
    with pytest.raises(ValueError):kss_class(10)


def test_split_leakage_fails():
    df=pd.DataFrame({'subject_id':['A','B','C'],'label':[0,1,2]})
    with pytest.raises(ValueError,match='leakage'):
        validate_split(df,{'train':['A','B'],'val':['B'],'test':['C']})


def test_missing_class_makes_auc_undefined():
    y=[0,1,0,1]
    prob=np.asarray([[.8,.1,.1],[.1,.8,.1],[.6,.3,.1],[.1,.8,.1]])
    out=summarize(y,prob)
    assert not out['class_complete']
    assert out['roc_auc_ovr'] is None
    assert out['pr_auc_ovr'] is None
    assert out['n_correct']==4


def test_no_invented_significance_for_three_subjects():
    rows=[]
    for subject in range(3):
        for cls in range(3):
            v=[.05,.05,.05];v[cls]=.9
            rows.append({'clip_id':f'{subject}-{cls}','subject_id':str(subject),
                         'true_label':cls,**dict(zip(['prob_0','prob_1','prob_2'],v))})
    df=pd.DataFrame(rows)
    with pytest.raises(ValueError,match='subjects'):
        paired_subject_bootstrap(df,df,metric='accuracy',n_boot=50)


def test_probabilities_validation():
    df=pd.DataFrame([{'clip_id':'A','subject_id':'1','true_label':0,
                      'prob_0':.5,'prob_1':.5,'prob_2':.5}])
    with pytest.raises(ValueError,match='sum'):
        from_predictions(df)
