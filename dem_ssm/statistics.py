"""Paired subject-clustered bootstrap from matched *real* prediction files."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import PROBS, summarize, verify_predictions


def paired_subject_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str = 'roc_auc_ovr',
                             n_boot: int = 2000, seed: int = 0,
                             min_subjects: int = 5) -> dict:
    verify_predictions(a)
    verify_predictions(b)
    keep = ['clip_id','subject_id','true_label',*PROBS]
    merged = a[keep].merge(b[keep],on='clip_id',suffixes=('_a','_b'),validate='one_to_one')
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError('Prediction files must contain exactly the same clip IDs')
    if not (merged.subject_id_a == merged.subject_id_b).all():
        raise ValueError('Paired predictions must agree on subject_id')
    if not (merged.true_label_a == merged.true_label_b).all():
        raise ValueError('Paired predictions must have same true_label')
    subjects = merged.subject_id_a.unique().tolist()
    if len(subjects) < min_subjects:
        raise ValueError(f'Only {len(subjects)} distinct test subjects; refusing unreliable bootstrap. '
                         f'At least {min_subjects} required by this analysis guard.')
    if n_boot < 50:
        raise ValueError('Use at least 50 replicates for diagnostic resampling')
    if metric not in {'accuracy','macro_precision_all3','macro_recall_all3',
                      'macro_f1_all3','roc_auc_ovr','pr_auc_ovr'}:
        raise ValueError(f'Unsupported metric {metric}')
    def difference(frame: pd.DataFrame):
        y = frame.true_label_a.to_numpy(int)
        pa = frame[[p+'_a' for p in PROBS]].to_numpy(float)
        pb = frame[[p+'_b' for p in PROBS]].to_numpy(float)
        x = summarize(y,pa)[metric]
        z = summarize(y,pb)[metric]
        return None if x is None or z is None else float(x-z)
    point = difference(merged)
    if point is None:
        raise ValueError('Point comparison metric undefined (check class coverage).')
    parts = {s:merged[merged.subject_id_a == s] for s in subjects}
    rng = np.random.default_rng(seed)
    deltas=[]
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([parts[s] for s in draw], ignore_index=True)
        delta = difference(sample)
        if delta is not None:
            deltas.append(delta)
    if len(deltas) < 0.8*n_boot:
        raise ValueError(f'Only {len(deltas)}/{n_boot} valid replicates: class imbalance too severe')
    array = np.asarray(deltas)
    # Finite resampling correction for a two-sided sign-based p-value.
    p_value = min(1.0, float(2 * min((np.count_nonzero(array <= 0)+1)/(len(array)+1),
                                     (np.count_nonzero(array >= 0)+1)/(len(array)+1))))
    return {'metric':metric,'point_delta_a_minus_b':point,'n_clips':len(merged),
            'n_subjects':len(subjects),'n_boot_requested':n_boot,
            'n_boot_valid':len(deltas),'ci95_percentile':list(map(float,np.quantile(array,[0.025,0.975]))),
            'p_two_sided_sign_bootstrap':p_value,'seed':seed,
            'method':'paired subject-clustered bootstrap; percentile CI; descriptive not causal',
            'warning':'Resampling subjects; DO NOT compare this to historical S14 without same raw predictions.'}
