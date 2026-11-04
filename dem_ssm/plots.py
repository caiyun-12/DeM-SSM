"""Figures from genuine prediction probability files (not paper-figure replicas)."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,auc,average_precision_score
from .metrics import PROBS,verify_predictions


def plot_multiclass_curves(prediction_path,output_dir):
    df=pd.read_csv(prediction_path,dtype={'clip_id':str,'subject_id':str})
    verify_predictions(df)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    p=df[PROBS].to_numpy(float)
    y=df.true_label.to_numpy(int)
    for kind in ('roc','pr'):
        fig,ax=plt.subplots(figsize=(6,5),dpi=140)
        curves=0
        for c in range(3):
            true=(y==c).astype(int)
            if true.min()==true.max():
                continue
            if kind=='roc':
                x,yy,_=roc_curve(true,p[:,c])
                value=auc(x,yy)
                ax.plot(x,yy,label=f'Class {c}: AUC={value:.3f}')
            else:
                yy,x,_=precision_recall_curve(true,p[:,c])
                value=average_precision_score(true,p[:,c])
                ax.plot(x,yy,label=f'Class {c}: AP={value:.3f}')
            curves+=1
        if kind=='roc':
            ax.plot([0,1],[0,1],'--',alpha=.45)
            ax.set(xlabel='False positive rate',ylabel='True positive rate',title='One-vs-rest ROC')
        else:
            ax.set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision–recall')
        if curves:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Insufficient class coverage',ha='center',va='center',transform=ax.transAxes)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output_dir/f'{kind}_per_class.png',dpi=200)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    plot_multiclass_curves(a.predictions,a.out)


if __name__=='__main__':main()
