"""Recalculate metrics ONLY from true class labels and matched probabilities."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

PROBS = ["prob_0", "prob_1", "prob_2"]


def verify_predictions(df: pd.DataFrame) -> None:
    required = {"clip_id", "subject_id", "true_label", *PROBS}
    if required - set(df.columns):
        raise ValueError(f"Missing prediction fields: {sorted(required - set(df.columns))}")
    if df.clip_id.isna().any() or df.clip_id.duplicated().any():
        raise ValueError("Each clip_id must be present exactly once")
    if df.subject_id.isna().any():
        raise ValueError("Missing subject_id")
    labels = pd.to_numeric(df.true_label, errors="raise").to_numpy(float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Unknown true label; must be exactly 0, 1 or 2")
    prob = df[PROBS].to_numpy(dtype=np.float64)
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(prob.sum(axis=1), 1.0, rtol=0, atol=1e-4):
        raise ValueError("Probabilities must sum to 1")
    if 'pred_label' in df.columns:
        predictions = pd.to_numeric(df.pred_label, errors="raise").to_numpy(float)
        if not np.array_equal(predictions, prob.argmax(axis=1)):
            raise ValueError("pred_label disagrees with argmax(probabilities)")


def summarize(y_true, prob) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    if len(y) == 0 or p.shape != (len(y), 3):
        raise ValueError("Labels and probabilities have inconsistent shapes")
    pred = p.argmax(axis=1)
    result = {
        "n_clips": int(len(y)),
        "n_correct": int((pred == y).sum()),
        "class_counts": {str(i): int(np.count_nonzero(y == i)) for i in range(3)},
        "class_complete": bool(len(set(y)) == 3),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision_all3": float(precision_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_recall_all3": float(recall_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_f1_all3": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_f1_present_classes": float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "roc_auc_ovr": None,
        "pr_auc_ovr": None,
    }
    if result["class_complete"]:
        y_bin = np.eye(3)[y]
        result["roc_auc_ovr"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        result["pr_auc_ovr"] = float(np.mean([average_precision_score(y_bin[:, i], p[:, i]) for i in range(3)]))
    return result


def from_predictions(df: pd.DataFrame, by_subject: bool = True) -> dict:
    verify_predictions(df)
    result = {"overall": summarize(df.true_label, df[PROBS].to_numpy()), "subjects": {}}
    if by_subject:
        for subject, group in df.groupby("subject_id", sort=True):
            result["subjects"][str(subject)] = summarize(group.true_label, group[PROBS].to_numpy())
    return result


def save_report(path: str | Path, report: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
