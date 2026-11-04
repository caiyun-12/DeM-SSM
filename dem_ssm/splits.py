from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def make_fixed(df: pd.DataFrame, dataset: str, seed: int = 2026) -> dict:
    """RECONSTRUCTED random partition; original manuscript subject IDs unavailable."""
    datasets = {"uta_rldd": (38, 10, 12), "drozy": (9, 2, 3)}
    if dataset not in datasets:
        raise ValueError("dataset must be uta_rldd or drozy")
    subjects = sorted(df.subject_id.unique().tolist())
    train_n, val_n, test_n = datasets[dataset]
    if len(subjects) != train_n + val_n + test_n:
        raise ValueError(f"{dataset} requires {sum(datasets[dataset])} participants, got {len(subjects)}")
    rng = np.random.default_rng(seed)
    subjects = rng.permutation(subjects).tolist()
    return {"train": subjects[:train_n], "val": subjects[train_n:train_n + val_n],
            "test": subjects[train_n + val_n:],
            "description": "Reconstructed random partition; NOT recovered paper subject IDs", "seed": seed}


def make_loso(df: pd.DataFrame, subject: str) -> dict:
    all_subjects = sorted(df.subject_id.unique().tolist())
    if len(all_subjects) != 14:
        raise ValueError("LOSO configured for DROZY's 14 participants")
    if subject not in all_subjects:
        raise ValueError(f"Unknown test subject: {subject}")
    idx = all_subjects.index(subject)
    others = [x for x in all_subjects if x != subject]
    # Deterministic rotating 11/2 split. Not the unavailable original rotation.
    pos = idx % len(others)
    validation = [others[pos], others[(pos + 1) % len(others)]]
    train = [x for x in others if x not in validation]
    return {"train": train, "val": validation, "test": [subject],
            "description": "Reconstructed LOSO rotation; not author-supplied partition"}


def validate_split(df: pd.DataFrame, split: dict) -> None:
    parts = [set(map(str, split.get(k, []))) for k in ("train", "val", "test")]
    if any(not group for group in parts):
        raise ValueError("train/val/test subject lists cannot be empty")
    if (parts[0] & parts[1]) or (parts[0] & parts[2]) or (parts[1] & parts[2]):
        raise ValueError("Subject leakage: train/val/test lists overlap")
    available = set(df.subject_id.astype(str))
    if set.union(*parts) != available:
        raise ValueError("Subject lists must partition all manifest participants exactly")


def split_data(df: pd.DataFrame, split: dict) -> dict[str, pd.DataFrame]:
    validate_split(df, split)
    return {k: df.loc[df.subject_id.isin(split[k])].copy() for k in ("train", "val", "test")}


def save_json(path: str | Path, value: dict) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
