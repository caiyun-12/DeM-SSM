from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Config must define a model section")
    return config


def variant_config(base: dict, name: str) -> dict:
    variants = {
        "baseline": ("cnn", "static", "ema"),
        "a": ("vim", "static", "ema"),
        "b": ("cnn", "deformable", "ema"),
        "c": ("cnn", "static", "ssm"),
        "d": ("vim", "deformable", "ema"),
        "e": ("vim", "static", "ssm"),
        "dem_ssm": ("vim", "deformable", "ssm"),
    }
    k = name.lower()
    if k not in variants:
        raise ValueError(f"Unknown ablation {name!r}; choose {', '.join(variants)}")
    out = deepcopy(base)
    out["model"].update(zip(("backbone", "prior", "temporal"), variants[k]))
    out["model"]["variant"] = k
    return out
