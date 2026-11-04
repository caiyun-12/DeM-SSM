"""Small deterministic and safe I/O helpers."""
from __future__ import annotations
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch reports CUDA unavailable.')
    return device


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.write-', suffix='.json', delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            f.write('\n')
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def atomic_save_torch(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.ckpt-', suffix='.pt', delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_safe_checkpoint(path: str | Path, map_location='cpu') -> dict:
    # Restrict unpickling; use only trusted locally generated checkpoints.
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError('Checkpoint is not a dict.')
    return obj


def file_sha256(path: str | Path, block_size: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(block_size), b''):
            h.update(data)
    return h.hexdigest()
