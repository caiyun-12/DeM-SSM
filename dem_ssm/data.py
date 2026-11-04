"""Video decoding and validated per-clip cache with explicit landmark provenance."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FALLBACK_ANCHORS = np.array([
    [0.36, 0.39],  # left-eye region, image-coordinate convention
    [0.64, 0.39],  # right-eye region
    [0.50, 0.54],  # nose
    [0.42, 0.70],  # left mouth corner
    [0.58, 0.70],  # right mouth corner
], dtype=np.float32)
LM_VERSION = "clipdecode-v3-20261103"


def resolve_path(path: str, parent: Path) -> Path:
    p = Path(str(path)).expanduser()
    return p.resolve() if p.is_absolute() else (parent / p).resolve()


def read_manifest(path: str | Path) -> pd.DataFrame:
    filename = Path(path)
    df = pd.read_csv(filename, dtype={"subject_id": str, "session_id": str, "clip_id": str})
    required = {"clip_id", "video_path", "dataset", "subject_id", "label", "start_frame", "num_frames"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if df["clip_id"].isna().any() or df["clip_id"].duplicated().any():
        raise ValueError("clip_id must be non-null and unique")
    if df["subject_id"].isna().any() or (df["subject_id"].str.strip() == "").any():
        raise ValueError("subject_id cannot be empty")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df.label.unique()).issubset({0, 1, 2}):
        raise ValueError("Only labels {0,1,2} allowed")
    for c in ("start_frame", "num_frames"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    if (df.start_frame < 0).any() or (df.num_frames < 1).any():
        raise ValueError("start_frame >=0 and num_frames >=1 required")
    df["video_path"] = df["video_path"].map(lambda p: str(resolve_path(p, filename.parent)))
    if "landmarks_path" in df:
        df["landmarks_path"] = df["landmarks_path"].fillna("").map(
            lambda p: str(resolve_path(p, filename.parent)) if str(p).strip() else "")
    return df


def load_landmarks(path: str, n_frames: int) -> np.ndarray:
    file = Path(path)
    if file.suffix.lower() == ".npy":
        xy = np.load(file, allow_pickle=False)
    elif file.suffix.lower() == ".json":
        xy = np.asarray(json.loads(file.read_text(encoding="utf-8")), dtype=np.float32)
    else:
        raise ValueError("landmarks_path must point to a .npy or .json array")
    if xy.shape != (n_frames, 5, 2) or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
        raise ValueError(f"Expected normalized landmarks [{n_frames},5,2] within [0,1]")
    return np.asarray(xy, dtype=np.float32)


def _face_crop(rgb: np.ndarray, detector: str, classifier: Any, margin: float) -> tuple[np.ndarray, bool]:
    if detector == "none":
        return rgb, False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = classifier.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return rgb, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * margin
    top, bottom = max(0, int(cy - s / 2)), min(rgb.shape[0], int(cy + s / 2))
    left, right = max(0, int(cx - s / 2)), min(rgb.shape[1], int(cx + s / 2))
    return rgb[top:bottom, left:right], True


class ClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 224,
                 n_frames: int = 64, cache_dir: str | Path = "data/cache",
                 landmark_mode: str = "from_file", detector: str = "none",
                 training: bool = False, jitter: bool = True, crop_margin: float = 1.2):
        self.df = manifest.reset_index(drop=True).copy()
        if len(self.df) == 0:
            raise ValueError("No clips provided")
        if not (self.df.num_frames == n_frames).all():
            raise ValueError(f"All manifest num_frames must equal config frames={n_frames}")
        if landmark_mode not in ("from_file", "mediapipe", "static"):
            raise ValueError("landmark_mode must be from_file/mediapipe/static")
        if detector not in ("none", "haar"):
            raise ValueError("detector must be none/haar")
        self.image_size, self.n_frames = image_size, n_frames
        self.landmark_mode, self.detector = landmark_mode, detector
        self.training, self.jitter, self.margin = training, jitter, crop_margin
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = None
        if detector == "haar":
            self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.classifier.empty():
                raise RuntimeError("Haar classifier was not loaded")
        self.face_mesh = None
        if landmark_mode == "mediapipe":
            try:
                import mediapipe as mp
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True)
            except (ImportError, AttributeError) as exc:
                raise RuntimeError("MediaPipe FaceMesh unavailable. Use a compatible mediapipe version or precomputed landmark files.") from exc
        if landmark_mode == "from_file" and ("landmarks_path" not in self.df or (self.df.landmarks_path == "").any()):
            raise ValueError("from_file landmark mode requires landmarks_path for every clip")

    def __len__(self) -> int:
        return len(self.df)

    def _cache_key(self, row) -> tuple[str, str]:
        video = Path(row.video_path)
        if not video.is_file():
            raise FileNotFoundError(video)
        stat = video.stat()
        payload = {"version": LM_VERSION, "video": str(video), "size": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns, "start": int(row.start_frame),
                   "frames": self.n_frames, "image": self.image_size,
                   "landmarks": self.landmark_mode, "detector": self.detector,
                   "margin": self.margin, "dataset": str(row.dataset)}
        if self.landmark_mode == "from_file":
            lm = Path(row.landmarks_path)
            if not lm.is_file():
                raise FileNotFoundError(lm)
            s = lm.stat()
            payload["lm_file"] = [str(lm), s.st_size, s.st_mtime_ns]
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return fingerprint, fingerprint[:2] + "/" + fingerprint + ".npz"

    def _build(self, row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cap = cv2.VideoCapture(str(row.video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video {row.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.start_frame))
        frames, xy, success = [], [], []
        recorded_lm = load_landmarks(row.landmarks_path, self.n_frames) if self.landmark_mode == "from_file" else None
        try:
            for i in range(self.n_frames):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Short clip: {row.clip_id} at frame {i}")
                if str(row.dataset).lower() == "drozy":
                    # An EXPLICIT, documented approximation; original channel method is unknown.
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb, crop_ok = _face_crop(rgb, self.detector, self.classifier, self.margin)
                rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                if recorded_lm is not None:
                    points, found = recorded_lm[i], True
                elif self.face_mesh is not None:
                    result = self.face_mesh.process(rgb)
                    if result.multi_face_landmarks:
                        coords = result.multi_face_landmarks[0].landmark
                        # Image left/right and FaceMesh indexing are convention-dependent.
                        points = np.array([[coords[j].x, coords[j].y] for j in (33, 263, 1, 61, 291)], dtype=np.float32)
                        points = np.clip(points, 0.0, 1.0)
                        found = True
                    else:
                        points, found = FALLBACK_ANCHORS.copy(), False
                else:
                    points, found = FALLBACK_ANCHORS.copy(), False
                frames.append(rgb)
                xy.append(points)
                success.append(bool(found))
        finally:
            cap.release()
        return np.stack(frames).astype(np.uint8), np.stack(xy).astype(np.float32), np.array(success, dtype=np.bool_)

    def _cached(self, row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fingerprint, relative = self._cache_key(row)
        target = self.cache_dir / relative
        if target.is_file():
            try:
                with np.load(target, allow_pickle=False) as z:
                    if str(z["fingerprint"].item()) == fingerprint:
                        a, p, ok = z["frames"], z["landmarks"], z["found"]
                        valid = (
                            a.shape == (self.n_frames, self.image_size, self.image_size, 3)
                            and a.dtype == np.uint8
                            and p.shape == (self.n_frames, 5, 2)
                            and np.issubdtype(p.dtype, np.floating)
                            and np.isfinite(p).all()
                            and (p >= 0).all() and (p <= 1).all()
                            and ok.shape == (self.n_frames,)
                            and ok.dtype == np.bool_
                        )
                        if valid:
                            return a, p, ok
            except (ValueError, OSError, KeyError):
                pass
        a, p, ok = self._build(row)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + f".{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(temp, frames=a, landmarks=p, found=ok, fingerprint=np.array(fingerprint))
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return a, p, ok

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        frames, points, found = self._cached(row)
        frames, points = frames.copy(), points.copy()
        if self.training and random.random() < 0.5:
            frames = np.ascontiguousarray(frames[:, :, ::-1, :])
            points[..., 0] = 1 - points[..., 0]
            points[:, [0, 1]] = points[:, [1, 0]]
            points[:, [3, 4]] = points[:, [4, 3]]
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        if self.training:
            scale = random.uniform(0.90, 1.10)
            tensor = (tensor * scale).clamp(0, 1)
            if self.jitter and str(row.dataset).lower() != "drozy":
                # Single scalar brightness/contrast preserves the input modality.
                brightness = random.uniform(-0.04, 0.04)
                tensor = (tensor + brightness).clamp(0, 1)
        tensor = (tensor - tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return {"video": tensor, "landmarks": torch.from_numpy(points),
                "label": torch.tensor(int(row.label), dtype=torch.long),
                "clip_id": str(row.clip_id), "subject_id": str(row.subject_id),
                "landmark_success_fraction": float(found.mean())}
