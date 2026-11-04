# Data preparation and provenance

## Required input files

The code does **not** download or distribute UTA-RLDD or DROZY. Keep videos under `data/` or a separate local filesystem. Use an index CSV with one row per source video/session:

| Column | Requirement | Meaning |
|---|---|---|
| `dataset` | required unless CLI `--dataset` given | `uta_rldd` or `drozy` |
| `subject_id` | required, unique person identity | all clips from a person must remain in one partition |
| `session_id` | recommended | PVT session or source recording identifier |
| `video_path` | required | relative to index location, or absolute path |
| `label` | **required for UTA-RLDD** | verified 0=alert, 1=low-alert, 2=drowsy mapping |
| `kss_score` | **recommended for DROZY**, 1–9 | assigned to the *entire session*; 1–3→0, 4–6→1, 7–9→2 |

For DROZY, if both `label` and `kss_score` are present, the current builder uses `label` by preference. To avoid ambiguity, provide **only `kss_score`** if building a new DROZY label mapping. Do not extrapolate missing KSS scores, and keep the original per-session KSS metadata for audit.

## Build manifest

```bash
python -m dem_ssm.cli build-manifest --index data/uta_video_index.csv --out data/uta_manifest.csv --frames 64 --dataset uta_rldd
python -m dem_ssm.cli build-manifest --index data/drozy_video_index.csv --out data/drozy_manifest.csv --frames 64 --dataset drozy
```

Output schema is `clip_id, dataset, subject_id, session_id, video_path, label, start_frame, num_frames`. Each source video is segmented into consecutive non-overlapping 64-frame clips; any remainder is discarded and counted in `*_sessions.csv`. `*_errors.csv` records missing/unreadable videos if `--skip-invalid` is explicitly selected. The program does **not** force the historical paper clip totals; instead it audits what your actual source video files yield.

### Important source issues

- The UTA-RLDD 38/10/12 split and DROZY 9/2/3 split in the manuscript establish **counts**, not original participant IDs. `cli split` constructs *different* partitions unless author-supplied IDs are used.
- For DROZY, 64-frame windowing from source videos need not match the paper's 2,845 clip count without original FPS, exclusions and sampling manifest. Never arbitrarily subsample clips to match a target count.
- The DROZY `videos_i8` files are near infrared. This implementation explicitly converts BGR-decoded intensities to grayscale then replicates three channels. It is **not confirmed to be the original transformation**.
- UTA-RLDD raw videos need reliable, documented class labels. Filenames alone are not scientific labels.

## Face crop and five normalized landmarks

The manuscript states RetinaFace-based cropping at `1.2×` margin with MediaPipe Face Mesh for landmarks. The original exact checkpoints, detector thresholds and order of preprocessing were not supplied. Reproduce those operations externally and record their version/configuration if aiming for historical alignment.

`ClipDataset` supports:

1. `landmark_mode: from_file` (**paper.yaml default**): supply `landmarks_path` in the clip manifest. Each `.npy` or JSON file has shape `[64, 5, 2]`, ordered `[left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner]`, coordinates normalized into `[0,1]` on already cropped video frames. If a source video yields multiple clips, each clip needs its own landmarks file; do not point all clips to the same frames.
2. `landmark_mode: mediapipe`: derive reference locations from the **resized frame**, when installed with a compatible `mediapipe` version. It is a new implementation, not original evidence; optional fallback anchors are recorded.
3. `landmark_mode: static`: constant geometry for smoke tests **only**.

`detector: none` assumes source inputs were already face-cropped. `detector: haar` enables an OpenCV face detector as a clearly identified approximation (not RetinaFace). Landmark and face-detector success fractions are available from each dataset sample. The default fails rather than silently pretending missing `landmarks_path` files were recovered.

## Cache safety

Cache key fingerprints include source path, byte size, modification timestamp, clip offsets, number of frames, resolution, selected crop/landmark pipeline, and landmark-file metadata. Files use atomic replacement; malformed files are decoded again. Cached content stores **pre-augmentation frames**, ensuring each training epoch can apply a different stochastic augmentation. Never commit `data/cache`, private video frames, landmarks containing identifying biometrics, or generated predictions without review.

## Leakage checks

Before training:

```bash
python -m dem_ssm.cli audit --manifest data/uta_manifest.csv --split data/uta_split.json --check-videos --out outputs/uta_audit.json
```

The audit checks clip ID uniqueness, exact participant partitioning, any overlapping source-video frame intervals, session-label conflicts and clip bounds. This code does not automatically detect two different file paths containing the same person's face; verify duplicate identities and source-video overlaps independently.
