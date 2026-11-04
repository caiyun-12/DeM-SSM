# DeM-SSM

**Auditable, independent code reconstruction** for the architecture described in:

> *DeM-SSM: A Spatio-Temporal State-Space Model with Deformable Prior for Robust Driver Drowsiness Detection* (manuscript supplied 2026-11-03).

This repository implements a research workflow for **three-class driver-vigilance classification from facial video**: global bidirectional vision selective-state-space scanning, a five-anchor deformable facial prior, global-to-prior multi-head cross-attention, temporal selective-state-space modeling, attention pooling, and a classification head. It includes data manifests, per-clip caches, participant-disjoint splits, the seven Table-5 architectural variants, training and resume, predictions, cross-subject LOSO, metric recomputation, figures, and subject-clustered paired bootstrap.

## Repository layout

```text
DeM-SSM/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── paper.yaml              # Manuscript Table 1 and training values
│   └── smoke.yaml              # Small, CPU-friendly SOFTWARE test
├── dem_ssm/
│   ├── model.py                # Vision SSM, deformable/static priors, temporal SSM
│   ├── data.py                 # OpenCV decoder, normalized landmarks, hash caches
│   ├── manifest.py             # User-verified videos and labels -> clip manifest
│   ├── splits.py               # No-overlap subject partitions and DROZY LOSO
│   ├── runner.py               # Training / resume / predictions and audits
│   ├── metrics.py              # Recompute accuracy, macro scores, ROC/PR
│   ├── statistics.py           # Paired subject-clustered bootstrap
│   ├── audit.py                # Subject, label, interval, split checks
│   ├── experiments.py          # Seven architectural ablations × five seeds
│   ├── plots.py                # ROC/PR plots from real predictions
│   ├── toy.py                  # Explicitly synthetic video generator
│   └── cli.py                  # Main command-line interface
├── docs/
│   ├── DATA_PREPARATION.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── REPRODUCIBILITY_GAPS.md
│   └── GITHUB_CHECKLIST.md
├── examples/
│   ├── video_index_template.csv
│   ├── clip_manifest_template.csv
│   ├── split_template.json
│   └── predictions_template.csv
├── scripts/
│   ├── smoke_test.sh
│   └── smoke_test.ps1
├── tests/                      # Automated unit/integration checks
├── data/                       # Ignored by Git; create your own real data here
└── outputs/                    # Ignored by Git; new results only
```

## 1. Install

Python **3.10+** and PyTorch **2.2+** are required. For model training at full 224×224×64 scale, use a CUDA GPU with a compatible PyTorch installation. The reference SSM is an **explicit PyTorch state recurrence** and may be very slow even with a GPU. It is not the official fused CUDA Mamba kernel.

From inside the `DeM-SSM` directory:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
# Optional MediaPipe (may depend on your Python version/platform):
# python -m pip install "mediapipe>=0.10"
```

Install the NVIDIA-enabled build of PyTorch if your local PyTorch is CPU-only; refer to the **official** PyTorch install selector for the exact CUDA build. No network downloads occur automatically in this code.

## 2. Verify that the repository works (no external dataset)

```bash
python -m dem_ssm.cli toy-data --out data/toy --image-size 32 --frames 3
python -m dem_ssm.cli audit --manifest data/toy/manifest.csv --split data/toy/split.json --check-videos
python -m dem_ssm.cli train --config configs/smoke.yaml --manifest data/toy/manifest.csv --split data/toy/split.json --out outputs/toy_run
python -m dem_ssm.plots --predictions outputs/toy_run/predictions_test.csv --out outputs/toy_plots
python -m pytest -q
```

Or run `bash scripts/smoke_test.sh` (Linux/macOS) or `powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1` (Windows). These commands generate small, artificial AVI files, make an explicit subject-disjoint split, train one epoch, write `best.pt` and `last.pt`, export `predictions_test.csv`, and recompute metrics. **Toy metrics have no scientific significance.**

## 3. Prepare real UTA-RLDD or DROZY videos and ground-truth labels

1. Request/download datasets **from their original custodians** and follow their terms of use. Original videos are **not included** in the repository.
2. Create a session/video index using the schema in [`examples/video_index_template.csv`](examples/video_index_template.csv). Use real subject IDs and labels sourced from the original dataset. For DROZY, `kss_score` (1–9) is converted via 1–3→0, 4–6→1, 7–9→2. For UTA-RLDD, supply verified per-video `label` values; **do not infer vigilance states from filenames**.
3. Generate non-overlapping 64-frame clips and a per-video frame-count/sampling audit:

```bash
python -m dem_ssm.cli build-manifest \
  --index data/uta_video_index.csv \
  --out data/uta_manifest.csv --frames 64 --dataset uta_rldd

python -m dem_ssm.cli build-manifest \
  --index data/drozy_video_index.csv \
  --out data/drozy_manifest.csv --frames 64 --dataset drozy
```

4. Use the **actual historical participant split** if you have it. The following alternative produces a **new, reproducible** participant-disjoint split, which **does not replicate the paper's unknown subject identities**:

```bash
python -m dem_ssm.cli split --manifest data/uta_manifest.csv --dataset uta_rldd \
  --out data/uta_split_RECONSTRUCTED.json --seed 2026
python -m dem_ssm.cli audit --manifest data/uta_manifest.csv \
  --split data/uta_split_RECONSTRUCTED.json --check-videos --out outputs/uta_audit.json
```

**Face crop and landmarks:** `configs/paper.yaml` uses `landmark_mode: from_file`, which requires a `landmarks_path` in **every clip row**. Each file must be a shape `[T, 5, 2]` `.npy`/`.json` array of finite normalized X,Y coordinates on the *already cropped* 224×224 face, in order: left eye, right eye, nose, left mouth corner, right mouth corner. For a new implementation where annotations are unavailable, you can switch to `landmark_mode: mediapipe` with a compatible MediaPipe version (`num_workers: 0`) and optionally use `detector: haar`; **Haar is not the manuscript's RetinaFace** and should not be used for strict reproduction. `landmark_mode: static` is for software diagnostics only and is not an equivalent experimental condition. Consult [Data preparation](docs/DATA_PREPARATION.md).

**DROZY modality caution:** publicly supplied `videos_i8` are near-infrared intensity videos. The manuscript did not retain the original conversion into three network channels. This repo uses a **documented, explicit approximation**: grayscale intensity repeated into three channels. It must not be presented as the original input pipeline without provenance.

## 4. Train, continue after interruption, and evaluate

```bash
python -m dem_ssm.cli train \
  --config configs/paper.yaml \
  --manifest data/uta_manifest.csv \
  --split data/uta_split_RECONSTRUCTED.json \
  --seed 0 --device cuda --out outputs/uta_seed0

# Resume same config, same seed, same manifest and same split:
python -m dem_ssm.cli train \
  --config configs/paper.yaml \
  --manifest data/uta_manifest.csv \
  --split data/uta_split_RECONSTRUCTED.json \
  --seed 0 --device cuda --out outputs/uta_seed0 --resume

# Evaluate the best saved checkpoint with identity-checked split & manifest:
python -m dem_ssm.cli evaluate --checkpoint outputs/uta_seed0/best.pt \
  --manifest data/uta_manifest.csv --split data/uta_split_RECONSTRUCTED.json \
  --out outputs/uta_seed0/rechecked
```

By default `configs/paper.yaml` provides the manuscript values: `224×224`, `64` frames, patch `16`, latent dimension `256`, state size `16`, `L=5`, `M=4`, `K=4`, four cross-attention heads, `λ=0.05`, AdamW with LR `0.001` / weight decay `0.05`, batch size `16`, up to `150` epochs, validation-loss early stopping patience `15`, cosine schedule, and seeds `0, 42, 1234, 2026, 3407`.

**Caveat:** the paper also mentions pretrained ImageNet-1K visual weights. None are included; `--init-checkpoint /path/to/trusted_state.pt` can load a compatible state dict non-strictly and produces `initialization_audit.json` listing unmatched keys. The reconstruction's module naming is not the same as official ViM. **Do not claim matched pretraining merely because a checkpoint load did not crash.**

Training writes `run_config.json`, `split_audit.json`, `history.csv`, `best.pt`, `last.pt`, `predictions_test.csv` and `metrics_test.json`. Predictions contain actual model probabilities, true labels, subject IDs, clip IDs and seed. Cached decoded clips live under `data/cache` with versioned fingerprints and atomic writes. They are not committed.

## 5. Table-5 ablations and five independent runs

The seven supported variants are `baseline`, `a`, `b`, `c`, `d`, `e` and `dem_ssm`:

| Variant | Spatial encoder | Prior | Temporal |
|---|---|---|---|
| baseline | CNN | Static Gaussian | EMA |
| a | Vision SSM | Static Gaussian | EMA |
| b | CNN | Deformable | EMA |
| c | CNN | Static Gaussian | Temporal SSM |
| d | Vision SSM | Deformable | EMA |
| e | Vision SSM | Static Gaussian | Temporal SSM |
| dem_ssm | Vision SSM | Deformable | Temporal SSM |

The CNN and fixed Gaussian implementations are **engineering analogues**, since the manuscript does not specify a canonical CNN architecture or Gaussian bandwidth. The suite **re-trains all variants**; it does not load the manuscript's displayed scores.

```bash
python -m dem_ssm.experiments \
  --manifest data/uta_manifest.csv --split data/uta_split_RECONSTRUCTED.json \
  --config configs/paper.yaml --out outputs/table5_runs \
  --variants baseline a b c d e dem_ssm \
  --seeds 0 42 1234 2026 3407
```

Outputs: `run_metrics.csv`, `summary.csv`, and every individual run's prediction/weights and split audit.

## 6. 14-subject DROZY leave-one-subject-out (LOSO)

```bash
python -m dem_ssm.cli loso \
  --manifest data/drozy_manifest.csv --config configs/paper.yaml \
  --out outputs/drozy_loso --device cuda

# Optional: inspect/redo a single held-out subject:
python -m dem_ssm.cli loso \
  --manifest data/drozy_manifest.csv --config configs/paper.yaml \
  --out outputs/drozy_loso_one --only-subject S01 --device cuda
```

The model trains one model per held-out participant, each with 11 train and 2 validation subjects under a deterministic **reconstructed**, not original, rotation. All 14 folds are retained in output. When a held-out subject lacks a vigilance class, its class-dependent ROC-AUC/PR-AUC is recorded as `null`, **not invented**. Aggregate pooled OOF metrics are separately distinguished from means over selected class-complete folds; do not claim the historical nine-fold summary is validated until actual KSS/session-level labels have been checked.

## 7. Recalculate metrics and paired statistical comparisons

```bash
# From a real exported predictions file:
python -m dem_ssm.cli metrics \
  --predictions outputs/uta_seed0/predictions_test.csv \
  --out outputs/uta_seed0/recomputed.json

# ROC and PR curves, one-vs-rest, only when labels include both cases:
python -m dem_ssm.plots \
  --predictions outputs/uta_seed0/predictions_test.csv \
  --out outputs/uta_seed0/figures

# Matched model predictions, same clips and subjects, >=5 test subjects:
python -m dem_ssm.cli paired-bootstrap \
  --model-a outputs/model_a/predictions_test.csv \
  --model-b outputs/model_b/predictions_test.csv \
  --metric roc_auc_ovr --replicates 10000 --seed 0 \
  --out outputs/bootstrap_roc.json
```

The bootstrap refuses comparisons with only three test subjects by default. Results are newly calculated from the supplied predictions; the code **never outputs the historical S14 intervals or p-values by fiat**. It reports an empirical two-sided sign-bootstrap p estimate, not a universal significance certificate; the protocol, cluster count, and possible repeated-seed dependence must be disclosed.

**Metric convention in this code:** accuracy = exact correct clips / total; macro Precision/Recall/F1 = mean over all three labels; ROC-AUC = macro one-vs-rest multiclass ROC; PR-AUC = mean of per-class one-vs-rest *average precision*. The historical manuscript does not fully establish its underlying prediction-level PR calculation, so comparisons of PR-AUC depend on verifying the convention.

## 8. Required items before claiming paper reproduction

- Original UTA-RLDD and DROZY data in compliance with their licenses.
- Real clip/session manifests with FPS, excluded-frame logs, KSS labels, and subject-ID split files.
- Original face cropping and NIR→three-channel preprocessing procedure.
- Exact visual and temporal SSM implementations, pretrained checkpoints and model weights.
- Per-seed raw predictions and true labels for the manuscript's baseline and DeM-SSM runs.
- Original significance-test scripts and matching subject IDs.

A successful unit test or synthetic training run establishes **software executability only**. For the precise disclosure list and verification checklist see [Reproducibility gaps](docs/REPRODUCIBILITY_GAPS.md) and [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

## Scope, ethics, and reuse

Not intended as an on-road safety device or as proof of suitability for real-time deployment. Research and software testing only. Follow consent, data-provider conditions, and privacy obligations for any driver videos. The original public datasets and pretrained model licenses are separate from this project's source code license.

Before making the GitHub repository public, inspect [GitHub checklist](docs/GITHUB_CHECKLIST.md), remove any confidential material, and confirm source/license ownership with the relevant researchers/institution.

## Local software validation

The code was checked on Linux with Python 3.13 / CPU PyTorch 2.10; all **16** automated tests and the end-to-end synthetic smoke script passed. This does **not** validate the paper's research results. See [Validation report](docs/VALIDATION_REPORT.md).
