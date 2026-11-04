# Experiment protocol implemented here

This document records **implementation choices**, including places where this reconstruction necessarily differs from unavailable original code. Paper numbers must not be attached to regenerated output without independent verification.

## Architecture

| Component | Reconstruction | Original manuscript relation |
|---|---|---|
| Input | `[batch, 64, 3, 224, 224]` | T=64, RGB-like 3-channel tensor, 224² |
| Patch embedding | Conv2d kernel=stride=16, 196 tokens per frame | patch=16, S=196 |
| Spatial SSM | Separate forward and reverse row-major selective diagonal-state recurrences | bidirectional raster scanning; not official fused Mamba/ViM |
| Deformable prior | L=5, M=4, K=4, learned 2D offsets, softmax weights, bilinear grid sampling | follows described deformable sampling; offset clip bound is an engineering assumption |
| Gaussian prior control | Fixed spatial Gaussian ROI pooling around the five anchors | manuscript gives static Gaussian control but not Gaussian width |
| Global/prior fusion | 4-head PyTorch cross-attention with queries from global tokens, keys/values from prior tokens, residual LayerNorm, dim=256 | paper Eqs. (13)–(16) |
| Temporal model | Input-dependent diagonal-state selective recurrence with ZOH discretization, state=16, temporal position embedding | interpretable approximation of Temporal SSM, not original kernel |
| Temporal pooling | trainable softmax weights over T, then MLP classifier | manuscript Eqs. (21)–(22) |
| Objective | cross entropy + λ × mean L2 offset norm in pixels | λ=0.05; exact penalty normalization was not fully specified |

The code's input-dependent SSM uses explicit Python loops. Its behavior is useful for audit and small-scale prototyping but is not bit-for-bit equivalent to optimized Mamba or the original paper code, and 64-frame training on GPU may be slow. MediaPipe detections and ROI behavior can differ from the historical implementation.

## Training

- AdamW, `lr=1e-3`, `weight_decay=0.05`, cosine annealing.
- Up to `150` epochs; batch `16`; early stopping after `15` unimproved validation-loss epochs.
- Seeds `0`, `42`, `1234`, `2026`, `3407`. The subject partition is *fixed* across seeds (provided in JSON).
- `best.pt` records the lowest validation-loss model; `last.pt` stores model, optimizer, scheduler and GradScaler states to support interruption recovery. Configuration and manifest SHA256 are checked on resume.
- Cache is per clip and validated; augmentation is not cached.
- If you do not supply matching ImageNet-1K pretrained weights, this is **random initialization**, unlike the paper's claimed backbone initialization. The optional loader reports unmatched state-dict keys and cannot by itself establish a valid backbone import.

## Evaluation and significance

- `predictions_test.csv` includes `clip_id`, `subject_id`, `true_label`, `pred_label`, `prob_0`, `prob_1`, `prob_2`, `seed`.
- Every metric is recomputed from the saved raw probabilities. Accuracy always includes exact correct-class count.
- For ROC-AUC and PR-AUC, when a test *subject* is missing one of the three classes the subject-specific metric is null. It is not silently filled with a fabricated value.
- `--paired-bootstrap` aligns predictions by unique `clip_id`, checks all true labels and subject identities, samples **subjects** with replacement, and exports CI/p diagnostics. For fewer than five subjects it refuses to generate confidence intervals. The 3-subject DROZY fixed split does not support meaningful conventional bootstrap significance statements.
- Fold-specific F1 across subjects with incomplete class coverage is included as a descriptive all-3-class statistic only. Do not directly compare it to fully class-complete fold F1 without disclosing missing classes.
- Historical tables lack original raw probabilities and original class-wise definitions. A match in high-level model names or hyperparameters is **not** a sufficient basis to report recovered paper results.

## Seven modular ablations

`baseline`: CNN + static Gaussian + EMA; `a`: ViM + static Gaussian + EMA; `b`: CNN + deformable + EMA; `c`: CNN + static Gaussian + Temporal SSM; `d`: ViM + deformable + EMA; `e`: ViM + static Gaussian + Temporal SSM; `dem_ssm`: ViM + deformable + Temporal SSM.

The actual configuration count is **seven**, despite an erroneous claim of eight in an earlier reviewer response draft. Architecture-oriented comparisons do not isolate physiological-signal modality effects because every input in this implementation is visual.

## Not bundled

Historical human participant files; original recorded training artifacts; original supplementary S3–S8 robustness/event data; VideoMamba-Tiny official source or weights; original EffRes-DrowsyNet or adapted physiological model source code; exact ImageNet ViM weights; true sequence/KSS preprocessing manifests. Their absence is documented, not concealed.
