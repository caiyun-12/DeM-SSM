# Reproducibility gaps, claims, and evidence status

The author-supplied manuscript and supplementary material contain historical statistics, but no clip-level raw predictions or complete preprocessing/split manifests. This repository **does not and must not invent** them.

| Issue | Status in supplied documents | What this repository does | Evidence still needed |
|---|---|---|---|
| UTA-RLDD subject split | train=38, val=10, test=12; exact IDs not provided | constructs deterministic NEW split if requested | subject-ID manifest of original split |
| DROZY fixed split | 9/2/3 subjects, 587 held-out clips | can audit user-supplied data/split | actual three test identities and all session labels |
| DROZY LOSO | 14 test folds, nine potentially class-complete; 5 missing some sessions | evaluates all provided subjects and marks uncomputable subject AUC as null | original per-session KSS, FPS, clip exclusions and selection rules |
| Accuracy | withdrawn as NR because historical decimals do not reconcile with exact clip counts | computes count-based accuracy from new real predictions | original predictions per seed and labels |
| Other historical metrics | seed-level numbers arithmetically consistent but not independently regenerated | never uses printed performance tables as inputs | matched per-sample predictions and evaluation implementation |
| Historical S14 p-values | subject-clustered intervals retained but pending regeneration | computes new paired subject-clustered intervals only from matched predictions | original subject-linked predictions and bootstrap scripts |
| S3 gate vs EAR statistics | historical event summaries pending raw-event audit | exports model internal gate, but no p-values without external frame-level EAR and annotations | EAR extraction code, event annotations, statistical code |
| DROZY channels | near-infrared intensity `videos_i8`, exact three-channel transformation undocumented | repeats grayscale into three channels, explicitly marked an approximation | exact historical conversion/configuration |
| Face detector/landmarks | RetinaFace + MediaPipe described; versions/checkpoints not provided | optional MediaPipe, external landmark files, Haar marked approximate | model/detector version, thresholds, crop log |
| Visual backbone pretraining | ImageNet-1K initialization claimed | new SSM from scratch unless an independently compatible checkpoint is provided | original backbone weights and mapping procedure |
| VideoMamba baseline | VideoMamba-Tiny Kinetics-400 referenced in manuscript | no unlicensed baseline source or pretrained weights bundled | official code/weights and matched data preprocessing |
| Deployment efficiency | paper suppresses FLOPs/FPS pending unit audit | no invented FLOPs or FPS | traced profiling with explicit per-frame/per-clip units |
| Published exact performance | no independently verified complete audit | report metrics derived from each new output run only | complete original experimental artifacts |

## Ethics for scientific reporting

1. Do **not** claim that a successful `smoke.yaml` test reproduces the manuscript's reported AUC, F1, p-values, or original ROC figures.
2. Clearly label reconstructed splits and input-channel expansions; never describe them as recovered original protocols.
3. Verify the final training/test class counts and original KSS bin boundaries before treating LOSO results as complete.
4. Do not edit or filter test clips after looking at model performance; log all exclusions before the analysis.
5. Do not fabricate trial-to-trial seeds, test-set predictions, statistical intervals or artifact hashes.
6. Keep any copied external code under its original license, with explicit source attribution.

## Acceptance checklist for a genuine reproduction

- [ ] Real data source and version identified with a stable link and license.
- [ ] All raw video-to-session identities and KSS labels verified.
- [ ] Train, validation, test participant IDs recovered and logged.
- [ ] Preprocessing transformation, face detector weights, crop and anchors versioned.
- [ ] Same code and verified pretrained weights used across paper and reproducibility run.
- [ ] `predictions.csv` from every seed, model, sample and split exported.
- [ ] All metrics recomputed with exact class-average and confidence interval definitions.
- [ ] Bootstrap matches participants and full predicted probability vectors.
- [ ] Test-set participant identities unique between training/validation/testing.
- [ ] Every result has an artifact hash and provenance that another researcher can independently inspect.
