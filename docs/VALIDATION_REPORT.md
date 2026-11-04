# Local validation report (software checks only)

Date: 2026-11-03. Host environment: Linux, Python 3.13, PyTorch 2.10.0+cpu, OpenCV 4.13.0, pytest 9.0.2.

The generated project passed:

- [x] PyTorch spatial/temporal SSM forward pass and gradient backpropagation.
- [x] All seven modular ablation variants can execute a forward pass.
- [x] KSS 1–3 / 4–6 / 7–9 mapping and invalid-KSS rejection.
- [x] Subject-split leakage rejection.
- [x] Class-incomplete ROC/PR metric handling (`null`).
- [x] Refusal to produce subject-clustered significance estimates for three test subjects.
- [x] Probability-simplex validation.
- [x] Tiny artificial AVI generation and complete three-class clip manifest.
- [x] Dataset cache repair after deliberate corruption.
- [x] One-epoch end-to-end training, checkpoint save, test predictions and re-evaluation.
- [x] Split integrity checks on checkpoint re-evaluation.
- [x] Matched paired-bootstrap software validation.
- [x] ROC and PR PNG creation from synthetic model predictions.
- [x] `python -m pytest -q`: **16 passed**.

**These checks establish code functionality only.** They do not validate any historical paper AUC/F1 values, raw videos, subject lists, official pretrained weights, or numerical claims about VideoMamba or other baselines.

To re-run locally:

```bash
python -m pip install -e ".[test]"
bash scripts/smoke_test.sh
```
