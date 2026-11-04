# GitHub release checklist

- [ ] Verify that the institution and contributors authorize public release and select the intended license; adjust the LICENSE copyright statement if needed.
- [ ] Review `git status` / `git diff`; confirm no raw UTA-RLDD/DROZY videos, private participant identifiers, medical metadata, copied baseline weights, or personal paths are committed.
- [ ] Remove temporary synthetic outputs and verify `.gitignore` excludes `data/*`, `outputs/*`, `*.pt`, and clip caches.
- [ ] Record an actual run with `python -m pytest -q` and the smoke-test script in CI before release.
- [ ] Verify README commands from a fresh environment and specify tested Python, PyTorch and CUDA versions.
- [ ] If adding original authors' code, retain its copyright/license headers; do not replace reconstruction modules without checking code provenance.
- [ ] Do not describe this independent implementation as "the official repository" without authorization.
- [ ] Add manuscript citation metadata and DOI only after confirmed; do not invent a DOI.
- [ ] If an official release is intended, upload genuine, rechecked split manifests and exact pretrained/checkpoint files only when redistribution rights permit.

Suggested commands after manually creating a repository on GitHub:

```bash
cd DeM-SSM
git init
git add .
git status --short
git commit -m "Initial auditable DeM-SSM research reconstruction"
git branch -M main
# Replace <OWNER> and <REPO> with your actual GitHub account and repository name.
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```

Do not put GitHub account credentials, personal access tokens, raw experimental video, or pretrained weights in the repository.
