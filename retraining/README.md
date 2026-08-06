# Phase 17 — Retraining Pipeline

Implemented lifecycle components:

- Dataset validation before training
- Retraining triggers by sample count or drift score
- Candidate training/evaluation orchestration
- Filesystem model registry with versions and production promotion
- Deterministic A/B assignment for champion/candidate models

Run the tests:

```bash
.venv/bin/python -m unittest discover -s retraining -p 'test_*.py'
```
