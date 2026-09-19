# Validation record

Performed locally on Windows 11 / Intel Core i7-13700H / Python 3.12.10 / PyTorch 2.9.0+cpu.
The experiment environment is also recorded in each result JSON; direct versions are pinned in
`requirements-repro.txt`. No GPU, paid API or cloud computation was used.

| Check | Actual outcome |
|---|---|
| Quick example, 5 epochs / seed 17 | Completed; 234 test windows |
| Full synthetic, 35 epochs / seeds 17, 29, 43 | Completed; 936 test windows each, 2,808 total |
| Public NAB AWS, 35 epochs / seed 17 | All 17 files completed; 1,665 test windows |
| `python -m pytest -q` | 11 passed |
| `python -m ruff check src tests scripts` | Passed |
| `python -m ruff format --check src tests scripts` | Passed |
| `python scripts/audit_results.py` | Recomputed metrics, confusion matrices, gate decisions, split spans, synthetic summary mean/SD and public pooled AP/counts agree |
| Quick rerun after limited orchestration simplification | Prediction CSV, summary JSON and failure JSON byte-identical to original quick run |
| `python -m pip wheel --no-deps --no-build-isolation . --wheel-dir dist` | Wheel built successfully; generated build outputs ignored |
| Visual inspection | Comparison, confusion, error-propagation, failure examples and public per-series charts inspected |

Tests address sequence/window overlap across splits, training-only standardization, normal-only
calibration provenance, classifier input dimensions, gated implementation versus metric composition,
counting misses and false alarms, no positive predictions, absent classes, empty diagnosis arrays,
constant dimensions, public label interval mapping and duplicate timestamp aggregation. A label
perturbation test verifies that test labels do not affect AE parameters or thresholds.

Recorded artifact-generation commands:

```bash
python -m anomaly_study.cli synthetic --config configs/quick.json --output results/quick
python -m anomaly_study.cli synthetic --config configs/full.json --output results/synthetic
python -m anomaly_study.cli download --data data/nab
python -m anomaly_study.cli public --config configs/full.json --data data/nab --output results/public
python -m anomaly_study.report
python -m anomaly_study.cli synthetic --config configs/quick.json --output results/local-verify
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python scripts/audit_results.py
```

Implementation corrections were limited to timestamp handling, formatting/imports, explicit loop
captures, report generation and separation of single-stage training from the anomaly-head loop.
Existing scientific configurations were retained. Training/inference algorithms were not modified
to improve test results. The limited simplification did not alter model kernels, memory copies,
loop counts or data access; before/after quick predictions were identical. No before/after
performance claim is made for that orchestration-only change.

Known coverage limits: tests do not establish robustness to arbitrary distribution shifts, real
network root-cause correctness, a future 5% false-positive guarantee or production thread safety.
No cross-platform run, capacity-matched probe or multi-seed public experiment is claimed.

The repository publishes implementation, experiments and technical documentation under MIT.
Raw datasets, model weights and local working notes are excluded from the current tracked tree.
