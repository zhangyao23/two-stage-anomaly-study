# Data and experiment protocol

This is the original v1 protocol. The [revision protocol](REVISION_PROTOCOL.md) adds validation-only
selection, new synthetic test seeds and the complete NAB realAdExchange confirmation subset.

## Synthetic data

No company-specific schemas, metric names, measurements or label taxonomies are used. Six generic
channels share sinusoidal variation plus autoregressive noise and per-sequence offsets. Three
synthetic shapes are injected into random channels with random signs. They do not correspond to
identified real network causes.

Sequences are generated using independent children of `numpy.random.SeedSequence` for the four
splits. The full configuration and seeds were fixed before observing test metrics. All 24 test
sequences per seed are included. No seed, event or test file was replaced to improve results.

The Autoencoder and standardizer use only completely normal train windows. Anomaly classifiers
train on all labeled abnormal train windows, not only windows admitted by the detector; this avoids
silently restricting training to easy detected cases. The AE remains frozen during head training.
Heads are selected by abnormal validation cross-entropy, the AE by normal validation MSE, and the
single-stage baseline by all-label validation cross-entropy. All epochs run; only checkpoint choice
depends on validation. The threshold quantile and network dimensions are preset, not tuned.

Input ordering is time-major flattening. AE is `96 → 64 → 8 → 64 → 96`; ReLU is used only in the
two hidden layers, leaving the latent and output layers linear. The MLP is `input → 32 → classes`.
Encoded features are not separately standardized; this is a declared comparison limitation.

## Public data source and permission

- Official repository: https://github.com/numenta/NAB
- Pinned revision: `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`
- Source directory: `data/realAWSCloudwatch/`, all 17 CSV files, selected before test execution.
- Source description: `data/README.md` at that revision, AWS CloudWatch measurements.
- Labels: `labels/combined_windows.json` at that revision.
- Label semantics verified against upstream `nab/labeler.py`: time intervals become point labels;
  the intervals originate from benchmark anomaly annotations and scoring windows, not a taxonomy
  of CPU/network faults. No multiclass labels are inferred from filenames or metric channels.
- License: upstream root `LICENSE.txt`, MIT, Copyright 2014–2024 Numenta Inc.
- Paper: Ahmad et al. (2017), *Unsupervised real-time anomaly detection for streaming data*,
  https://doi.org/10.1016/j.neucom.2017.04.070

The downloader uses official raw GitHub URLs, stores the upstream license, README and labels in
ignored `data/nab/`, and hashes every downloaded file. `results/public/data_manifest.json` contains
only paths, revision, hashes and sizes. No upstream source code is copied. Derived evaluation
statistics/predictions are committed; raw CSV values, labels JSON and models are not.

## Chronological protocol and source irregularities

Each AWS file is treated as a separate univariate sequence with its own model and scaler. Adjacent
duplicate timestamps are aggregated by arithmetic mean. This local aggregation does not consult
labels or future distinct timestamps. Decreasing timestamps or nonfinite values raise an error.

Two files (`ec2_disk_write_bytes_1ef3de.csv`, `ec2_network_in_5abac7.csv`) each contain 11 repeated
timestamp rows; their values are not necessarily identical. An initial strict-timestamp check
stopped the public run. The documented aggregation rule fixed the adapter, with a regression test;
the entire fixed subset was then run again. No data file was dropped and no metric-based tuning
occurred. Row spans in outputs refer to the aggregated sequence, with end-exclusive indices.

Cut indices are floor(0.4N), floor(0.6N), floor(0.8N) on unique timestamps. Windows of width 16,
stride 8 are created separately inside each segment. The last incomplete window of a segment
is discarded. Windows do not cross cuts; no padding borrows neighboring-segment data. Window
length is in observations, not a guarantee of a uniform wall-clock duration.

Training/calibration normal filtering assumes labels are available within those segments. Validation
labels identify normal checkpointing windows. Test labels never choose a scaler, checkpoint,
threshold, architecture, seed or file. All test windows, including normal-only test segments, are
retained. This is an offline clean-normal training study, not the online unsupervised NAB protocol.

## Metrics and uncertainty

Detection positive class is any anomaly. Precision = TP/(TP+FP), recall = TP/(TP+FN),
FPR = FP/(FP+TN), and F1 is the harmonic mean of precision and recall. PR-AUC means sklearn's
average precision, with a stepwise recall-weighted sum. No positives → AP null; no negatives →
FPR null. Other binary zero-denominator metrics are zero by explicit convention.

For diagnosis, fixed-label macro-F1 averages all specified class F1 values, including zero for
missing classes; unsupported per-class recall is null, not a fabricated estimate. Confusion
matrices include fixed rows/columns. Normal and anomaly errors all contribute to full-pipeline
metrics. The independent three-class diagnostic task and four-class pipeline cannot be compared
by subtracting macro-F1 because their class sets differ.

Public pooled metrics count all 1,665 test windows once, weighting series by available windows.
Pooled AP ranks each score divided by its own series' calibration threshold (floor 1e-12);
this ranking normalization is a design choice. Binary counts use exact raw score/threshold
decisions. Per-series AP/counts are retained so pooled performance is not the only view.

Three synthetic seeds vary both data realization and model initialization; SD is sample SD with
ddof=1, not a standard error, CI or significance test. Windows overlap within a split and are
correlated. Public results use a single initialization seed, so no public seed uncertainty is
estimated. Interval-derived positives do not imply precise fault duration. None of these metrics
are the NAB leaderboard score, and no point adjustment or event expansion is applied.
