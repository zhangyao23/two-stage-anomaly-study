# Revision protocol, fixed before new confirmation evaluation

The original results are retained in `results/synthetic` and `results/public`. They motivated
this revision and are not untouched test evidence for revised model selection.

## Synthetic selection and confirmation

Keep the original generator, anomaly strengths, 16-step target labeling, stride, sequence counts,
and normal-only AE. Do not relabel or drop difficult windows. Development uses only train,
calibration and validation splits for seeds 17/29/43. Compare unchanged v1 with balanced heads,
train-fitted feature scaling, approximately matched 6,400-parameter heads, and temporal features.
All revised heads receive 100 epochs and the same optimizer, batches and shuffling. Record exact
parameter and update counts. Added epochs/budget versus v1 are an explicit intervention, not a
pure feature ablation. Include a revised single-stage baseline with the same features and budget.

The temporal variant uses a 64-step causal context (including the target), target-window and
historical summaries, and a normal-data-fitted rank-one cross-channel residual. It does not use
the generator's latent clean signal, event start, affected channel, severity or labels as features.
Padding at a sequence start repeats that sequence's first observed value; it never borrows from
another split/sequence. AE reconstruction training remains on normal 16-step target windows only.

Pick the revised representation using mean validation end-to-end macro-F1 across development
seeds, then freeze it. New test realizations use the predeclared seeds 101/211/307 (test child RNG
only), paired respectively with training seeds 17/29/43. Evaluate v1 and all declared controls
on identical new test windows. Do not select a different winner after reading confirmation results.
Record paired per-seed differences; three seeds do not establish statistical significance.

## Public data

Use all 17 original AWS series' training/calibration/validation portions to select one global
detector/quantile. Candidate scores: AE MSE, max absolute training z-score, and trailing robust
innovation (lookback 32/64/128); quantile 0.95/0.975/0.99. Threshold values always come from normal
calibration windows. Select highest pooled validation F1 among candidates with validation FPR
<=0.10; if none qualify, select lowest FPR, then highest F1. No test-based per-file choice.

Robust innovation uses only earlier observations' median and MAD, with a scale floor equal to
0.1 times normal training-window standard deviation (minimum 1e-6). It is a statistical baseline,
not a new learned AE method. Each split starts from a normal training median. No test labels
enter adaptation. A rolling detector may adapt to persistent faults, so missed sustained events
must remain visible. No future FPR guarantee is claimed.

Run the selected detector and fixed original baselines on original AWS test segments as
**retrospective** evidence. Also download and evaluate the entire `realAdExchange` directory
from the same pinned NAB revision as a new **out-of-domain confirmation** subset. Its six
advertising metric series are not network faults; do not claim network diagnosis from this set.
Use the same chronological 40/20/20/20 division and duplicate policy. Never inspect confirmation
metrics before writing the selection record. Preserve every file, including no-positive segments.

Configuration: `configs/revision.json`. Selection and evaluation commands are separate. Original
test results are not overwritten. Any unresolved weakness must be reported rather than hidden.
