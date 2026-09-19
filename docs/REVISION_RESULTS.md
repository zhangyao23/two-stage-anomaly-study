# Revision: confirmed gains and unresolved limits

Protocol and validation selection were committed in `17bd171` before confirmation evaluation.
Development seeds: 17/29/43; fresh test seeds: 101/211/307. Same generator, difficulty and all 936 windows per seed retained.
Numbers are mean ± sample SD (3 seeds), not significance tests. Original files remain unchanged.

## Same-test complete-pipeline comparison

| System | Macro-F1 | normal recall | spike recall | shift recall | drift recall |
| --- | --- | --- | --- | --- | --- |
| v1 AE + concat | 0.446 ± 0.025 | 0.963 ± 0.035 | 0.025 ± 0.028 | 0.326 ± 0.076 | 0.474 ± 0.036 |
| balanced matched concat | 0.525 ± 0.013 | 0.963 ± 0.035 | 0.369 ± 0.031 | 0.543 ± 0.079 | 0.256 ± 0.089 |
| AE + temporal (3 classes) | 0.844 ± 0.011 | 0.963 ± 0.035 | 0.866 ± 0.022 | 0.804 ± 0.065 | 0.623 ± 0.058 |
| AE + temporal rejection | 0.855 ± 0.009 | 0.986 ± 0.019 | 0.860 ± 0.031 | 0.816 ± 0.062 | 0.611 ± 0.031 |
| temporal single-stage | 0.863 ± 0.026 | 0.893 ± 0.018 | 0.950 ± 0.038 | 0.852 ± 0.060 | 0.767 ± 0.024 |

The selected two-stage system improves markedly over v1, but the feature-matched single-stage model has a slightly higher mean.
The evidence supports the feature/training repair; it does not establish that an AE gate is necessary or superior.

## Matched-capacity representation controls

| Input | Parameters | Independent anomaly macro-F1 | End-to-end macro-F1 |
| --- | --- | --- | --- |
| raw | 6403 | 0.428 ± 0.116 | 0.507 ± 0.059 |
| encoded | 6399 | 0.344 ± 0.017 | 0.455 ± 0.012 |
| concat | 6375 | 0.454 ± 0.028 | 0.525 ± 0.013 |
| temporal | 6394 | 0.919 ± 0.024 | 0.844 ± 0.011 |
| single_raw | 6367 | 0.442 ± 0.027 | 0.527 ± 0.021 |
| single_temporal | 6400 | 0.895 ± 0.021 | 0.855 ± 0.009 |

All revised heads are within 1% of 6,400 trainable parameters, use train-fitted scaling and inverse-frequency cross-entropy, and run 100 epochs.
Class weights are derived only from training counts and also define checkpoint validation loss. Head widths differ to match parameter budgets; this is not exact functional-capacity equivalence.
Anomaly-only heads see identical training samples/update counts. Four-class heads see normal windows too and therefore more steps; the single-stage temporal comparator uses the exact same four-class head.
Compared with v1, budget, scaling and class balance changed jointly. No one-factor causal attribution is claimed.
The temporal representation adds 64 observations of causal context and a normal-trained rank-one cross-channel residual (PCA), not an anomaly-trained AE encoding.
These features exploit the generator's shared normal variation and sparse channel disturbances; general multivariate real-fault benefit is untested.

## Paired changes on every confirmation seed

| Train seed | Fresh test seed | v1 concat | Selected revised | Difference | Matched single-stage |
| --- | --- | --- | --- | --- | --- |
| 17 | 101 | 0.4265 | 0.8654 | +0.4389 | 0.8902 |
| 29 | 211 | 0.4747 | 0.8542 | +0.3794 | 0.8394 |
| 43 | 307 | 0.4375 | 0.8468 | +0.4094 | 0.8593 |

## Public detection: lower false alarms are not a universal improvement

AWS validation selected rolling median/MAD innovation with lookback 128 and normal-calibration quantile 0.975.
Selection maximized validation F1 subject to pooled validation FPR <=0.10. It used neither old test labels nor new AdExchange confirmation metrics.

### AWS retrospective

| Detector | Precision | Recall | F1 | FPR | AP |
| --- | --- | --- | --- | --- | --- |
| autoencoder@0.95 | 0.1602 | 0.4689 | 0.2388 | 0.2923 | 0.1525 |
| max_abs_z@0.95 | 0.1640 | 0.4633 | 0.2422 | 0.2809 | 0.1721 |
| rolling_128@0.975 | 0.1479 | 0.1186 | 0.1317 | 0.0813 | 0.1533 |

17 series; 177 positive and 1488 negative test windows.

### AdExchange confirmation

| Detector | Precision | Recall | F1 | FPR | AP |
| --- | --- | --- | --- | --- | --- |
| autoencoder@0.95 | 0.0769 | 0.7143 | 0.1389 | 0.2667 | 0.3400 |
| max_abs_z@0.95 | 0.0667 | 0.7143 | 0.1220 | 0.3111 | 0.3505 |
| rolling_128@0.975 | 0.1136 | 0.7143 | 0.1961 | 0.1733 | 0.4332 |

6 series; 7 positive and 225 negative test windows.

On AWS the false-alarm constraint reduces FPR but sacrifices recall/F1: this is an explicit cost-sensitive operating point, not a Pareto improvement.
The AdExchange confirmation set improves F1/AP and FPR at the same measured recall, but contains only seven positive windows: evidence is weak and not a network fault-type benchmark.
Its FPR still exceeds 0.10: a validation FPR constraint is not a future-distribution guarantee. No post-confirmation retuning was performed.

## Actual full-pipeline CPU timing

Intel Core i7-13700H; one PyTorch CPU thread; 20 warmups/100 repeats. Includes train-fitted scaling, temporal features, MLP, and AE gate when enabled; excludes context assembly and I/O.
The same first 1/128 test windows are used per seed; gate acceptance rate affects timing. These medians are not worst-case latency or a production SLA.

| System | Batch | Batch median ms | Batch p95 ms |
| --- | --- | --- | --- |
| Temporal single-stage | 1 | 0.245 ± 0.008 | 0.775 ± 0.359 |
| Temporal single-stage | 128 | 0.775 ± 0.027 | 1.371 ± 0.211 |
| AE + temporal rejection | 1 | 0.284 ± 0.199 | 0.655 ± 0.585 |
| AE + temporal rejection | 128 | 0.769 ± 0.139 | 1.260 ± 0.309 |

Full model parameters: AE 13,544 + selected head 6,400 = 19,944. The matched single-stage has 6,400 neural parameters; both also store small PCA/scaler statistics.
Public rolling timing includes rolling-statistic computation over the segment, with 20 warmups/20 repeats. Old AE detector timings exclude preprocessing and must not be treated as an equal-scope speed comparison.

## Remaining actual failures

| Seed | Window | Category | Truth | Final | Independent head | AE score / threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | test-0:56:72 | missed_spike | spike | normal | spike | 0.152 / 0.186 |
| 101 | test-0:136:152 | missed_drift | drift | normal | normal | 0.125 / 0.186 |
| 101 | test-0:168:184 | wrong_type | drift | shift | shift | 0.426 / 0.186 |
| 211 | test-6:56:72 | missed_spike | spike | normal | normal | 0.128 / 0.148 |
| 211 | test-0:224:240 | missed_drift | drift | normal | normal | 0.181 / 0.148 |
| 211 | test-4:40:56 | wrong_type | shift | drift | drift | 0.165 / 0.148 |
| 211 | test-0:0:16 | false_alarm | normal | drift | drift | 0.161 / 0.148 |
| 307 | test-4:128:144 | missed_spike | spike | normal | shift | 0.110 / 0.173 |
| 307 | test-0:136:152 | missed_drift | drift | normal | drift | 0.111 / 0.173 |
| 307 | test-6:48:64 | wrong_type | shift | drift | drift | 0.457 / 0.173 |
| 307 | test-11:24:40 | false_alarm | normal | drift | drift | 0.204 / 0.173 |

Examples are first matches per seed/category, not handpicked successes. Their generated contexts are in `results/revision/failure_cases.json`.
Drift end-to-end recall remains limited by the unchanged AE detector; correct independent predictions still disappear at the gate. Rejecting normal false alarms can also reject weak true anomalies.

## Confusion matrices

Rows=true, columns=predicted; order [normal, spike, shift, drift]. Independent diagnostic matrices retain the normal rejection column; macro-F1 there averages only anomaly classes.

### Test seed 101

raw independent: `[[0, 0, 0, 0], [0, 29, 15, 8], [0, 19, 92, 28], [0, 30, 86, 74]]`

raw cascade: `[[545, 2, 5, 3], [7, 23, 14, 8], [5, 18, 91, 25], [68, 14, 76, 32]]`

encoded independent: `[[0, 0, 0, 0], [0, 27, 4, 21], [0, 43, 51, 45], [0, 81, 46, 63]]`

encoded cascade: `[[545, 6, 0, 4], [7, 24, 4, 17], [5, 42, 51, 41], [68, 49, 40, 33]]`

concat independent: `[[0, 0, 0, 0], [0, 25, 12, 15], [0, 19, 87, 33], [0, 39, 78, 73]]`

concat cascade: `[[545, 3, 4, 3], [7, 20, 11, 14], [5, 19, 87, 28], [68, 22, 70, 30]]`

temporal independent: `[[0, 0, 0, 0], [0, 51, 0, 1], [0, 5, 127, 7], [0, 3, 5, 182]]`

temporal cascade: `[[545, 0, 0, 10], [7, 44, 0, 1], [5, 5, 122, 7], [68, 1, 4, 117]]`

single_raw independent: `[[0, 0, 0, 0], [9, 24, 11, 8], [20, 9, 90, 20], [50, 12, 80, 48]]`

single_raw cascade: `[[548, 0, 5, 2], [12, 22, 11, 7], [21, 8, 90, 20], [76, 8, 75, 31]]`

single_temporal independent: `[[0, 0, 0, 0], [0, 50, 1, 1], [2, 3, 128, 6], [33, 1, 5, 151]]`

single_temporal cascade: `[[555, 0, 0, 0], [7, 43, 1, 1], [7, 3, 123, 6], [69, 1, 4, 116]]`

Temporal single-stage: `[[506, 0, 3, 46], [0, 50, 1, 1], [2, 3, 128, 6], [33, 1, 5, 151]]`

### Test seed 211

raw independent: `[[0, 0, 0, 0], [0, 24, 10, 20], [0, 17, 77, 44], [0, 32, 60, 98]]`

raw cascade: `[[511, 11, 13, 19], [5, 23, 10, 16], [12, 15, 74, 37], [59, 12, 59, 60]]`

encoded independent: `[[0, 0, 0, 0], [0, 28, 7, 19], [0, 38, 44, 56], [0, 86, 43, 61]]`

encoded cascade: `[[511, 29, 9, 5], [5, 25, 7, 17], [12, 35, 42, 49], [59, 54, 31, 46]]`

concat independent: `[[0, 0, 0, 0], [0, 23, 7, 24], [0, 18, 77, 43], [0, 39, 57, 94]]`

concat cascade: `[[511, 14, 11, 18], [5, 21, 7, 21], [12, 16, 74, 36], [59, 13, 55, 63]]`

temporal independent: `[[0, 0, 0, 0], [0, 49, 2, 3], [0, 6, 109, 23], [0, 0, 6, 184]]`

temporal cascade: `[[511, 0, 3, 40], [5, 48, 0, 1], [12, 3, 104, 19], [59, 0, 1, 130]]`

single_raw independent: `[[0, 0, 0, 0], [15, 14, 5, 20], [21, 7, 75, 35], [60, 15, 52, 63]]`

single_raw cascade: `[[529, 5, 11, 9], [18, 13, 5, 18], [24, 7, 74, 33], [78, 7, 51, 54]]`

single_temporal independent: `[[0, 0, 0, 0], [4, 49, 1, 0], [6, 5, 114, 13], [44, 0, 3, 143]]`

single_temporal cascade: `[[534, 0, 2, 18], [6, 48, 0, 0], [13, 3, 110, 12], [68, 0, 0, 122]]`

Temporal single-stage: `[[485, 1, 6, 62], [4, 49, 1, 0], [6, 5, 114, 13], [44, 0, 3, 143]]`

### Test seed 307

raw independent: `[[0, 0, 0, 0], [0, 28, 15, 8], [0, 78, 45, 18], [0, 120, 30, 36]]`

raw cascade: `[[549, 7, 0, 2], [7, 22, 15, 7], [17, 63, 43, 18], [74, 60, 23, 29]]`

encoded independent: `[[0, 0, 0, 0], [0, 1, 33, 17], [0, 8, 85, 48], [0, 14, 94, 78]]`

encoded cascade: `[[549, 1, 5, 3], [7, 1, 31, 12], [17, 5, 72, 47], [74, 6, 53, 53]]`

concat independent: `[[0, 0, 0, 0], [0, 17, 12, 22], [0, 20, 69, 52], [0, 34, 64, 88]]`

concat cascade: `[[549, 2, 1, 6], [7, 17, 12, 15], [17, 16, 66, 42], [74, 15, 45, 52]]`

temporal independent: `[[0, 0, 0, 0], [0, 50, 1, 0], [0, 2, 117, 22], [0, 0, 6, 180]]`

temporal cascade: `[[549, 0, 0, 9], [7, 44, 0, 0], [17, 1, 110, 13], [74, 0, 6, 106]]`

single_raw independent: `[[0, 0, 0, 0], [9, 16, 12, 14], [17, 11, 71, 42], [31, 20, 63, 72]]`

single_raw cascade: `[[549, 1, 2, 6], [10, 16, 12, 13], [25, 10, 70, 36], [78, 11, 56, 41]]`

single_temporal independent: `[[0, 0, 0, 0], [0, 50, 1, 0], [8, 1, 114, 18], [45, 0, 1, 140]]`

single_temporal cascade: `[[555, 0, 0, 3], [7, 44, 0, 0], [17, 1, 108, 15], [77, 0, 1, 108]]`

Temporal single-stage: `[[497, 2, 2, 57], [0, 50, 1, 0], [8, 1, 114, 18], [45, 0, 1, 140]]`
