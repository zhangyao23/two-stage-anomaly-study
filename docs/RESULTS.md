# Measured experiment results

Generated from stored JSON/CSV with `python -m anomaly_study.report`.
No point adjustment. All values are fractions; ± is sample SD over seeds, not a confidence interval.

## Synthetic detection

Positive class: any synthetic anomaly. Evaluation unit: a 16-point window, stride 8.

| Detector | Precision | Recall | F1 | FPR | PR-AUC (AP) |
| --- | --- | --- | --- | --- | --- |
| autoencoder | 0.924 ± 0.034 | 0.808 ± 0.035 | 0.861 ± 0.005 | 0.047 ± 0.024 | 0.937 ± 0.004 |
| max_abs_z | 0.884 ± 0.011 | 0.556 ± 0.049 | 0.682 ± 0.040 | 0.050 ± 0.003 | 0.821 ± 0.027 |

Single-stage binary views: summing abnormal probabilities at 0.5 differs from four-class argmax.

| Single-stage decision | Precision | Recall | F1 | FPR | AP |
| --- | --- | --- | --- | --- | --- |
| P(anomaly) > 0.5 | 0.757 ± 0.036 | 0.649 ± 0.023 | 0.698 ± 0.004 | 0.145 ± 0.033 | 0.823 ± 0.003 |
| argmax is not normal | 0.899 ± 0.025 | 0.536 ± 0.026 | 0.671 ± 0.015 | 0.042 ± 0.013 | not defined from hard predictions |

## Independent diagnosis on ALL true anomalies

No detector filtering here. Fixed labels: spike, shift, drift. Different denominator and class set from the four-class cascade.

| Input | Macro-F1 | Spike recall | Shift recall | Drift recall |
| --- | --- | --- | --- | --- |
| raw | 0.359 ± 0.055 | 0.011 ± 0.020 | 0.396 ± 0.233 | 0.788 ± 0.152 |
| encoded | 0.307 ± 0.012 | 0.000 ± 0.000 | 0.210 ± 0.011 | 0.851 ± 0.054 |
| concat | 0.368 ± 0.030 | 0.012 ± 0.011 | 0.395 ± 0.076 | 0.778 ± 0.016 |

## Complete pipeline on ALL test windows

Includes false alarms and missed anomalies. Rows in confusion matrices are truth; columns are predictions.

| Model | Macro-F1 | Normal recall | Spike recall | Shift recall | Drift recall |
| --- | --- | --- | --- | --- | --- |
| autoencoder + raw | 0.450 ± 0.038 | 0.953 ± 0.024 | 0.011 ± 0.020 | 0.384 ± 0.222 | 0.509 ± 0.132 |
| autoencoder + encoded | 0.415 ± 0.005 | 0.953 ± 0.024 | 0.000 ± 0.000 | 0.203 ± 0.004 | 0.569 ± 0.053 |
| autoencoder + concat | 0.457 ± 0.020 | 0.953 ± 0.024 | 0.012 ± 0.011 | 0.383 ± 0.080 | 0.498 ± 0.027 |
| max_abs_z + raw | 0.369 ± 0.022 | 0.950 ± 0.003 | 0.011 ± 0.020 | 0.258 ± 0.110 | 0.265 ± 0.066 |
| max_abs_z + encoded | 0.345 ± 0.026 | 0.950 ± 0.003 | 0.000 ± 0.000 | 0.169 ± 0.044 | 0.290 ± 0.043 |
| max_abs_z + concat | 0.375 ± 0.029 | 0.950 ± 0.003 | 0.012 ± 0.011 | 0.282 ± 0.106 | 0.253 ± 0.006 |
| single-stage | 0.435 ± 0.009 | 0.958 ± 0.013 | 0.019 ± 0.019 | 0.457 ± 0.051 | 0.288 ± 0.019 |

## Detection ceilings and error propagation

End-to-end class recall = detection recall for that class × classification accuracy conditional on detection.
The following AE detection recalls are upper bounds, even for a perfect downstream classifier.

| Spike | Shift | Drift |
| --- | --- | --- |
| 0.880 ± 0.054 | 0.945 ± 0.034 | 0.685 ± 0.030 |

## Capacity and training budget

| Model | Input dim | Head params | Total with AE | Train windows (seed 17) | Adam steps |
| --- | --- | --- | --- | --- | --- |
| raw | 96 | 3203 | 16747 | 637 | 175 |
| encoded | 8 | 387 | 13931 | 637 | 175 |
| concat | 104 | 3459 | 17003 | 637 | 175 |
| single-stage | 96 | 3236 | 3236 | 1560 | 455 |

AE parameters: 13544. Statistical detector: no neural parameters; train means/scales and one calibration threshold.
All three anomaly heads share samples, order seed, hidden width 32, Adam lr 0.001, batch 128 and 35 epochs; validation loss selects the checkpoint.
Encoded inputs are frozen AE outputs without separate rescaling. Capacity, feature scale, class imbalance and optimization confound a pure representation claim.
The single-stage baseline sees all labeled training windows, hence more optimizer steps at the same epoch budget. This is not a matched-supervision or matched-compute comparison.

## Measured CPU inference

Hardware: Intel Core i7-13700H, Windows 11, CPU only, one PyTorch thread. Timings exclude data loading, windowing and standardization.
20 warmups + 100 repetitions with perf_counter_ns and torch.inference_mode. Batch = first 128 held-out synthetic windows. Gated cascades reuse the encoding and classify only flagged windows.
Numbers below summarize per-run batch medians across the three seeds; per-run p95 and per-window amortized times are in JSON. This is not online request latency or a production SLA.

| Operation | Batch median ms (mean ± seed SD) |
| --- | --- |
| autoencoder_detector | 0.098 ± 0.005 |
| max_abs_z_detector | 0.012 ± 0.001 |
| raw_head_only | 0.025 ± 0.004 |
| raw_ae_cascade | 0.197 ± 0.044 |
| encoded_head_only | 0.018 ± 0.002 |
| encoded_ae_cascade | 0.194 ± 0.048 |
| concat_head_only | 0.027 ± 0.006 |
| concat_ae_cascade | 0.192 ± 0.050 |
| single_stage | 0.035 ± 0.001 |

## Public NAB AWS detection

All 17 files; separate model/scaler/threshold per file; final 20% held out. One fixed initialization seed (17), not three public seeds.
PR-AUC is average precision pooled over score / per-series threshold. Per-series AP is also stored. Pooling normalization affects ranking.

| Detector | Precision | Recall | F1 | FPR | PR-AUC (AP) |
| --- | --- | --- | --- | --- | --- |
| autoencoder | 0.160 | 0.469 | 0.239 | 0.292 | 0.152 |
| max_abs_z | 0.164 | 0.463 | 0.242 | 0.281 | 0.172 |

| Series | Positive test windows | AE F1 | Stat F1 | AE FPR | Stat FPR |
| --- | --- | --- | --- | --- | --- |
| ec2_cpu_utilization_24ae8d | 54 | 0.361 | 0.242 | 0.111 | 0.089 |
| ec2_cpu_utilization_53ea38 | 0 | 0.000 | 0.000 | 0.091 | 0.000 |
| ec2_cpu_utilization_5f5533 | 0 | 0.000 | 0.000 | 1.000 | 1.000 |
| ec2_cpu_utilization_77c1ca | 0 | 0.000 | 0.000 | 0.010 | 0.020 |
| ec2_cpu_utilization_825cc2 | 0 | 0.000 | 0.000 | 0.020 | 0.020 |
| ec2_cpu_utilization_ac20cd | 52 | 0.509 | 0.536 | 0.638 | 0.638 |
| ec2_cpu_utilization_c6585a | 0 | 0.000 | 0.000 | 0.061 | 0.081 |
| ec2_cpu_utilization_fe7f93 | 0 | 0.000 | 0.000 | 0.040 | 0.040 |
| ec2_disk_write_bytes_1ef3de | 0 | 0.000 | 0.000 | 0.248 | 0.162 |
| ec2_disk_write_bytes_c0d644 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| ec2_network_in_257a54 | 0 | 0.000 | 0.000 | 0.010 | 0.020 |
| ec2_network_in_5abac7 | 0 | 0.000 | 0.000 | 0.231 | 0.205 |
| elb_request_count_8c0756 | 27 | 0.194 | 0.324 | 0.014 | 0.056 |
| grok_asg_anomaly | 17 | 0.190 | 0.176 | 1.000 | 1.000 |
| iio_us-east-1_i-a2eb1cd9_NetworkIn | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| rds_cpu_utilization_cc0c53 | 27 | 0.429 | 0.429 | 1.000 | 1.000 |
| rds_cpu_utilization_e47b3b | 0 | 0.000 | 0.000 | 0.525 | 0.515 |

Twelve test segments have no positive windows: AP is undefined (null); binary precision/recall/F1 use zero_division=0. They remain in pooled false-alarm accounting.
No type classifier is trained on public data. NAB windows are broad annotation/evaluation intervals, not exact fault-duration ground truth; this custom window task is not the official NAB score.

## Confusion matrices for every synthetic seed

Fixed order [normal, spike, shift, drift] for pipelines; [spike, shift, drift] for independent heads.

### Seed 17

raw independent: `[[2, 20, 36], [0, 79, 57], [1, 62, 124]]`

autoencoder + raw: `[[542, 0, 8, 5], [9, 2, 17, 30], [12, 0, 76, 48], [64, 1, 52, 70]]`

max_abs_z + raw: `[[527, 0, 1, 27], [9, 2, 16, 31], [57, 0, 40, 39], [124, 0, 26, 37]]`

encoded independent: `[[0, 4, 54], [0, 27, 109], [0, 26, 161]]`

autoencoder + encoded: `[[542, 0, 4, 9], [9, 0, 4, 45], [12, 0, 27, 97], [64, 0, 23, 100]]`

max_abs_z + encoded: `[[527, 0, 0, 28], [9, 0, 4, 45], [57, 0, 16, 63], [124, 0, 17, 46]]`

concat independent: `[[1, 11, 46], [0, 42, 94], [0, 41, 146]]`

autoencoder + concat: `[[542, 0, 2, 11], [9, 1, 10, 38], [12, 0, 40, 84], [64, 0, 33, 90]]`

max_abs_z + concat: `[[527, 0, 0, 28], [9, 1, 9, 39], [57, 0, 22, 57], [124, 0, 15, 48]]`

Single-stage: `[[527, 0, 13, 15], [40, 0, 6, 12], [41, 0, 70, 25], [86, 0, 51, 50]]`

### Seed 29

raw independent: `[[0, 20, 32], [0, 67, 75], [0, 48, 139]]`

autoencoder + raw: `[[515, 0, 6, 34], [3, 0, 20, 29], [3, 0, 65, 74], [53, 0, 38, 96]]`

max_abs_z + raw: `[[526, 0, 0, 29], [3, 0, 20, 29], [40, 0, 49, 53], [111, 0, 26, 50]]`

encoded independent: `[[0, 6, 46], [0, 30, 112], [0, 19, 168]]`

autoencoder + encoded: `[[515, 0, 4, 36], [3, 0, 6, 43], [3, 0, 29, 110], [53, 0, 16, 118]]`

max_abs_z + encoded: `[[526, 0, 3, 26], [3, 0, 6, 43], [40, 0, 28, 74], [111, 0, 14, 62]]`

concat independent: `[[1, 7, 44], [0, 64, 78], [0, 39, 148]]`

autoencoder + concat: `[[515, 0, 0, 40], [3, 1, 7, 41], [3, 0, 64, 75], [53, 0, 35, 99]]`

max_abs_z + concat: `[[526, 0, 0, 29], [3, 1, 7, 41], [40, 0, 51, 51], [111, 0, 30, 46]]`

Single-stage: `[[540, 0, 2, 13], [39, 1, 1, 11], [47, 0, 60, 35], [101, 0, 31, 55]]`

### Seed 43

raw independent: `[[0, 2, 52], [1, 19, 121], [3, 5, 180]]`

autoencoder + raw: `[[528, 0, 0, 25], [8, 0, 2, 44], [8, 1, 19, 113], [60, 3, 5, 120]]`

max_abs_z + raw: `[[527, 0, 0, 26], [9, 0, 2, 43], [35, 1, 19, 86], [120, 1, 5, 62]]`

encoded independent: `[[0, 15, 39], [0, 31, 110], [2, 37, 149]]`

autoencoder + encoded: `[[528, 0, 4, 21], [8, 0, 11, 35], [8, 0, 29, 104], [60, 2, 24, 102]]`

max_abs_z + encoded: `[[527, 0, 8, 18], [9, 0, 9, 36], [35, 0, 27, 79], [120, 2, 11, 55]]`

concat independent: `[[0, 15, 39], [2, 60, 79], [2, 43, 143]]`

autoencoder + concat: `[[528, 1, 6, 18], [8, 0, 12, 34], [8, 1, 57, 75], [60, 2, 35, 91]]`

max_abs_z + concat: `[[527, 0, 4, 22], [9, 0, 11, 34], [35, 1, 46, 59], [120, 2, 18, 48]]`

Single-stage: `[[526, 0, 4, 23], [40, 2, 1, 11], [46, 0, 61, 34], [91, 0, 40, 57]]`
