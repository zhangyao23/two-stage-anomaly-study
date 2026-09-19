"""Generate detailed Markdown tables from measured JSON/CSV, without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np


def mean_sd(values):
    return (
        f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"
        if len(values) > 1
        else f"{values[0]:.3f}"
    )


def table(header, rows):
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        "",
    ]


def generate(synthetic, public, output):
    synthetic, public = Path(synthetic), Path(public)
    config = json.loads((synthetic / "config.json").read_text())
    runs = [json.loads((synthetic / f"seed_{seed}.json").read_text()) for seed in config["seeds"]]
    pub = json.loads((public / "results.json").read_text())
    lines = [
        "# Measured experiment results",
        "",
        "Generated from stored JSON/CSV with `python -m anomaly_study.report`.",
        "No point adjustment. All values are fractions; ± is sample SD over seeds, not a confidence interval.",
        "",
        "## Synthetic detection",
        "",
        "Positive class: any synthetic anomaly. Evaluation unit: a 16-point window, stride 8.",
        "",
    ]
    keys = ["precision", "recall", "f1", "false_positive_rate", "pr_auc_ap"]
    lines += table(
        ["Detector", "Precision", "Recall", "F1", "FPR", "PR-AUC (AP)"],
        [
            [name] + [mean_sd([r["detection"][name][key] for r in runs]) for key in keys]
            for name in ("autoencoder", "max_abs_z")
        ],
    )
    lines += [
        "Single-stage binary views: summing abnormal probabilities at 0.5 differs from four-class argmax.",
        "",
    ]
    lines += table(
        ["Single-stage decision", "Precision", "Recall", "F1", "FPR", "AP"],
        [
            ["P(anomaly) > 0.5"]
            + [
                mean_sd([r["single_stage"]["detection_p_anomaly_gt_half"][key] for r in runs])
                for key in keys
            ],
            ["argmax is not normal"]
            + [
                mean_sd([r["single_stage"]["detection_argmax"][key] for r in runs])
                for key in keys[:-1]
            ]
            + ["not defined from hard predictions"],
        ],
    )
    lines += [
        "## Independent diagnosis on ALL true anomalies",
        "",
        "No detector filtering here. Fixed labels: spike, shift, drift. Different denominator and class set from the four-class cascade.",
        "",
    ]
    rows = []
    for name in ("raw", "encoded", "concat"):
        metrics = [r["heads"][name]["independent_anomaly_diagnosis"] for r in runs]
        rows.append(
            [name, mean_sd([m["macro_f1"] for m in metrics])]
            + [mean_sd([m["recall_per_class"][i] for m in metrics]) for i in range(3)]
        )
    lines += table(["Input", "Macro-F1", "Spike recall", "Shift recall", "Drift recall"], rows)
    lines += [
        "## Complete pipeline on ALL test windows",
        "",
        "Includes false alarms and missed anomalies. Rows in confusion matrices are truth; columns are predictions.",
        "",
    ]
    rows = []
    for detector in ("autoencoder", "max_abs_z"):
        for name in ("raw", "encoded", "concat"):
            metrics = [r["heads"][name]["end_to_end"][detector] for r in runs]
            rows.append(
                [f"{detector} + {name}", mean_sd([m["macro_f1"] for m in metrics])]
                + [mean_sd([m["recall_per_class"][i] for m in metrics]) for i in range(4)]
            )
    metrics = [r["single_stage"]["diagnosis"] for r in runs]
    rows.append(
        ["single-stage", mean_sd([m["macro_f1"] for m in metrics])]
        + [mean_sd([m["recall_per_class"][i] for m in metrics]) for i in range(4)]
    )
    lines += table(
        ["Model", "Macro-F1", "Normal recall", "Spike recall", "Shift recall", "Drift recall"], rows
    )
    lines += [
        "## Detection ceilings and error propagation",
        "",
        "End-to-end class recall = detection recall for that class × classification accuracy conditional on detection.",
        "The following AE detection recalls are upper bounds, even for a perfect downstream classifier.",
        "",
    ]
    per_seed = []
    for r in runs:
        p = np.genfromtxt(
            synthetic / f"predictions_{r['seed']}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        per_seed.append([np.mean(p["ae_flag"][p["truth"] == k]) for k in (1, 2, 3)])
    lines += table(
        ["Spike", "Shift", "Drift"], [[mean_sd([r[i] for r in per_seed]) for i in range(3)]]
    )
    lines += ["## Capacity and training budget", ""]
    first = runs[0]
    rows = []
    for name, h in first["heads"].items():
        rows.append(
            [
                name,
                h["input_dimension"],
                h["head_parameters"],
                h["cascade_parameters"],
                h["training"]["training_samples"],
                h["training"]["optimizer_steps"],
            ]
        )
    single = first["single_stage"]
    rows.append(
        [
            "single-stage",
            config["window"] * config["channels"],
            single["parameters"],
            single["parameters"],
            single["training"]["training_samples"],
            single["training"]["optimizer_steps"],
        ]
    )
    lines += table(
        [
            "Model",
            "Input dim",
            "Head params",
            "Total with AE",
            "Train windows (seed 17)",
            "Adam steps",
        ],
        rows,
    )
    lines += [
        f"AE parameters: {first['detector_metadata']['autoencoder_parameters']}. Statistical detector: no neural parameters; train means/scales and one calibration threshold.",
        "All three anomaly heads share samples, order seed, hidden width 32, Adam lr 0.001, batch 128 and 35 epochs; validation loss selects the checkpoint.",
        "Encoded inputs are frozen AE outputs without separate rescaling. Capacity, feature scale, class imbalance and optimization confound a pure representation claim.",
        "The single-stage baseline sees all labeled training windows, hence more optimizer steps at the same epoch budget. This is not a matched-supervision or matched-compute comparison.",
        "",
        "## Measured CPU inference",
        "",
        "Hardware: Intel Core i7-13700H, Windows 11, CPU only, one PyTorch thread. Timings exclude data loading, windowing and standardization.",
        "20 warmups + 100 repetitions with perf_counter_ns and torch.inference_mode. Batch = first 128 held-out synthetic windows. Gated cascades reuse the encoding and classify only flagged windows.",
        "Numbers below summarize per-run batch medians across the three seeds; per-run p95 and per-window amortized times are in JSON. This is not online request latency or a production SLA.",
        "",
    ]
    lines += table(
        ["Operation", "Batch median ms (mean ± seed SD)"],
        [
            [name, mean_sd([r["timing"][name]["median_ms_per_batch"] for r in runs])]
            for name in first["timing"]
        ],
    )
    lines += [
        "## Public NAB AWS detection",
        "",
        "All 17 files; separate model/scaler/threshold per file; final 20% held out. One fixed initialization seed (17), not three public seeds.",
        "PR-AUC is average precision pooled over score / per-series threshold. Per-series AP is also stored. Pooling normalization affects ranking.",
        "",
    ]
    lines += table(
        ["Detector", "Precision", "Recall", "F1", "FPR", "PR-AUC (AP)"],
        [[name] + [f"{m[key]:.3f}" for key in keys] for name, m in pub["pooled"].items()],
    )
    lines += table(
        ["Series", "Positive test windows", "AE F1", "Stat F1", "AE FPR", "Stat FPR"],
        [
            [
                s["series"],
                s["detection"]["autoencoder"]["positives"],
                f"{s['detection']['autoencoder']['f1']:.3f}",
                f"{s['detection']['max_abs_z']['f1']:.3f}",
                f"{s['detection']['autoencoder']['false_positive_rate']:.3f}",
                f"{s['detection']['max_abs_z']['false_positive_rate']:.3f}",
            ]
            for s in pub["series"]
        ],
    )
    lines += [
        "Twelve test segments have no positive windows: AP is undefined (null); binary precision/recall/F1 use zero_division=0. They remain in pooled false-alarm accounting.",
        "No type classifier is trained on public data. NAB windows are broad annotation/evaluation intervals, not exact fault-duration ground truth; this custom window task is not the official NAB score.",
        "",
        "## Confusion matrices for every synthetic seed",
        "",
        "Fixed order [normal, spike, shift, drift] for pipelines; [spike, shift, drift] for independent heads.",
        "",
    ]
    for r in runs:
        lines += [f"### Seed {r['seed']}", ""]
        for name, h in r["heads"].items():
            lines += [
                f"{name} independent: `{h['independent_anomaly_diagnosis']['confusion_matrix']}`",
                "",
            ]
            for detector, m in h["end_to_end"].items():
                lines += [f"{detector} + {name}: `{m['confusion_matrix']}`", ""]
        lines += [f"Single-stage: `{r['single_stage']['diagnosis']['confusion_matrix']}`", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default="results/synthetic")
    parser.add_argument("--public", default="results/public")
    parser.add_argument("--output", default="docs/RESULTS.md")
    args = parser.parse_args()
    generate(args.synthetic, args.public, args.output)


if __name__ == "__main__":
    main()
