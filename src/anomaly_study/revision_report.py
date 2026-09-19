"""Generate revised report and figures from frozen confirmation artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES
from .experiment import save_json
from .features import contextual_splits
from .report import mean_sd, table
from .revision import read_json


def comparisons(runs):
    return {
        "v1 AE + concat": [r["v1_heads"]["concat"]["end_to_end"]["autoencoder"] for r in runs],
        "balanced matched concat": [r["revised"]["concat"]["cascade"] for r in runs],
        "AE + temporal (3 classes)": [r["revised"]["temporal"]["cascade"] for r in runs],
        "AE + temporal rejection": [r["revised"]["single_temporal"]["cascade"] for r in runs],
        "temporal single-stage": [r["revised"]["single_temporal"]["standalone"] for r in runs],
    }


def figures(root, runs, public):
    groups = comparisons(runs)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    values = [[m["macro_f1"] for m in ms] for ms in groups.values()]
    ax.bar(
        np.arange(len(groups)),
        np.mean(values, axis=1),
        yerr=np.std(values, axis=1, ddof=1),
        capsize=4,
    )
    ax.set(
        xticks=np.arange(len(groups)),
        xticklabels=[
            "v1\nAE + concat",
            "Balanced + matched\nAE + concat",
            "Temporal\n3-class head",
            "Temporal + reject\nvalidation selected",
            "Temporal\nsingle-stage",
        ],
        ylim=(0, 1),
        ylabel="Four-class macro-F1",
        title="Fresh synthetic confirmation: same windows, mean +/- seed SD",
    )
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(root / "revision_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for offset, key, label in [
        (-0.18, "v1 AE + concat", "v1 AE + concat"),
        (0.18, "AE + temporal rejection", "Revised selected cascade"),
    ]:
        rec = np.array([m["recall_per_class"] for m in groups[key]])
        axes[0].bar(np.arange(4) + offset, rec.mean(0), width=0.36, label=label)
    axes[0].set(
        xticks=range(4),
        xticklabels=CLASSES,
        ylim=(0, 1),
        ylabel="Recall",
        title="End-to-end per-class recall",
    )
    axes[0].legend(fontsize=8)
    cm = np.sum([m["confusion_matrix"] for m in groups["AE + temporal rejection"]], axis=0)
    axes[1].imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            axes[1].text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    axes[1].set(
        xticks=range(4),
        xticklabels=CLASSES,
        yticks=range(4),
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title="Selected cascade: all 2,808 windows",
    )
    fig.savefig(root / "revision_recall.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, (name, result) in zip(axes, public.items(), strict=True):
        for offset, (detector, m) in zip([-0.25, 0, 0.25], result["pooled"].items(), strict=True):
            ax.bar(
                np.arange(3) + offset,
                [m["f1"], m["recall"], m["false_positive_rate"]],
                width=0.25,
                label=detector,
            )
        ax.set(
            xticks=range(3),
            xticklabels=["F1", "Recall", "FPR (lower better)"],
            ylim=(0, 1),
            title=name,
        )
        ax.legend(fontsize=7)
    fig.savefig(root / "revision_public.png", dpi=160)
    plt.close(fig)


def export_failures(root, config, base, runs):
    cases = []
    for r in runs:
        seed = r["test_seed"]
        _, contexts = contextual_splits(base, r["train_seed"], config["history"], seed)
        p = np.genfromtxt(
            root / f"predictions_{seed}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        pred = p["single_temporal_cascade"]
        for label, mask in [
            ("missed_spike", (p["truth"] == 1) & (pred == 0)),
            ("missed_drift", (p["truth"] == 3) & (pred == 0)),
            ("wrong_type", (p["truth"] > 0) & (pred > 0) & (pred != p["truth"])),
            ("false_alarm", (p["truth"] == 0) & (pred > 0)),
        ]:
            if not mask.any():
                continue
            i = int(np.flatnonzero(mask)[0])
            cases.append(
                {
                    "seed": seed,
                    "category": label,
                    "window_id": str(p["window_id"][i]),
                    "truth": int(p["truth"][i]),
                    "prediction": int(pred[i]),
                    "independent_prediction": int(p["single_temporal"][i]),
                    "ae_score": float(p["ae_score"][i]),
                    "threshold": float(p["threshold"][i]),
                    "context": contexts["test"][i].tolist(),
                    "selection": "first matching window per seed/category",
                }
            )
    save_json(root / "failure_cases.json", cases)
    shown = cases[:4]
    fig, axes = plt.subplots(len(shown), 1, figsize=(9, 2.5 * len(shown)), layout="constrained")
    for ax, c in zip(np.atleast_1d(axes), shown, strict=True):
        ax.plot(np.array(c["context"]))
        ax.axvspan(
            config["history"] - base["window"], config["history"] - 1, color="grey", alpha=0.15
        )
        ax.set_title(
            f"Seed {c['seed']} {c['window_id']}: {c['category']}; "
            f"{CLASSES[c['truth']]} -> {CLASSES[c['prediction']]}",
            fontsize=9,
        )
        ax.set_xlabel("Causal context offset; shaded region = target window")
    fig.savefig(root / "revision_failures.png", dpi=140)
    plt.close(fig)
    return cases


def main():
    root = Path("results/revision")
    config = read_json(root / "config.json")
    base = read_json(config["base_config"])
    result = read_json(root / "confirmation.json")
    runs = result["runs"]
    public = {
        "AWS retrospective": read_json("results/revision_public/retrospective_aws.json"),
        "AdExchange confirmation": read_json("results/revision_public/confirmation.json"),
    }
    figures(root, runs, public)
    failures = export_failures(root, config, base, runs)
    lines = [
        "# Revision: confirmed gains and unresolved limits",
        "",
        "Protocol and validation selection were committed in `17bd171` before confirmation evaluation.",
        "Development seeds: 17/29/43; fresh test seeds: 101/211/307. Same generator, difficulty and all 936 windows per seed retained.",
        "Numbers are mean ± sample SD (3 seeds), not significance tests. Original files remain unchanged.",
        "",
        "## Same-test complete-pipeline comparison",
        "",
    ]
    groups = comparisons(runs)
    lines += table(
        ["System", "Macro-F1", *[f"{k} recall" for k in CLASSES]],
        [
            [name, mean_sd([m["macro_f1"] for m in ms])]
            + [mean_sd([m["recall_per_class"][i] for m in ms]) for i in range(4)]
            for name, ms in groups.items()
        ],
    )
    lines += [
        "The selected two-stage system improves markedly over v1, but the feature-matched single-stage model has a slightly higher mean.",
        "The evidence supports the feature/training repair; it does not establish that an AE gate is necessary or superior.",
        "",
        "## Matched-capacity representation controls",
        "",
    ]
    lines += table(
        ["Input", "Parameters", "Independent anomaly macro-F1", "End-to-end macro-F1"],
        [
            [
                kind,
                runs[0]["revised"][kind]["parameters"],
                mean_sd([r["revised"][kind]["independent"]["macro_f1"] for r in runs]),
                mean_sd([r["revised"][kind]["cascade"]["macro_f1"] for r in runs]),
            ]
            for kind in ["raw", "encoded", "concat", "temporal", "single_raw", "single_temporal"]
        ],
    )
    lines += [
        "All revised heads are within 1% of 6,400 trainable parameters, use train-fitted scaling and inverse-frequency cross-entropy, and run 100 epochs.",
        "Class weights are derived only from training counts and also define checkpoint validation loss. Head widths differ to match parameter budgets; this is not exact functional-capacity equivalence.",
        "Anomaly-only heads see identical training samples/update counts. Four-class heads see normal windows too and therefore more steps; the single-stage temporal comparator uses the exact same four-class head.",
        "Compared with v1, budget, scaling and class balance changed jointly. No one-factor causal attribution is claimed.",
        "The temporal representation adds 64 observations of causal context and a normal-trained rank-one cross-channel residual (PCA), not an anomaly-trained AE encoding.",
        "These features exploit the generator's shared normal variation and sparse channel disturbances; general multivariate real-fault benefit is untested.",
        "",
        "## Paired changes on every confirmation seed",
        "",
    ]
    lines += table(
        [
            "Train seed",
            "Fresh test seed",
            "v1 concat",
            "Selected revised",
            "Difference",
            "Matched single-stage",
        ],
        [
            [
                r["train_seed"],
                r["test_seed"],
                f"{r['v1_heads']['concat']['end_to_end']['autoencoder']['macro_f1']:.4f}",
                f"{r['revised']['single_temporal']['cascade']['macro_f1']:.4f}",
                f"{r['revised']['single_temporal']['cascade']['macro_f1'] - r['v1_heads']['concat']['end_to_end']['autoencoder']['macro_f1']:+.4f}",
                f"{r['revised']['single_temporal']['standalone']['macro_f1']:.4f}",
            ]
            for r in runs
        ],
    )
    lines += [
        "## Public detection: lower false alarms are not a universal improvement",
        "",
        "AWS validation selected rolling median/MAD innovation with lookback 128 and normal-calibration quantile 0.975.",
        "Selection maximized validation F1 subject to pooled validation FPR <=0.10. It used neither old test labels nor new AdExchange confirmation metrics.",
        "",
    ]
    for name, data in public.items():
        lines += [f"### {name}", ""]
        lines += table(
            ["Detector", "Precision", "Recall", "F1", "FPR", "AP"],
            [
                [det]
                + [
                    f"{m[k]:.4f}"
                    for k in ["precision", "recall", "f1", "false_positive_rate", "pr_auc_ap"]
                ]
                for det, m in data["pooled"].items()
            ],
        )
        first = next(iter(data["pooled"].values()))
        lines += [
            f"{len(data['series'])} series; {first['positives']} positive and {first['negatives']} negative test windows.",
            "",
        ]
    lines += [
        "On AWS the false-alarm constraint reduces FPR but sacrifices recall/F1: this is an explicit cost-sensitive operating point, not a Pareto improvement.",
        "The AdExchange confirmation set improves F1/AP and FPR at the same measured recall, but contains only seven positive windows: evidence is weak and not a network fault-type benchmark.",
        "Its FPR still exceeds 0.10: a validation FPR constraint is not a future-distribution guarantee. No post-confirmation retuning was performed.",
        "",
        "## Actual full-pipeline CPU timing",
        "",
        "Intel Core i7-13700H; one PyTorch CPU thread; 20 warmups/100 repeats. Includes train-fitted scaling, temporal features, MLP, and AE gate when enabled; excludes context assembly and I/O.",
        "The same first 1/128 test windows are used per seed; gate acceptance rate affects timing. These medians are not worst-case latency or a production SLA.",
        "",
    ]
    timing = read_json(root / "pipeline_timing.json")
    rows = []
    for gated in [False, True]:
        for size in [1, 128]:
            records = [
                r for r in timing["records"] if r["gated"] == gated and r["batch_size"] == size
            ]
            rows.append(
                [
                    "AE + temporal rejection" if gated else "Temporal single-stage",
                    size,
                    mean_sd([r["median_ms_per_batch"] for r in records]),
                    mean_sd([r["p95_ms_per_batch"] for r in records]),
                ]
            )
    lines += table(["System", "Batch", "Batch median ms", "Batch p95 ms"], rows)
    lines += [
        "Full model parameters: AE 13,544 + selected head 6,400 = 19,944. The matched single-stage has 6,400 neural parameters; both also store small PCA/scaler statistics.",
        "Public rolling timing includes rolling-statistic computation over the segment, with 20 warmups/20 repeats. Old AE detector timings exclude preprocessing and must not be treated as an equal-scope speed comparison.",
        "",
        "## Remaining actual failures",
        "",
    ]
    lines += table(
        [
            "Seed",
            "Window",
            "Category",
            "Truth",
            "Final",
            "Independent head",
            "AE score / threshold",
        ],
        [
            [
                c["seed"],
                c["window_id"],
                c["category"],
                CLASSES[c["truth"]],
                CLASSES[c["prediction"]],
                CLASSES[c["independent_prediction"]],
                f"{c['ae_score']:.3f} / {c['threshold']:.3f}",
            ]
            for c in failures
        ],
    )
    lines += [
        "Examples are first matches per seed/category, not handpicked successes. Their generated contexts are in `results/revision/failure_cases.json`.",
        "Drift end-to-end recall remains limited by the unchanged AE detector; correct independent predictions still disappear at the gate. Rejecting normal false alarms can also reject weak true anomalies.",
        "",
        "## Confusion matrices",
        "",
        "Rows=true, columns=predicted; order [normal, spike, shift, drift]. Independent diagnostic matrices retain the normal rejection column; macro-F1 there averages only anomaly classes.",
        "",
    ]
    for r in runs:
        lines += [f"### Test seed {r['test_seed']}", ""]
        for kind, m in r["revised"].items():
            lines += [
                f"{kind} independent: `{m['independent']['confusion_matrix']}`",
                "",
                f"{kind} cascade: `{m['cascade']['confusion_matrix']}`",
                "",
            ]
        lines += [
            f"Temporal single-stage: `{r['revised']['single_temporal']['standalone']['confusion_matrix']}`",
            "",
        ]
    Path("docs/REVISION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
