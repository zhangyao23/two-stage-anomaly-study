"""Figures built only from measured result files."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES


def synthetic_plots(output, results, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    names = ["raw", "encoded", "concat"]
    x = np.arange(3)
    for ax, prefix, title in zip(
        axes,
        ["diagnosis", "cascade/autoencoder"],
        ["All true anomalies (3 classes)", "All windows (4 classes)"],
        strict=True,
    ):
        keys = [f"{prefix}/{n}/macro_f1" for n in names]
        ax.bar(
            x,
            [summary[k]["mean"] for k in keys],
            width=0.6,
            yerr=[summary[k]["std"] or 0 for k in keys],
            capsize=4,
        )
        ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Macro-F1", title=title)
    single = summary["single_stage/macro_f1"]
    axes[1].axhline(single["mean"], color="black", ls="--", label="Single-stage MLP")
    axes[1].legend(fontsize=8)
    fig.suptitle("Independent classification and AE cascade; mean +/- seed SD")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    propagation_plot(output, results)
    first = results[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, kind in zip(axes, names, strict=True):
        cm = np.array(first["heads"][kind]["end_to_end"]["autoencoder"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set(
            xticks=range(4),
            xticklabels=CLASSES,
            yticks=range(4),
            yticklabels=CLASSES,
            xlabel="Predicted",
            ylabel="True",
            title=f"AE + {kind} (seed {first['seed']})",
        )
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)
    failure_plot(output, first["seed"])


def propagation_plot(output, results):
    independent, gated, detected = [], [], []
    for result in results:
        values = np.genfromtxt(
            output / f"predictions_{result['seed']}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        truth = values["truth"]
        independent.append([np.mean(values["raw"][truth == k] == k) for k in (1, 2, 3)])
        gated.append(
            [
                np.mean((values["raw"][truth == k] == k) & (values["ae_flag"][truth == k] == 1))
                for k in (1, 2, 3)
            ]
        )
        detected.append([np.mean(values["ae_flag"][truth == k]) for k in (1, 2, 3)])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = np.arange(3)
    for offset, data, label in [
        (-0.25, independent, "Raw head: all true anomalies"),
        (0, gated, "AE + raw: end-to-end recall"),
        (0.25, detected, "AE detection recall (ceiling)"),
    ]:
        ax.bar(x + offset, np.mean(data, axis=0), width=0.25, label=label)
    ax.set(
        xticks=x,
        xticklabels=CLASSES[1:],
        ylim=(0, 1),
        ylabel="Class recall",
        title="Same denominators: misses cannot reach the classifier",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "error_propagation.png", dpi=160)
    plt.close(fig)


def failure_plot(output, seed):
    cases = json.loads((output / f"failures_{seed}.json").read_text())
    if cases:
        fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.2 * len(cases)), layout="constrained")
        for ax, case in zip(np.atleast_1d(axes), cases, strict=True):
            ax.plot(np.asarray(case["series"]))
            ax.set_title(
                f"{case['category']}: {case['window_id']} | "
                f"error={case['ae_score']:.3f}, threshold={case['ae_threshold']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("Offset within synthetic window")
        fig.savefig(output / "failures.png", dpi=140)
        plt.close(fig)


def public_plot(output, results):
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    y = np.arange(len(results))
    for offset, detector in [(-0.18, "autoencoder"), (0.18, "max_abs_z")]:
        ax.barh(
            y + offset,
            [r["detection"][detector]["f1"] for r in results],
            height=0.35,
            label=detector,
        )
    ax.set(
        yticks=y,
        yticklabels=[r["series"] for r in results],
        xlim=(0, 1),
        xlabel="Unadjusted window F1 (zero when no positives/predictions)",
        title="NAB AWS: every file, chronological held-out final 20%",
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.legend()
    fig.savefig(output / "per_series_f1.png", dpi=160)
    plt.close(fig)
