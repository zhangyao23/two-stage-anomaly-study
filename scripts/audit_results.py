"""Recompute saved metrics, denominators and propagation bounds from prediction CSVs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from anomaly_study.experiment import flatten_metrics
from anomaly_study.metrics import cascade, detection, diagnosis


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equal(actual, expected):
    if isinstance(expected, dict):
        for key in expected:
            equal(actual[key], expected[key])
    elif expected is None:
        assert actual is None
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    else:
        assert actual == expected


def check_spans(manifest):
    spans = []
    for split, entry in manifest.items():
        for sequence, start, end in entry["spans"]:
            for old_split, old_seq, old_start, old_end in spans:
                if sequence == old_seq and split != old_split:
                    assert end <= old_start or old_end <= start
            spans.append((split, sequence, start, end))


def audit_synthetic(root):
    config = load(root / "config.json")
    n = 0
    flat = []
    for seed in config["seeds"]:
        result = load(root / f"seed_{seed}.json")
        flat.append(flatten_metrics(result))
        predictions = np.genfromtxt(
            root / f"predictions_{seed}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        truth = predictions["truth"]
        n += len(truth)
        assert len(truth) == result["splits"]["test"]["windows"]
        assert len(set(predictions["window_id"])) == len(truth)
        check_spans(result["splits"])
        for detector, score_col, flag_col in [
            ("autoencoder", "ae_score", "ae_flag"),
            ("max_abs_z", "stat_score", "stat_flag"),
        ]:
            scores = predictions[score_col].astype(np.float32)
            threshold = result["detection"][detector]["threshold"]
            equal(detection(truth, scores, threshold), result["detection"][detector])
            np.testing.assert_array_equal(scores > threshold, predictions[flag_col])
            for name, h in result["heads"].items():
                full = cascade(scores, threshold, predictions[name])
                equal(diagnosis(truth, full, [0, 1, 2, 3]), h["end_to_end"][detector])
                for k in (1, 2, 3):
                    selected = truth == k
                    assert np.mean(full[selected] == k) <= np.mean(scores[selected] > threshold)
        for name, h in result["heads"].items():
            anomalies = truth > 0
            equal(
                diagnosis(truth[anomalies], predictions[name][anomalies], [1, 2, 3]),
                h["independent_anomaly_diagnosis"],
            )
        equal(
            diagnosis(truth, predictions["single_stage"], [0, 1, 2, 3]),
            result["single_stage"]["diagnosis"],
        )
    summary = {
        key: {
            "mean": float(np.mean([f[key] for f in flat])),
            "std": float(np.std([f[key] for f in flat], ddof=1)) if len(flat) > 1 else None,
        }
        for key in flat[0]
    }
    equal(summary, load(root / "summary.json"))
    print(
        f"Synthetic audit: {len(config['seeds'])} seeds, {n} windows, metrics/matrices/gates/spans consistent"
    )


def audit_public(root):
    result = load(root / "results.json")
    assert len(result["series"]) == 17
    n = 0
    pooled = {
        name: {"truth": [], "normalized": [], "margin": []} for name in ("autoencoder", "max_abs_z")
    }
    for series in result["series"]:
        with (root / f"predictions_{series['series']}.csv").open() as f:
            rows = list(csv.DictReader(f))
        truth = np.array([int(r["truth"]) for r in rows])
        n += len(rows)
        assert len(rows) == series["splits"]["test"]["windows"]
        check_spans(series["splits"])
        for detector, prefix in [("autoencoder", "ae"), ("max_abs_z", "stat")]:
            scores = np.array([float(r[prefix + "_score"]) for r in rows], dtype=np.float32)
            threshold = float(rows[0][prefix + "_threshold"])
            assert all(float(r[prefix + "_threshold"]) == threshold for r in rows)
            equal(detection(truth, scores, threshold), series["detection"][detector])
            pooled[detector]["truth"].extend(truth.tolist())
            pooled[detector]["normalized"].extend((scores / max(threshold, 1e-12)).tolist())
            pooled[detector]["margin"].extend((scores - threshold).tolist())
    for detector, metrics in result["pooled"].items():
        cm = np.sum(
            [s["detection"][detector]["confusion_matrix"] for s in result["series"]], axis=0
        )
        np.testing.assert_array_equal(cm, metrics["confusion_matrix"])
        assert cm.sum() == n
        values = pooled[detector]
        exact = detection(values["truth"], values["margin"], 0.0)
        ranked = detection(values["truth"], values["normalized"], 1.0)
        for key in ["precision", "recall", "f1", "false_positive_rate"]:
            equal(exact[key], metrics[key])
        equal(ranked["pr_auc_ap"], metrics["pr_auc_ap"])
    print(
        f"Public audit: 17 series, {n} windows, per-series metrics and pooled confusion counts consistent"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)
    audit_synthetic(root / "quick")
    audit_synthetic(root / "synthetic")
    audit_public(root / "public")


if __name__ == "__main__":
    main()
