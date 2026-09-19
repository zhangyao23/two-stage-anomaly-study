"""Audit frozen selection and every derived confirmation prediction; no training needed."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from audit_results import check_spans, equal

from anomaly_study.metrics import detection
from anomaly_study.public_revision import choose_public
from anomaly_study.revision import read_json, score_predictions


def audit_synthetic():
    root = Path("results/revision")
    selection = read_json(root / "selection.json")
    development = [read_json(root / f"development_{s}.json") for s in [17, 29, 43]]
    means = {
        k: np.mean([r["heads"][k]["validation"]["cascade"]["macro_f1"] for r in development])
        for k in selection["validation_macro_f1"]
    }
    equal(means, selection["validation_macro_f1"])
    assert max(means, key=means.get) == selection["selected_cascade_head"]
    results = read_json(root / "confirmation.json")
    n = 0
    for r in results["runs"]:
        check_spans(r["splits"])
        p = np.genfromtxt(
            root / f"predictions_{r['test_seed']}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        n += len(p)
        assert len(p) == r["splits"]["test"]["windows"]
        score = p["ae_score"].astype(np.float32)
        threshold = p["threshold"][0]
        equal(detection(p["truth"], score, threshold), r["detection"]["autoencoder"])
        for name, metrics in r["revised"].items():
            actual, full = score_predictions(p["truth"], p[name], score, threshold)
            for unit in ["independent", "cascade", "standalone"]:
                equal(actual[unit], metrics[unit])
            np.testing.assert_array_equal(full, p[name + "_cascade"])
            assert np.array(metrics["cascade"]["confusion_matrix"]).sum() == len(p)
        assert all(abs(h["parameters"] - 6400) / 6400 < 0.01 for h in r["revised"].values())
    print(f"Revision synthetic: frozen selection, capacity and all {n} test windows verified.")


def audit_public():
    root = Path("results/revision_public")
    selection = read_json(root / "selection.json")
    assert choose_public(selection["validation_metrics"], 0.1) == selection["selected"]
    total = 0
    for stage in ["development", "retrospective_aws", "confirmation"]:
        result = read_json(root / f"{stage}.json")
        pooled = defaultdict(lambda: {"truth": [], "margin": [], "rank": []})
        for r in result["series"]:
            check_spans(r["splits"])
            groups = defaultdict(list)
            with (root / f"{stage}_{r['series']}.csv").open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    groups[row["detector"]].append(row)
            for detector, rows in groups.items():
                truth = np.array([int(row["truth"]) for row in rows])
                dtype = np.float64 if detector.startswith("rolling_") else np.float32
                scores = np.array([float(row["score"]) for row in rows], dtype=dtype)
                threshold = float(rows[0]["threshold"])
                equal(detection(truth, scores, threshold), r["metrics"][detector])
                pooled[detector]["truth"].extend(truth.tolist())
                pooled[detector]["margin"].extend((scores - threshold).tolist())
                pooled[detector]["rank"].extend((scores / max(threshold, 1e-12)).tolist())
            total += len(next(iter(groups.values())))
        for detector, p in pooled.items():
            actual = detection(p["truth"], p["margin"], 0.0)
            actual["pr_auc_ap"] = detection(p["truth"], p["rank"], 1.0)["pr_auc_ap"]
            equal(actual, result["pooled"][detector])
    print(
        f"Revision public: validation-only selection and {total} validation/test windows verified."
    )


if __name__ == "__main__":
    audit_synthetic()
    audit_public()
