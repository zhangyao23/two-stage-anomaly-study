"""NAB AWS adapter. No invented anomaly types, no NAB leaderboard scoring."""

import csv
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import SPLITS, window_sequence
from .experiment import environment, fit_detectors, save_json, split_manifest
from .metrics import detection

NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
RAW = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "two-stage-anomaly-study"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_nab(root, subset="realAWSCloudwatch"):
    if subset not in {"realAWSCloudwatch", "realAdExchange"}:
        raise ValueError("Subset is not in the declared study protocol")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    listing = json.loads(
        fetch(f"https://api.github.com/repos/numenta/NAB/contents/data/{subset}?ref={NAB_COMMIT}")
    )
    files = sorted(
        f"data/{subset}/" + entry["name"] for entry in listing if entry["name"].endswith(".csv")
    )
    files += ["labels/combined_windows.json", "LICENSE.txt", "data/README.md", "README.md"]

    def get_file(relative):
        content = fetch(RAW + relative)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(get_file, files))
    manifest = {
        "repository": "https://github.com/numenta/NAB",
        "commit": NAB_COMMIT,
        "license": "MIT",
        "subset": f"entire {subset} directory",
        "files": entries,
    }
    save_json(root / "manifest.json", manifest)
    print(
        f"Downloaded {len(files) - 4} {subset} series; raw files remain in ignored data directory."
    )
    return manifest


def read_series(path, intervals):
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    timestamps = np.array([r["timestamp"] for r in rows], dtype="datetime64[s]")
    if np.any(np.diff(timestamps).astype(int) < 0):
        raise ValueError("Timestamps must be nondecreasing")
    x = np.array([float(r["value"]) for r in rows], dtype=np.float32)[:, None]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite public measurements")
    # Upstream has repeated timestamps, sometimes with different values. Aggregate
    # locally at that timestamp, before splitting, without consulting anomaly labels.
    unique, inverse, counts = np.unique(timestamps, return_inverse=True, return_counts=True)
    values = np.bincount(inverse, weights=x[:, 0]) / counts
    quality = {
        "source_rows": len(rows),
        "unique_timestamps": len(unique),
        "duplicate_rows_aggregated": int(len(rows) - len(unique)),
        "duplicate_policy": "arithmetic mean per timestamp before chronological splitting",
    }
    timestamps = unique
    x = values.astype(np.float32)[:, None]
    y = np.zeros(len(timestamps), dtype=int)
    for start, end in intervals:
        y[(timestamps >= np.datetime64(start)) & (timestamps <= np.datetime64(end))] = 1
    return x, y, quality


def chronological_splits(x, y, sequence_id, width=16, stride=8):
    bounds = [0, int(len(x) * 0.4), int(len(x) * 0.6), int(len(x) * 0.8), len(x)]
    return {
        split: window_sequence(
            x[start:end], y[start:end], width, stride, sequence_id, split, offset=start
        )
        for split, start, end in zip(SPLITS, bounds[:-1], bounds[1:], strict=True)
    }


def verify_download(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != NAB_COMMIT:
        raise ValueError("Unexpected NAB revision")
    for entry in manifest["files"]:
        actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Download hash mismatch: {entry['path']}")
    return manifest


def run_public(config, root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = verify_download(root)
    save_json(output / "data_manifest.json", manifest)
    save_json(output / "config.json", config)
    labels = json.loads((root / "labels/combined_windows.json").read_text(encoding="utf-8"))
    results, pooled = (
        [],
        {name: {"y": [], "score": [], "margin": []} for name in ("autoencoder", "max_abs_z")},
    )
    # Single fixed initialization seed; data is fixed, no claimed multi-seed public uncertainty.
    for entry in manifest["files"]:
        if not entry["path"].endswith(".csv"):
            continue
        relative = entry["path"]
        print(f"NAB {Path(relative).stem}", flush=True)
        x, y, quality = read_series(root / relative, labels[relative.removeprefix("data/")])
        splits = chronological_splits(x, y, Path(relative).stem, config["window"], config["stride"])
        _, _, scores, thresholds, det, timing, meta = fit_detectors(
            splits, config, config["seeds"][0]
        )
        failures = export_public_predictions(output, splits["test"], scores, thresholds)
        results.append(
            {
                "series": Path(relative).stem,
                "data_quality": quality,
                "failure_cases": failures,
                "detection": det,
                "timing": timing,
                "detector_metadata": meta,
                "splits": split_manifest(splits),
            }
        )
        for detector, values in scores.items():
            pooled[detector]["y"].extend(splits["test"].y.tolist())
            # Ranking uses per-series score / calibration threshold, clearly a choice of pooling.
            normalized = values / max(thresholds[detector], 1e-12)
            pooled[detector]["score"].extend(normalized.tolist())
            pooled[detector]["margin"].extend((values - thresholds[detector]).tolist())
    summary = {}
    for name, values in pooled.items():
        summary[name] = detection(values["y"], values["score"], 1.0)
        # Threshold could be zero for a constant series. Preserve the exact decisions.
        exact = detection(values["y"], values["margin"], 0.0)
        for key in (
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "confusion_matrix",
            "predicted_anomalies",
        ):
            summary[name][key] = exact[key]
    save_json(
        output / "results.json",
        {
            "series": results,
            "pooled": summary,
            "environment": environment(),
            "seed": config["seeds"][0],
            "status": "completed",
        },
    )
    from .plots import public_plot

    public_plot(output, results)
    return results


def export_public_predictions(output, test, scores, thresholds):
    """Publish derived scores and row spans, never source measurement values."""
    series = test.spans[0][0]
    path = output / f"predictions_{series}.csv"
    failures = []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["window_id", "truth", "ae_score", "ae_threshold", "stat_score", "stat_threshold"]
        )
        for i, wid in enumerate(test.ids):
            writer.writerow(
                [
                    wid,
                    test.y[i],
                    scores["autoencoder"][i],
                    thresholds["autoencoder"],
                    scores["max_abs_z"][i],
                    thresholds["max_abs_z"],
                ]
            )
    for name, values in scores.items():
        flag = values > thresholds[name]
        for category, mask in [
            ("false_negative", (test.y > 0) & ~flag),
            ("false_positive", (test.y == 0) & flag),
        ]:
            if mask.any():
                i = int(np.flatnonzero(mask)[0])
                failures.append(
                    {
                        "detector": name,
                        "category": category,
                        "window_id": test.ids[i],
                        "score": float(values[i]),
                        "threshold": thresholds[name],
                        "selection": "first matching held-out window",
                    }
                )
    return failures
