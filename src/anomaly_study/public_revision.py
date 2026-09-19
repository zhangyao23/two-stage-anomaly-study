"""Validation-selected public detectors, retrospective AWS and fresh domain confirmation."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .data import calibrate, only_normal
from .experiment import environment, fit_detectors, save_json, split_manifest
from .features import rolling_innovation
from .metrics import detection
from .models import benchmark
from .public_data import chronological_splits, read_series, verify_download
from .revision import config_digest, read_json


def window_scores(points, width, stride):
    return np.lib.stride_tricks.sliding_window_view(points, width)[::stride].max(1)


def candidates_for(raw, splits, base, config, evaluation):
    scoring = splits | {"test": splits[evaluation]}
    ae, normalized, evaluated, _, _, timing, metadata = fit_detectors(scoring, base, 17)
    import torch

    with torch.inference_mode():
        cal_x = normalized["calibration"]
        cal_scores = {
            "autoencoder": ((ae(cal_x) - cal_x) ** 2).mean(1).numpy(),
            "max_abs_z": cal_x.abs().amax(1).numpy(),
        }
    bounds = [0, int(len(raw) * 0.4), int(len(raw) * 0.6), int(len(raw) * 0.8), len(raw)]
    index = {"calibration": 1, "validation": 2, "test": 3}
    train_values = splits["train"].x[splits["train"].y == 0].reshape(-1)
    initial = float(np.median(train_values))
    floor = max(float(train_values.std()) * config["public_scale_floor_fraction"], 1e-6)
    for lookback in config["public_lookbacks"]:
        name = f"rolling_{lookback}"
        for split, destination in [("calibration", cal_scores), (evaluation, evaluated)]:
            i = index[split]
            values = raw[bounds[i] : bounds[i + 1], 0]
            points = rolling_innovation(values, lookback, initial, floor)
            destination[name] = window_scores(points, base["window"], base["stride"])
        # Includes rolling feature creation and score calculation for this measured segment.
        values = raw[bounds[index[evaluation]] : bounds[index[evaluation] + 1], 0]
        timing[name] = benchmark(
            lambda values=values, lookback=lookback: window_scores(
                rolling_innovation(values, lookback, initial, floor), base["window"], base["stride"]
            ),
            len(splits[evaluation].y),
            repeats=20,
        )
    metadata["rolling_initial_training_median"] = initial
    metadata["rolling_scale_floor"] = floor
    candidates = {}
    calibration = only_normal(splits["calibration"])
    mask = splits["calibration"].y == 0
    for name, scores in evaluated.items():
        for q in config["public_quantiles"]:
            threshold = calibrate(cal_scores[name][mask], calibration, q)
            key = f"{name}@{q}"
            candidates[key] = {
                "scores": scores,
                "threshold": threshold,
                "metrics": detection(splits[evaluation].y, scores, threshold),
            }
    return candidates, timing, metadata


def pool(records, keys):
    result = {}
    for key in keys:
        truth, margin, rank = [], [], []
        for r in records:
            candidate = r["candidates"][key]
            scores, threshold = candidate["scores"], candidate["threshold"]
            truth.extend(r["truth"])
            margin.extend((scores - threshold).tolist())
            rank.extend((scores / max(threshold, 1e-12)).tolist())
        result[key] = detection(truth, margin, 0.0)
        result[key]["pr_auc_ap"] = detection(truth, rank, 1.0)["pr_auc_ap"]
    return result


def choose_public(metrics, fpr_limit):
    eligible = [k for k, m in metrics.items() if m["false_positive_rate"] <= fpr_limit]
    if eligible:
        return max(eligible, key=lambda k: (metrics[k]["f1"], -metrics[k]["false_positive_rate"]))
    return min(metrics, key=lambda k: (metrics[k]["false_positive_rate"], -metrics[k]["f1"]))


def run_subset(root, base, config, evaluation, selected=None):
    manifest = verify_download(root)
    labels = read_json(root / "labels/combined_windows.json")
    records = []
    for entry in manifest["files"]:
        if not entry["path"].endswith(".csv"):
            continue
        path = entry["path"]
        print(f"Public {evaluation}: {Path(path).stem}", flush=True)
        raw, y, quality = read_series(root / path, labels[path.removeprefix("data/")])
        splits = chronological_splits(raw, y, Path(path).stem, base["window"], base["stride"])
        candidates, timing, meta = candidates_for(raw, splits, base, config, evaluation)
        if selected is not None:
            # Never rank/select using confirmation performance. Keep prespecified comparators only.
            keys = dict.fromkeys(["autoencoder@0.95", "max_abs_z@0.95", selected])
            candidates = {k: candidates[k] for k in keys}
        records.append(
            {
                "series": Path(path).stem,
                "truth": splits[evaluation].y.tolist(),
                "ids": splits[evaluation].ids,
                "candidates": candidates,
                "timing": timing,
                "model_metadata": meta,
                "data_quality": quality,
                "splits": split_manifest(splits),
            }
        )
    return records, manifest


def save_records(output, name, records, manifest):
    save_json(output / f"{name}_data_manifest.json", manifest)
    summaries = []
    for r in records:
        summary = {k: v for k, v in r.items() if k not in {"truth", "ids", "candidates"}}
        summary["metrics"] = {k: v["metrics"] for k, v in r["candidates"].items()}
        summaries.append(summary)
        with (output / f"{name}_{r['series']}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", "truth", "detector", "score", "threshold"])
            for key, candidate in r["candidates"].items():
                for wid, y, score in zip(r["ids"], r["truth"], candidate["scores"], strict=True):
                    writer.writerow([wid, y, key, score, candidate["threshold"]])
    save_json(
        output / f"{name}.json",
        {
            "series": summaries,
            "pooled": pool(records, records[0]["candidates"]),
            "environment": environment(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "evaluate"])
    parser.add_argument("--config", default="configs/revision.json")
    parser.add_argument("--data", default="data/nab")
    parser.add_argument("--confirmation-data", default="data/nab_ad")
    parser.add_argument("--output", default="results/revision_public")
    args = parser.parse_args()
    config = read_json(args.config)
    base = read_json(config["base_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if args.stage == "develop":
        if selection_path.exists():
            raise ValueError("Selection exists; do not silently retune")
        records, manifest = run_subset(Path(args.data), base, config, "validation")
        metrics = pool(records, records[0]["candidates"])
        selected = choose_public(metrics, config["public_validation_fpr_limit"])
        save_records(output, "development", records, manifest)
        save_json(
            selection_path,
            {
                "selected": selected,
                "validation_metrics": metrics,
                "config_digest": config_digest(config, base),
                "evaluation_used_for_selection": "validation",
            },
        )
        print("Frozen public selection:", selected, metrics[selected], flush=True)
    else:
        selection = read_json(selection_path)
        assert selection["config_digest"] == config_digest(config, base)
        if (output / "confirmation.json").exists():
            raise ValueError("Confirmation already evaluated")
        for name, root in [
            ("retrospective_aws", args.data),
            ("confirmation", args.confirmation_data),
        ]:
            records, manifest = run_subset(Path(root), base, config, "test", selection["selected"])
            save_records(output, name, records, manifest)


if __name__ == "__main__":
    main()
