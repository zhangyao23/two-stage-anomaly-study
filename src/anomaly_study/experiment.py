"""Reproducible training, independent classification and complete cascade evaluation."""

import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch

from .data import CLASSES, TrainStandardizer, calibrate, only_normal, synthetic_splits
from .metrics import cascade, detection, diagnosis
from .models import (
    Autoencoder,
    benchmark,
    classifier,
    parameter_count,
    representation,
    train_model,
)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "timing": "perf_counter_ns; 20 warmups, 100 repeats; median/p95; inference_mode; "
        "standardized in-memory CPU tensors; excludes windowing, scaling and I/O",
    }


def split_manifest(splits):
    return {
        name: {
            "windows": len(w.y),
            "class_counts": np.bincount(w.y).tolist(),
            "ids_sha256": hashlib.sha256("\n".join(w.ids).encode()).hexdigest(),
            "sequence_ids": sorted({s[0] for s in w.spans}),
            "spans": [
                [
                    seq,
                    min(s[1] for s in w.spans if s[0] == seq),
                    max(s[2] for s in w.spans if s[0] == seq),
                ]
                for seq in sorted({s[0] for s in w.spans})
            ],
        }
        for name, w in splits.items()
    }


def fit_detectors(splits, config, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    scaler = TrainStandardizer().fit(splits["train"])
    x = {name: torch.from_numpy(scaler.transform(w.x)) for name, w in splits.items()}
    train_normal = x["train"][splits["train"].y == 0]
    val_normal = x["validation"][splits["validation"].y == 0]
    ae = Autoencoder(x["train"].shape[1], config["ae_hidden"], config["latent"])
    training = train_model(
        ae, train_normal, train_normal, val_normal, val_normal, config, seed, reconstruction=True
    )
    cal = only_normal(splits["calibration"])
    # Public calibration segments may contain anomalies; only normal windows set thresholds.
    cal_x = x["calibration"][splits["calibration"].y == 0]
    with torch.inference_mode():
        cal_scores = {
            "autoencoder": ((ae(cal_x) - cal_x) ** 2).mean(1).numpy(),
            "max_abs_z": cal_x.abs().amax(1).numpy(),
        }
        test_scores = {
            "autoencoder": ((ae(x["test"]) - x["test"]) ** 2).mean(1).numpy(),
            "max_abs_z": x["test"].abs().amax(1).numpy(),
        }
    thresholds = {k: calibrate(v, cal, config["quantile"]) for k, v in cal_scores.items()}
    metrics = {k: detection(splits["test"].y, v, thresholds[k]) for k, v in test_scores.items()}
    batch = x["test"][: config["batch_size"]]
    timing = {
        "autoencoder_detector": benchmark(lambda: ((ae(batch) - batch) ** 2).mean(1), len(batch)),
        "max_abs_z_detector": benchmark(lambda: batch.abs().amax(1), len(batch)),
    }
    metadata = {
        "autoencoder_parameters": parameter_count(ae),
        "training": training,
        "normal_calibration_windows": len(cal.y),
        "quantile": config["quantile"],
        "threshold_fit_split": "calibration",
        "scaler_fit_split": "train_normal",
        "scaler_mean": scaler.mean.tolist(),
        "scaler_scale": scaler.scale.tolist(),
        "validation_detection": {},
    }
    with torch.inference_mode():
        v = x["validation"]
        val_scores = {
            "autoencoder": ((ae(v) - v) ** 2).mean(1).numpy(),
            "max_abs_z": v.abs().amax(1).numpy(),
        }
    metadata["validation_detection"] = {
        k: detection(splits["validation"].y, values, thresholds[k])
        for k, values in val_scores.items()
    }
    return ae, x, test_scores, thresholds, metrics, timing, metadata


def cascade_forward(ae, head, x, threshold, kind):
    """Real gated inference: classify only flagged windows and reuse the encoder result."""
    z = ae.encoder(x)
    score = ((ae.decoder(z) - x) ** 2).mean(1)
    mask = score > threshold
    result = torch.zeros(len(x), dtype=torch.long)
    if mask.any():
        features = representation(x[mask], z[mask], kind)
        result[mask] = head(features).argmax(1) + 1
    return result


def fit_heads(ae, x, splits, scores, thresholds, config, seed):
    with torch.inference_mode():
        # Clone outside inference tensors before training classifier weights.
        encoded = {name: ae.encoder(values).numpy().copy() for name, values in x.items()}
    z = {name: torch.from_numpy(values) for name, values in encoded.items()}
    tr_mask, va_mask, te_mask = [splits[k].y > 0 for k in ("train", "validation", "test")]
    heads, predictions, timings = {}, {}, {}
    for kind in ("raw", "encoded", "concat"):
        torch.manual_seed(seed + 100)
        features = {name: representation(x[name], z[name], kind) for name in x}
        head = classifier(features["train"].shape[1], config["head_hidden"], 3)
        info = train_model(
            head,
            features["train"][tr_mask],
            torch.from_numpy(splits["train"].y[tr_mask] - 1),
            features["validation"][va_mask],
            torch.from_numpy(splits["validation"].y[va_mask] - 1),
            config,
            seed + 100,
        )
        with torch.inference_mode():
            pred = head(features["test"]).argmax(1).numpy() + 1
        predictions[kind] = pred
        heads[kind] = {
            "input_dimension": features["train"].shape[1],
            "head_parameters": parameter_count(head),
            "cascade_parameters": parameter_count(ae) + parameter_count(head),
            "training": info,
            "independent_anomaly_diagnosis": diagnosis(
                splits["test"].y[te_mask], pred[te_mask], [1, 2, 3]
            ),
            "end_to_end": {
                det: diagnosis(
                    splits["test"].y, cascade(values, thresholds[det], pred), [0, 1, 2, 3]
                )
                for det, values in scores.items()
            },
        }
        batch = x["test"][: config["batch_size"]]
        feature_batch = features["test"][: len(batch)]
        timings[kind + "_head_only"] = benchmark(
            lambda head=head, features=feature_batch: head(features), len(batch)
        )
        timings[kind + "_ae_cascade"] = benchmark(
            lambda head=head, batch=batch, kind=kind: cascade_forward(
                ae, head, batch, thresholds["autoencoder"], kind
            ),
            len(batch),
        )
    single_result, single_prediction, single_timing = fit_single_stage(x, splits, config, seed)
    predictions["single_stage"] = single_prediction
    timings["single_stage"] = single_timing
    return heads, single_result, predictions, timings


def fit_single_stage(x, splits, config, seed):
    """Separate supervised baseline with four-class and binary reporting."""
    # Four-class supervised baseline uses the same train split, epochs, hidden size and optimizer.
    torch.manual_seed(seed + 100)
    single = classifier(x["train"].shape[1], config["head_hidden"], 4)
    info = train_model(
        single,
        x["train"],
        torch.from_numpy(splits["train"].y),
        x["validation"],
        torch.from_numpy(splits["validation"].y),
        config,
        seed + 100,
    )
    with torch.inference_mode():
        probs = single(x["test"]).softmax(1).numpy()
    pred = probs.argmax(1)
    # P(anomaly)>0.5 binary operating point is distinct from four-class argmax normal/anomaly.
    single_result = {
        "parameters": parameter_count(single),
        "training": info,
        "diagnosis": diagnosis(splits["test"].y, pred, [0, 1, 2, 3]),
        "detection_p_anomaly_gt_half": detection(splits["test"].y, 1 - probs[:, 0], 0.5),
        "detection_argmax": detection(splits["test"].y, (pred > 0).astype(float), 0.5),
    }
    single_result["detection_argmax"]["pr_auc_ap"] = None  # Hard decisions are not ranking scores.
    batch = x["test"][: config["batch_size"]]
    timing = benchmark(lambda: single(batch).argmax(1), len(batch))
    return single_result, pred, timing


def write_predictions(path, test, scores, thresholds, predictions):
    columns = [
        "window_id",
        "truth",
        "ae_score",
        "stat_score",
        "ae_flag",
        "stat_flag",
        "raw",
        "encoded",
        "concat",
        "single_stage",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, wid in enumerate(test.ids):
            writer.writerow(
                [
                    wid,
                    test.y[i],
                    scores["autoencoder"][i],
                    scores["max_abs_z"][i],
                    int(scores["autoencoder"][i] > thresholds["autoencoder"]),
                    int(scores["max_abs_z"][i] > thresholds["max_abs_z"]),
                    *[predictions[k][i] for k in columns[6:]],
                ]
            )


def failure_cases(test, scores, threshold, predictions, config):
    """First-in-time representative errors by category, never maximum-impact cherry-picks."""
    ae = scores["autoencoder"]
    masks = {f"missed_{CLASSES[k]}": (test.y == k) & (ae <= threshold) for k in (1, 2, 3)}
    masks["false_alarm"] = (test.y == 0) & (ae > threshold)
    masks["encoded_wrong_raw_correct"] = (
        (test.y > 0) & (predictions["encoded"] != test.y) & (predictions["raw"] == test.y)
    )
    masks["raw_wrong_encoded_correct"] = (
        (test.y > 0) & (predictions["raw"] != test.y) & (predictions["encoded"] == test.y)
    )
    cases = []
    for category, mask in masks.items():
        if not mask.any():
            continue
        i = int(np.flatnonzero(mask)[0])
        cases.append(
            {
                "category": category,
                "window_id": test.ids[i],
                "truth": int(test.y[i]),
                "ae_score": float(ae[i]),
                "ae_threshold": threshold,
                "predictions": {k: int(v[i]) for k, v in predictions.items()},
                "series": test.x[i].reshape(config["window"], config["channels"]).tolist(),
                "selection": "first matching test window in stored order",
            }
        )
    return cases


def flatten_metrics(result):
    flat = {}
    for detector, metrics in result["detection"].items():
        for metric in ("precision", "recall", "f1", "false_positive_rate", "pr_auc_ap"):
            flat[f"detection/{detector}/{metric}"] = metrics[metric]
    for kind, metrics in result["heads"].items():
        flat[f"diagnosis/{kind}/macro_f1"] = metrics["independent_anomaly_diagnosis"]["macro_f1"]
        for detector, end in metrics["end_to_end"].items():
            flat[f"cascade/{detector}/{kind}/macro_f1"] = end["macro_f1"]
            for i, recall in enumerate(end["recall_per_class"]):
                flat[f"cascade/{detector}/{kind}/recall_{CLASSES[i]}"] = recall
    flat["single_stage/macro_f1"] = result["single_stage"]["diagnosis"]["macro_f1"]
    return flat


def run_synthetic(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    results = []
    for seed in config["seeds"]:
        print(f"Synthetic seed {seed}", flush=True)
        splits = synthetic_splits(config, seed)
        ae, x, scores, thresholds, det, times, meta = fit_detectors(splits, config, seed)
        heads, single, predictions, head_times = fit_heads(
            ae, x, splits, scores, thresholds, config, seed
        )
        result = {
            "seed": seed,
            "detection": det,
            "heads": heads,
            "single_stage": single,
            "timing": times | head_times,
            "detector_metadata": meta,
            "splits": split_manifest(splits),
            "environment": environment(),
        }
        results.append(result)
        save_json(output / f"seed_{seed}.json", result)
        write_predictions(
            output / f"predictions_{seed}.csv", splits["test"], scores, thresholds, predictions
        )
        save_json(
            output / f"failures_{seed}.json",
            failure_cases(splits["test"], scores, thresholds["autoencoder"], predictions, config),
        )
    flat = [flatten_metrics(r) for r in results]
    summary = {
        key: {
            "mean": float(np.mean([f[key] for f in flat])),
            "std": float(np.std([f[key] for f in flat], ddof=1)) if len(flat) > 1 else None,
        }
        for key in flat[0]
    }
    save_json(output / "summary.json", summary)
    from .plots import synthetic_plots

    synthetic_plots(output, results, summary)
    return results
