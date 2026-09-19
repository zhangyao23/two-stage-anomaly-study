"""Separate development selection and untouched synthetic confirmation commands."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .data import TrainStandardizer
from .experiment import environment, fit_detectors, fit_heads, save_json, split_manifest
from .features import (
    FeatureScaler,
    TemporalFeatures,
    balanced_weights,
    contextual_splits,
    matched_width,
)
from .metrics import cascade, diagnosis
from .models import benchmark, classifier, parameter_count, train_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_digest(config, base):
    return hashlib.sha256(json.dumps([config, base], sort_keys=True).encode()).hexdigest()


def features_for(ae, scaler, temporal, splits, contexts):
    x = {k: scaler.transform(v.x) for k, v in splits.items()}
    with torch.inference_mode():
        z = {k: ae.encoder(torch.from_numpy(v)).numpy().copy() for k, v in x.items()}
    return {
        "raw": x,
        "encoded": z,
        "concat": {k: np.concatenate([x[k], z[k]], axis=1) for k in x},
        "temporal": {k: temporal.transform(contexts[k]) for k in x},
    }


def anomaly_diagnosis(y, pred):
    # Include rejection-to-normal column; do not silently discard rejected abnormal samples.
    result = diagnosis(y, pred, [0, 1, 2, 3])
    result["macro_f1"] = float(
        f1_score(y, pred, labels=[1, 2, 3], average="macro", zero_division=0)
    )
    result["macro_labels"] = [1, 2, 3]
    return result


def score_predictions(y, pred, detector_score, threshold):
    full = cascade(detector_score, threshold, pred)
    return {
        "independent": anomaly_diagnosis(y[y > 0], pred[y > 0]),
        "cascade": diagnosis(y, full, [0, 1, 2, 3]),
        "standalone": diagnosis(y, pred, [0, 1, 2, 3]),
    }, full


def fit_revised_heads(features, splits, config, base, seed):
    models, metadata = {}, {}
    train_config = base | {"epochs": config["head_epochs"]}
    # Single-stage controls use exactly the corresponding revised features.
    for name in [*config["representations"], "single_raw", "single_temporal"]:
        kind = name.removeprefix("single_")
        classes = 4 if name.startswith("single_") else 3
        keep = {k: (v.y >= 0 if classes == 4 else v.y > 0) for k, v in splits.items()}
        fs = FeatureScaler().fit(features[kind]["train"], "train")
        data = {k: torch.from_numpy(fs.transform(v)) for k, v in features[kind].items()}
        width = matched_width(data["train"].shape[1], classes, config["head_parameter_budget"])
        torch.manual_seed(seed + 100)
        head = classifier(data["train"].shape[1], width, classes)
        offset = 0 if classes == 4 else 1
        y = splits["train"].y[keep["train"]] - offset
        info = train_model(
            head,
            data["train"][keep["train"]],
            torch.from_numpy(y),
            data["validation"][keep["validation"]],
            torch.from_numpy(splits["validation"].y[keep["validation"]] - offset),
            train_config,
            seed + 100,
            class_weights=balanced_weights(y, classes),
        )
        models[name] = {
            "weights": head.state_dict(),
            "mean": torch.from_numpy(fs.mean),
            "scale": torch.from_numpy(fs.scale),
            "width": width,
            "dimension": data["train"].shape[1],
            "classes": classes,
        }
        with torch.inference_mode():
            pred = head(data["validation"]).argmax(1).numpy() + offset
        metadata[name] = {
            "parameters": parameter_count(head),
            "training": info,
            "validation_prediction": pred.tolist(),
            "input_dimension": data["train"].shape[1],
            "class_weights": balanced_weights(y, classes).tolist(),
        }
    return models, metadata


def develop(config, base, output, checkpoints):
    if (output / "selection.json").exists():
        raise ValueError("Selection already exists; use a new output directory for a new protocol")
    records = []
    checkpoints.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    for seed in config["development_seeds"]:
        print(f"Develop synthetic {seed} (no test generated)", flush=True)
        splits, context = contextual_splits(base, seed, config["history"])
        # Existing v1 fit routine requires a scoring slot: only validation is supplied there.
        scoring = splits | {"test": splits["validation"]}
        ae, _, scores, thresholds, _, _, meta = fit_detectors(scoring, base, seed)
        scaler = TrainStandardizer().fit(splits["train"])
        temporal = TemporalFeatures().fit(splits["train"], base["window"], base["channels"])
        features = features_for(ae, scaler, temporal, splits, context)
        models, heads = fit_revised_heads(features, splits, config, base, seed)
        for h in heads.values():
            h["validation"], _ = score_predictions(
                splits["validation"].y,
                np.array(h.pop("validation_prediction")),
                scores["autoencoder"],
                thresholds["autoencoder"],
            )
        record = {
            "seed": seed,
            "heads": heads,
            "detector": meta,
            "threshold": thresholds["autoencoder"],
            "splits": split_manifest(splits),
        }
        records.append(record)
        save_json(output / f"development_{seed}.json", record)
        torch.save(
            {"ae": ae.state_dict(), "heads": models, "config_digest": config_digest(config, base)},
            checkpoints / f"seed_{seed}.pt",
        )
    candidates = [*config["representations"], "single_temporal"]
    means = {
        name: float(
            np.mean([r["heads"][name]["validation"]["cascade"]["macro_f1"] for r in records])
        )
        for name in candidates
    }
    selected = max(candidates, key=lambda k: means[k])
    save_json(
        output / "selection.json",
        {
            "selected_cascade_head": selected,
            "validation_macro_f1": means,
            "config_digest": config_digest(config, base),
            "test_generated_during_selection": False,
        },
    )
    print("Frozen selection:", selected, means, flush=True)


def load_head(info):
    head = classifier(info["dimension"], info["width"], info["classes"])
    head.load_state_dict(info["weights"])
    return head.eval()


def evaluate(config, base, output, checkpoints):
    selection = read_json(output / "selection.json")
    assert selection["config_digest"] == config_digest(config, base)
    if (output / "confirmation.json").exists():
        raise ValueError("Confirmation already evaluated; keep results immutable")
    results = []
    for seed, test_seed in zip(
        config["development_seeds"], config["confirmation_seeds"], strict=True
    ):
        print(f"Confirm train {seed}, fresh test {test_seed}", flush=True)
        splits, contexts = contextual_splits(base, seed, config["history"], test_seed)
        ae, x, scores, thresholds, det, _, meta = fit_detectors(splits, base, seed)
        checkpoint = torch.load(checkpoints / f"seed_{seed}.pt", weights_only=True)
        assert checkpoint["config_digest"] == selection["config_digest"]
        for key, value in ae.state_dict().items():
            torch.testing.assert_close(value, checkpoint["ae"][key], rtol=0, atol=0)
        v1_heads, v1_single, _, _ = fit_heads(ae, x, splits, scores, thresholds, base, seed)
        scaler = TrainStandardizer().fit(splits["train"])
        temporal = TemporalFeatures().fit(splits["train"], base["window"], base["channels"])
        test = splits["test"]
        features = features_for(ae, scaler, temporal, {"test": test}, {"test": contexts["test"]})
        results_heads, predictions = {}, {}
        for name, info in checkpoint["heads"].items():
            head = load_head(info)
            values = torch.from_numpy(features[name.removeprefix("single_")]["test"])
            normalized = (values - info["mean"]) / info["scale"]
            with torch.inference_mode():
                pred = head(normalized).argmax(1).numpy() + (0 if info["classes"] == 4 else 1)
            result, full = score_predictions(
                test.y, pred, scores["autoencoder"], thresholds["autoencoder"]
            )
            result["parameters"] = parameter_count(head)
            batch = normalized[: base["batch_size"]]
            result["head_timing"] = benchmark(
                lambda head=head, batch=batch: head(batch), len(batch)
            )
            results_heads[name] = result
            predictions[name] = pred
            predictions[name + "_cascade"] = full
        result = {
            "train_seed": seed,
            "test_seed": test_seed,
            "detection": det,
            "revised": results_heads,
            "v1_heads": v1_heads,
            "v1_single": v1_single,
            "ae_parameters": meta["autoencoder_parameters"],
            "splits": split_manifest(splits),
        }
        results.append(result)
        with (output / f"predictions_{test_seed}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", "truth", "ae_score", "threshold", *predictions])
            for i, wid in enumerate(test.ids):
                writer.writerow(
                    [
                        wid,
                        test.y[i],
                        scores["autoencoder"][i],
                        thresholds["autoencoder"],
                        *[p[i] for p in predictions.values()],
                    ]
                )
        save_json(output / f"confirmation_{test_seed}.json", result)
    save_json(
        output / "confirmation.json",
        {"selection": selection, "runs": results, "environment": environment()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "evaluate"])
    parser.add_argument("--config", default="configs/revision.json")
    parser.add_argument("--output", default="results/revision")
    parser.add_argument("--checkpoints", default="checkpoints/revision")
    args = parser.parse_args()
    config = read_json(args.config)
    base = read_json(config["base_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    function = develop if args.stage == "develop" else evaluate
    function(config, base, output, Path(args.checkpoints))


if __name__ == "__main__":
    main()
