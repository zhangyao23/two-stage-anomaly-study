"""Benchmark actual revised inference including feature computation and gated rejection."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .data import TrainStandardizer
from .experiment import environment, save_json
from .features import TemporalFeatures, contextual_splits
from .models import Autoencoder, benchmark
from .revision import config_digest, load_head, read_json


def predict(raw, context, ae, head, info, scaler, temporal, threshold, gated):
    if gated:
        x = torch.from_numpy(scaler.transform(raw))
        z = ae.encoder(x)
        score = ((ae.decoder(z) - x) ** 2).mean(1)
        mask = (score > threshold).numpy()
    else:
        mask = np.ones(len(raw), dtype=bool)
    prediction = torch.zeros(len(raw), dtype=torch.long)
    if mask.any():
        features = torch.from_numpy(temporal.transform(context[mask]))
        features = (features - info["mean"]) / info["scale"]
        prediction[mask] = head(features).argmax(1)
    return prediction


def run(config, base, output, checkpoints):
    selection = read_json(output / "selection.json")
    if selection["selected_cascade_head"] != "single_temporal":
        raise ValueError(
            "This measurement is specifically for the declared temporal rejection model"
        )
    records = []
    torch.set_num_threads(base["torch_threads"])
    for seed, test_seed in zip(
        config["development_seeds"], config["confirmation_seeds"], strict=True
    ):
        splits, context = contextual_splits(base, seed, config["history"], test_seed)
        checkpoint = torch.load(checkpoints / f"seed_{seed}.pt", weights_only=True)
        assert checkpoint["config_digest"] == config_digest(config, base)
        scaler = TrainStandardizer().fit(splits["train"])
        temporal = TemporalFeatures().fit(splits["train"], base["window"], base["channels"])
        ae = Autoencoder(base["window"] * base["channels"], base["ae_hidden"], base["latent"])
        ae.load_state_dict(checkpoint["ae"])
        ae.eval()
        info = checkpoint["heads"]["single_temporal"]
        head = load_head(info)
        threshold = read_json(output / f"development_{seed}.json")["threshold"]
        expected = np.genfromtxt(
            output / f"predictions_{test_seed}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        for gated in [False, True]:
            fn = partial(
                predict,
                ae=ae,
                head=head,
                info=info,
                scaler=scaler,
                temporal=temporal,
                threshold=threshold,
                gated=gated,
            )
            with torch.inference_mode():
                actual = fn(splits["test"].x, context["test"]).numpy()
            column = "single_temporal_cascade" if gated else "single_temporal"
            np.testing.assert_array_equal(actual, expected[column])
            for size in [1, 128]:
                call = partial(fn, splits["test"].x[:size], context["test"][:size])
                result = benchmark(call, size)
                result.update(
                    {
                        "train_seed": seed,
                        "test_seed": test_seed,
                        "gated": gated,
                        "includes_feature_computation": True,
                        "flagged_windows": int(np.sum(expected["ae_score"][:size] > threshold)),
                    }
                )
                records.append(result)
    save_json(
        output / "pipeline_timing.json",
        {
            "records": records,
            "environment": environment(),
            "scope": "Includes standardization, PCA residual features, feature scaling, MLP and AE gate when enabled; excludes initial context assembly and I/O",
        },
    )
    print(
        "Pipeline predictions match saved confirmation for all 3 seeds; batch 1/128 timing saved."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revision.json")
    parser.add_argument("--output", default="results/revision")
    parser.add_argument("--checkpoints", default="checkpoints/revision")
    args = parser.parse_args()
    config = read_json(args.config)
    run(config, read_json(config["base_config"]), Path(args.output), Path(args.checkpoints))


if __name__ == "__main__":
    main()
