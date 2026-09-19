"""Train-fitted temporal features and causal context; no generator metadata as input."""

import numpy as np
import torch

from .data import SPLITS, combine, synthetic_sequence, window_sequence


def causal_context(x, ends, history):
    padded = np.pad(x, ((history - 1, 0), (0, 0)), mode="edge")
    return np.stack([padded[end - 1 : end - 1 + history] for end in ends]).astype(np.float32)


def contextual_splits(config, train_seed, history, test_seed=None):
    """Development call never generates test sequences. Confirmation replaces only test RNG."""
    splits, contexts = {}, {}
    children = np.random.SeedSequence(train_seed).spawn(4)
    if test_seed is not None:
        children[3] = np.random.SeedSequence(test_seed).spawn(4)[3]
    for index, split in enumerate(SPLITS):
        if split == "test" and test_seed is None:
            continue
        rng = np.random.default_rng(children[index])
        parts, context = [], []
        for seq in range(config["sequences"][split]):
            x, y = synthetic_sequence(
                rng, config["length"], config["channels"], split == "calibration"
            )
            windows = window_sequence(
                x, y, config["window"], config["stride"], f"{split}-{seq}", split
            )
            parts.append(windows)
            context.append(causal_context(x, [s[2] for s in windows.spans], history))
        splits[split] = combine(parts, split)
        contexts[split] = np.concatenate(context)
    return splits, contexts


class FeatureScaler:
    def fit(self, x, split):
        if split != "train" or not len(x):
            raise ValueError("Feature scaling requires nonempty training features")
        self.mean = x.mean(0)
        std = x.std(0)
        self.scale = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.scale).astype(np.float32)


class TemporalFeatures:
    def fit(self, training, width, channels):
        if training.split != "train":
            raise ValueError("Normal channel model must be fitted on training data")
        normal = training.x[training.y == 0].reshape(-1, channels)
        self.mean = normal.mean(0)
        _, _, vectors = np.linalg.svd(normal - self.mean, full_matrices=False)
        self.axis = vectors[0]
        self.width = width
        return self

    def transform(self, contexts):
        centered = contexts - self.mean
        common = centered @ self.axis
        residual = centered - common[:, :, None] * self.axis
        # Choose channel by observed target residual energy, never by injected channel metadata.
        energy = (residual[:, -self.width :] ** 2).mean(1)
        channel = energy.argmax(1)
        trace = residual[np.arange(len(residual)), :, channel]
        target = trace[:, -self.width :]
        peak = np.abs(target).argmax(1)
        sign = np.sign(target[np.arange(len(trace)), peak])
        trace = trace * np.where(sign == 0, 1.0, sign)[:, None]
        target = trace[:, -self.width :]
        diff = np.diff(trace, axis=1)
        stats = np.column_stack(
            [
                target.mean(1),
                target.std(1),
                target.max(1),
                target.min(1),
                np.abs(np.diff(target, axis=1)).max(1),
                np.abs(diff).max(1),
                (target**2).mean(1),
                np.quantile(target, 0.25, axis=1),
                np.quantile(target, 0.75, axis=1),
            ]
        )
        return np.concatenate([trace, stats], axis=1).astype(np.float32)


def matched_width(dimension, classes, budget):
    return max(1, round((budget - classes) / (dimension + classes + 1)))


def balanced_weights(y, classes):
    counts = np.bincount(y, minlength=classes)
    weights = np.zeros(classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return torch.from_numpy(weights)


def rolling_innovation(values, lookback, initial, scale_floor):
    """Score each observation from strictly earlier observations, never labels or future values."""
    if lookback < 2 or scale_floor <= 0:
        raise ValueError("Invalid rolling scale/window")
    values = np.asarray(values, dtype=float).reshape(-1)
    padded = np.concatenate([np.full(lookback, initial), values])
    past = np.lib.stride_tricks.sliding_window_view(padded, lookback)[: len(values)]
    median = np.median(past, axis=1)
    mad = np.median(np.abs(past - median[:, None]), axis=1)
    scale = np.maximum(1.4826 * mad, scale_floor)
    return np.abs(values - median) / scale
