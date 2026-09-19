"""Sequence-first splits, synthetic morphology labels and auditable windows."""

from dataclasses import dataclass

import numpy as np

CLASSES = ["normal", "spike", "shift", "drift"]
SPLITS = ("train", "calibration", "validation", "test")


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    ids: list[str]
    spans: list[tuple[str, int, int]]
    split: str


def window_sequence(x, y, width, stride, sequence_id, split, offset=0):
    """Inclusive-any anomaly labels; reject windows containing multiple anomaly types."""
    if width < 1 or stride < 1 or len(x) != len(y) or len(x) < width:
        raise ValueError("Invalid window dimensions or sequence length")
    features, labels, ids, spans = [], [], [], []
    for start in range(0, len(x) - width + 1, stride):
        end = start + width
        types = np.unique(y[start:end])
        anomalous = types[types != 0]
        if len(anomalous) > 1:
            raise ValueError("A window contains multiple anomaly types")
        features.append(x[start:end].reshape(-1))
        labels.append(int(anomalous[0]) if len(anomalous) else 0)
        ids.append(f"{sequence_id}:{offset + start}:{offset + end}")
        spans.append((sequence_id, offset + start, offset + end))
    return Windows(np.asarray(features, dtype=np.float32), np.asarray(labels), ids, spans, split)


def combine(parts, split):
    return Windows(
        np.concatenate([p.x for p in parts]),
        np.concatenate([p.y for p in parts]),
        [i for p in parts for i in p.ids],
        [s for p in parts for s in p.spans],
        split,
    )


def synthetic_sequence(rng, length, channels, normal_only=False):
    """Generic correlated metrics, not network fault causes or company measurements."""
    if length < 320 or channels < 2:
        raise ValueError("Generator requires length >= 320 and channels >= 2")
    t = np.arange(length)
    phase = rng.uniform(0, 2 * np.pi)
    common = np.sin(t / 17 + phase) + 0.3 * np.sin(t / 5 + phase)
    noise = rng.normal(0, 0.13, (length, channels))
    for i in range(1, length):
        noise[i] += 0.65 * noise[i - 1]
    x = common[:, None] * np.linspace(0.5, 1.1, channels) + noise
    x += rng.normal(0, 0.12, channels)
    y = np.zeros(length, dtype=int)
    if not normal_only:
        # Each type is equally likely to affect each channel/sign. No type-specific channel.
        for kind, anchor in zip(rng.permutation([1, 2, 3]), [48, 144, 240], strict=True):
            start = anchor + int(rng.integers(-8, 9))
            channel = int(rng.integers(channels))
            sign = rng.choice([-1, 1])
            duration = {1: 3, 2: 32, 3: 48}[int(kind)]
            amplitude = rng.uniform(0.6, 2.8)
            if kind == 1:
                shape = np.array([0.4, 1.0, 0.4]) * amplitude * 2
            elif kind == 2:
                shape = np.ones(duration) * amplitude
            else:
                shape = np.linspace(0.05, amplitude, duration)
            x[start : start + duration, channel] += sign * shape
            y[start : start + duration] = kind
    return x.astype(np.float32), y


def synthetic_splits(config, seed):
    result = {}
    for split, child in zip(SPLITS, np.random.SeedSequence(seed).spawn(4), strict=True):
        rng = np.random.default_rng(child)
        parts = []
        for seq in range(config["sequences"][split]):
            x, y = synthetic_sequence(
                rng, config["length"], config["channels"], split == "calibration"
            )
            parts.append(
                window_sequence(x, y, config["window"], config["stride"], f"{split}-{seq}", split)
            )
        result[split] = combine(parts, split)
    return result


class TrainStandardizer:
    """Fitted only on normal training windows; constant dimensions have unit scale."""

    def fit(self, train):
        if train.split != "train":
            raise ValueError("Standardization requires training data")
        normal = train.x[train.y == 0]
        if len(normal) == 0:
            raise ValueError("No normal training windows")
        self.mean = normal.mean(0)
        std = normal.std(0)
        self.scale = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.scale).astype(np.float32)


def calibrate(scores, calibration, quantile):
    if calibration.split != "calibration" or np.any(calibration.y != 0):
        raise ValueError("Threshold requires independent normal calibration windows")
    if len(scores) != len(calibration.y) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    if not 0 < quantile < 1:
        raise ValueError("Quantile must lie in (0, 1)")
    return float(np.quantile(scores, quantile, method="higher"))


def only_normal(windows):
    mask = windows.y == 0
    return Windows(
        windows.x[mask],
        windows.y[mask],
        [i for i, keep in zip(windows.ids, mask, strict=True) if keep],
        [s for s, keep in zip(windows.spans, mask, strict=True) if keep],
        windows.split,
    )
