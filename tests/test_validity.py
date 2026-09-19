import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anomaly_study.data import (
    TrainStandardizer,
    Windows,
    calibrate,
    synthetic_splits,
    window_sequence,
)
from anomaly_study.experiment import cascade_forward, fit_detectors
from anomaly_study.metrics import cascade, detection, diagnosis
from anomaly_study.models import Autoencoder, classifier, representation
from anomaly_study.public_data import chronological_splits, read_series


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/quick.json").read_text())


def test_sequence_split_and_reproducibility(config):
    splits = synthetic_splits(config, 17)
    again = synthetic_splits(config, 17)
    seen = set()
    for name, windows in splits.items():
        sequences = {s[0] for s in windows.spans}
        assert not sequences & seen
        seen |= sequences
        np.testing.assert_array_equal(windows.x, again[name].x)
        assert all(end - start == config["window"] for _, start, end in windows.spans)
    assert set(splits["train"].y) == {0, 1, 2, 3}
    assert set(splits["calibration"].y) == {0}


def test_temporal_split_before_overlapping_windows():
    x = np.arange(1000, dtype=np.float32)[:, None]
    splits = chronological_splits(x, np.zeros(1000, dtype=int), "one", 16, 8)
    seen = set()
    for windows in splits.values():
        covered = {i for _, start, end in windows.spans for i in range(start, end)}
        assert not seen & covered
        seen |= covered
        assert all(
            np.all(windows.x[j] == np.arange(start, end))
            for j, (_, start, end) in enumerate(windows.spans)
        )


def test_threshold_provenance_and_normal_only(config):
    splits = synthetic_splits(config, 17)
    cal = splits["calibration"]
    scores = np.arange(len(cal.y), dtype=float)
    assert calibrate(scores, cal, 0.95) == np.quantile(scores, 0.95, method="higher")
    for name in ("train", "validation", "test"):
        with pytest.raises(ValueError):
            calibrate(np.ones(len(splits[name].y)), splits[name], 0.95)
    cal.y[0] = 1
    with pytest.raises(ValueError):
        calibrate(scores, cal, 0.95)


def test_standardizer_never_fits_test_or_anomalies(config):
    splits = synthetic_splits(config, 17)
    train = splits["train"]
    first = TrainStandardizer().fit(train)
    train.x[train.y > 0] = 1e8
    second = TrainStandardizer().fit(train)
    np.testing.assert_array_equal(first.mean, second.mean)
    with pytest.raises(ValueError):
        TrainStandardizer().fit(splits["test"])


def test_input_dimensions_and_real_gate():
    torch.manual_seed(7)
    ae = Autoencoder(96, 64, 8)
    x = torch.randn(12, 96)
    z = ae.encoder(x)
    for name, dimension in [("raw", 96), ("encoded", 8), ("concat", 104)]:
        assert representation(x, z, name).shape == (12, dimension)
        head = classifier(dimension, 32, 3)
        with torch.inference_mode():
            score = ((ae(x) - x) ** 2).mean(1).numpy()
            pred = head(representation(x, z, name)).argmax(1).numpy() + 1
            expected = cascade(score, 1.0, pred)
            actual = cascade_forward(ae, head, x, 1.0, name).numpy()
        np.testing.assert_array_equal(expected, actual)


def test_missed_anomalies_and_false_alarms_count_in_cascade():
    truth = np.array([0, 1, 2, 3])
    pred = cascade([3, 0, 3, 0], 1, [2, 1, 2, 3])
    result = diagnosis(truth, pred, [0, 1, 2, 3])
    assert sum(map(sum, result["confusion_matrix"])) == 4
    assert result["recall_per_class"] == [0, 0, 1, 0]
    assert result["confusion_matrix"][1][0] == 1
    assert result["confusion_matrix"][0][2] == 1


def test_empty_predictions_missing_classes_and_ties():
    result = detection([0, 1, 0], [0, 0, 0], 0)
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["false_positive_rate"] == 0
    assert detection([0, 0], [0, 1], 0.5)["pr_auc_ap"] is None
    assert detection([1, 1], [0, 1], 0.5)["false_positive_rate"] is None
    missing = diagnosis([0, 0], [0, 1], [0, 1, 2, 3])
    assert missing["recall_per_class"] == [0.5, None, None, None]
    assert np.asarray(missing["confusion_matrix"]).shape == (4, 4)
    json.dumps(missing, allow_nan=False)
    empty = diagnosis([], [], [1, 2, 3])
    assert empty["macro_f1"] == 0
    assert empty["recall_per_class"] == [None, None, None]
    json.dumps(empty, allow_nan=False)


def test_public_label_intervals_and_no_mixed_types(tmp_path):
    path = tmp_path / "series.csv"
    path.write_text(
        "timestamp,value\n2020-01-01 00:00:00,1\n2020-01-01 00:01:00,2\n2020-01-01 00:02:00,3\n"
    )
    x, y, quality = read_series(path, [["2020-01-01 00:01:00", "2020-01-01 00:02:00"]])
    assert quality["duplicate_rows_aggregated"] == 0
    np.testing.assert_array_equal(y, [0, 1, 1])
    assert window_sequence(x, y, 2, 1, "x", "test").y.tolist() == [1, 1]
    with pytest.raises(ValueError):
        window_sequence(x, np.array([0, 1, 2]), 3, 1, "x", "test")


def test_test_labels_cannot_change_training_or_threshold(config):
    config = config | {"epochs": 1}
    splits = synthetic_splits(config, 17)
    ae1, _, _, thresholds1, _, _, _ = fit_detectors(splits, config, 17)
    splits["test"].y[:] = 0
    ae2, _, _, thresholds2, _, _, _ = fit_detectors(splits, config, 17)
    assert thresholds1 == thresholds2
    for first, second in zip(ae1.parameters(), ae2.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_constant_training_dimension():
    w = Windows(np.ones((5, 2), dtype=np.float32), np.zeros(5, dtype=int), [], [], "train")
    scaler = TrainStandardizer().fit(w)
    assert np.isfinite(scaler.transform(w.x)).all()


def test_public_duplicate_timestamp_aggregation(tmp_path):
    path = tmp_path / "duplicates.csv"
    path.write_text(
        "timestamp,value\n2020-01-01 00:00:00,1\n2020-01-01 00:00:00,3\n2020-01-01 00:01:00,8\n"
    )
    x, y, quality = read_series(path, [])
    np.testing.assert_array_equal(x[:, 0], [2, 8])
    assert quality["duplicate_rows_aggregated"] == 1
    assert y.tolist() == [0, 0]
