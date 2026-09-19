import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from anomaly_study.data import synthetic_splits
from anomaly_study.features import (
    FeatureScaler,
    TemporalFeatures,
    balanced_weights,
    causal_context,
    contextual_splits,
    matched_width,
    rolling_innovation,
)
from anomaly_study.models import classifier, parameter_count
from anomaly_study.public_revision import choose_public
from anomaly_study.revision import anomaly_diagnosis


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/quick.json").read_text())


def test_development_has_no_test_and_confirmation_changes_only_test(config):
    first, context = contextual_splits(config, 17, 64)
    assert "test" not in first and "test" not in context
    second, _ = contextual_splits(config, 17, 64, 101)
    old = synthetic_splits(config, 17)
    for split in first:
        np.testing.assert_array_equal(first[split].x, second[split].x)
        np.testing.assert_array_equal(first[split].x, old[split].x)
    assert not np.array_equal(second["test"].x, old["test"].x)


def test_context_no_future_and_target_alignment():
    x = np.arange(100, dtype=np.float32)[:, None]
    a = causal_context(x, [16, 24, 80], 64)
    x[24:] = 999
    b = causal_context(x, [16, 24, 80], 64)
    np.testing.assert_array_equal(a[:2], b[:2])
    np.testing.assert_array_equal(a[0, -16:, 0], np.arange(16))
    assert a[0, 0, 0] == 0


def test_balancing_and_capacity():
    labels = np.array([0] * 20 + [1] * 2 + [2] * 8)
    weights = balanced_weights(labels, 3).numpy()
    np.testing.assert_allclose(weights * np.bincount(labels), [10, 10, 10])
    for dimension in [8, 73, 96, 104]:
        count = parameter_count(classifier(dimension, matched_width(dimension, 3, 6400), 3))
        assert abs(count - 6400) / 6400 < 0.01
    assert balanced_weights(np.array([0, 0]), 3).tolist() == [1.0, 0.0, 0.0]


def test_feature_models_are_train_only(config):
    splits, context = contextual_splits(config, 17, 64)
    model = TemporalFeatures().fit(splits["train"], 16, 6)
    features = model.transform(context["train"])
    assert features.shape == (len(splits["train"].y), 73)
    assert np.isfinite(features).all()
    with pytest.raises(ValueError):
        TemporalFeatures().fit(splits["validation"], 16, 6)
    with pytest.raises(ValueError):
        FeatureScaler().fit(features, "test")


def test_rolling_is_strictly_causal_and_constant_safe():
    x = np.arange(100, dtype=float)
    first = rolling_innovation(x, 16, 0.0, 0.1)
    x[51:] = 10000
    second = rolling_innovation(x, 16, 0.0, 0.1)
    np.testing.assert_array_equal(first[:51], second[:51])
    assert first[0] == 0
    assert np.isfinite(rolling_innovation(np.ones(100), 16, 1.0, 0.1)).all()


def test_rejection_does_not_remove_diagnostic_denominator():
    result = anomaly_diagnosis(np.array([1, 1, 2, 3]), np.array([0, 1, 2, 0]))
    assert result["support_per_class"] == [0, 2, 1, 1]
    assert result["recall_per_class"] == [None, 0.5, 1.0, 0.0]
    assert np.array(result["confusion_matrix"]).sum() == 4


def test_public_selector_respects_false_alarm_constraint():
    scores = {
        "high": {"f1": 0.9, "false_positive_rate": 0.5},
        "low": {"f1": 0.7, "false_positive_rate": 0.09},
    }
    assert choose_public(scores, 0.1) == "low"
    assert choose_public(scores, 0.01) == "low"


def test_publication_guard_rejects_staged_personal_notes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "TASK_STATE.md").write_text("private task notes", encoding="utf-8")
    subprocess.run(["git", "add", "TASK_STATE.md"], cwd=tmp_path, check=True)
    script = Path(__file__).parents[1] / "scripts/check_public_tree.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "TASK_STATE.md" in result.stdout
