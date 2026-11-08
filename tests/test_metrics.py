"""Tests for evaluation metrics (no GPU or model needed)."""

import numpy as np

from src.evaluation.metrics import (binary_metrics, expected_calibration_error,
                                    multiclass_metrics, softmax)


def _onehot_probs(labels, n=10, conf=0.9):
    probs = np.full((len(labels), n), (1 - conf) / (n - 1))
    probs[np.arange(len(labels)), labels] = conf
    return probs


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(5, 10)) * 10)
    assert np.allclose(p.sum(1), 1)


def test_perfect_predictions():
    labels = np.arange(10).repeat(5)
    m = multiclass_metrics(_onehot_probs(labels), labels)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_ece_zero_when_confidence_matches_accuracy():
    labels = np.arange(10).repeat(10)
    probs = _onehot_probs(labels, conf=1.0)
    assert expected_calibration_error(probs, labels) < 1e-9


def test_ece_detects_overconfidence():
    labels = np.zeros(100, dtype=int)
    preds = np.where(np.arange(100) < 50, 0, 1)        # 50% accurate
    probs = _onehot_probs(preds, conf=0.99)             # but 99% confident
    assert expected_calibration_error(probs, labels) > 0.4


def test_binary_safe_vs_distracted():
    labels = np.array([0, 0, 3, 5])                     # 2 safe, 2 distracted
    preds = np.array([0, 2, 3, 5])                      # one safe frame flagged
    m = binary_metrics(_onehot_probs(preds), labels)
    assert m["bin_recall"] == 1.0
    assert m["false_alarm_rate"] == 0.5
