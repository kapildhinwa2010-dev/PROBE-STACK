import numpy as np
import pytest

from probe_stack.stack import probe_report, probe_score, train_probe

N, D = 40, 6
LABELS = np.array([0.0] * (N // 2) + [1.0] * (N // 2))


def constant_layer(seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=D) * 0.5
    return np.tile(base, (N, 1))


def separable_layer(seed):
    rng = np.random.default_rng(seed)
    direction = np.zeros(D)
    direction[0] = 2.0
    noise = rng.normal(size=(N, D)) * 0.1
    return np.outer(LABELS, direction) + noise


def test_constant_layer_not_separable():
    acc = probe_score(constant_layer(0), LABELS)
    assert acc == pytest.approx(0.5)


def test_separable_layer_read_out():
    acc = probe_score(separable_layer(1), LABELS)
    assert acc >= 0.95


def test_report_finds_best_layer():
    layers = [constant_layer(0), constant_layer(1), separable_layer(2), constant_layer(3)]
    r = probe_report(layers, LABELS)
    assert len(r["accuracies"]) == 4
    assert r["best_layer"] == 2
    assert r["best_accuracy"] >= 0.95
    assert r["accuracies"][0] == pytest.approx(0.5)


def test_probe_is_deterministic():
    p1 = train_probe(separable_layer(2), LABELS, seed=5)
    p2 = train_probe(separable_layer(2), LABELS, seed=5)
    assert np.array_equal(p1["weights"], p2["weights"])
    assert p1["bias"] == p2["bias"]


def test_errors():
    with pytest.raises(ValueError):
        probe_score(np.zeros((4, 2, 2)), LABELS[:4])
    with pytest.raises(ValueError):
        probe_score(np.zeros((4, 2)), np.array([0.0, 1.0, 2.0, 0.0]))
    with pytest.raises(ValueError):
        probe_score(np.zeros((0, 2)), np.zeros((0,)))
    with pytest.raises(ValueError):
        probe_report([], LABELS)