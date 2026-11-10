"""Train tiny linear probes on per-layer hidden states.

PROBE-STACK trains a plain logistic probe by gradient descent on
the hidden states of every layer, then reports per-layer train
accuracy and the layer where the property is easiest to read out.
Everything is seeded, so results are reproducible. Pure NumPy.
"""

import numpy as np


def _check(states, labels):
    x = np.asarray(states, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 2:
        raise ValueError("states must have shape (N, D)")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must have shape (N,) aligned with states")
    if x.shape[0] == 0:
        raise ValueError("states must contain at least one sample")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("labels must be binary 0 or 1")
    return x, y


def train_probe(states, labels, epochs=300, lr=0.5, seed=0):
    """Train a logistic probe by full-batch gradient descent."""
    x, y = _check(states, labels)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=x.shape[1]) * 0.01
    b = 0.0
    n = max(len(y), 1)
    for _ in range(epochs):
        z = x @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) / n
        w = w - lr * (x.T @ g)
        b = b - lr * float(np.mean(g))
    return {"weights": w, "bias": b}


def probe_score(states, labels, epochs=300, lr=0.5, seed=0):
    """Train accuracy of a probe on one layer."""
    probe = train_probe(states, labels, epochs=epochs, lr=lr, seed=seed)
    x = np.asarray(states, dtype=float)
    y = np.asarray(labels, dtype=float)
    z = x @ probe["weights"] + probe["bias"]
    pred = (z >= 0).astype(float)
    return float(np.mean(pred == y))


def probe_report(states_by_layer, labels, epochs=300, lr=0.5, seed=0):
    """Train accuracy per layer and the layer where the property is easiest to read."""
    if not isinstance(states_by_layer, (list, tuple)) or len(states_by_layer) == 0:
        raise ValueError("states_by_layer must be a non-empty list")
    accs = [probe_score(s, labels, epochs=epochs, lr=lr, seed=seed)
            for s in states_by_layer]
    best = int(np.argmax(accs))
    return {"accuracies": accs, "best_layer": best, "best_accuracy": accs[best]}