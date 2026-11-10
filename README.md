# PROBE-STACK

Map where a property lives inside a model, one tiny probe per layer.

PROBE-STACK trains a plain logistic probe by seeded gradient descent on the hidden states of every layer and reports per-layer train accuracy. The layer where the property is easiest to read out is where it lives.

## What it does

- Logistic probes trained by full-batch gradient descent, reproducible.
- Per-layer train accuracy and the best-read layer.
- Works on any (N, D) states, from any framework.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from probe_stack.stack import probe_report

labels = np.load("labels.npy")           # binary 0/1
states_by_layer = [np.load(f) for f in ["l0.npy", "l1.npy", "l2.npy"]]
report = probe_report(states_by_layer, labels)
print(report["accuracies"], report["best_layer"])
```

## How it works

Each layer gets an independent logistic regression trained with the same seed. The accuracy curve shows at which depth the property becomes linearly readable, a standard first estimate of where information is concentrated.

## Honest scope

- Accuracies are train accuracies: they measure readability, not generalization.
- A linear probe can miss properties stored nonlinearly.
- Tests use synthetic states, so no model download is needed.

## License

MIT