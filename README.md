# Nested Learning

A PyTorch implementation of the Nested Learning paradigm from the NeurIPS 2025 paper:
**"Nested Learning: The Illusion of Deep Learning Architectures"** by Behrouz et al.

## Overview

Nested Learning (NL) is a learning paradigm that represents ML models as nested, multi-level optimization problems, each with its own "context flow" and update frequency. This provides a mathematically transparent, "white-box" view that makes the internal dynamics of learning explicit.

### Key Components

1. **Deep Optimizers**: Enhanced gradient descent methods that replace linear momentum with deep neural networks for richer gradient compression
2. **Continuum Memory System (CMS)**: Multi-frequency memory architecture where different components update at different rates
3. **Self-Modifying Layers**: Layers that modify their own weights during the forward pass
4. **HOPE Architecture**: A self-modifying sequence model combining all these concepts

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from nested_learning import HOPE, DeepMomentumGD, ContinuumMemorySystem

# Create a HOPE model
model = HOPE(
    d_model=512,
    n_heads=8,
    n_layers=6,
    vocab_size=32000,
    frequencies=[1, 16, 64, 256]
)

# Use the Deep Momentum optimizer
optimizer = DeepMomentumGD(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Project Structure

```
nested_learning/
├── optimizers/           # Deep optimizer implementations
│   ├── deep_momentum.py  # Deep Momentum Gradient Descent
│   ├── delta_rule.py     # Delta Rule Momentum
│   └── preconditioned.py # Preconditioned Momentum
├── memory/               # Memory systems
│   ├── associative.py    # Associative memory modules
│   └── continuum.py      # Continuum Memory System
├── models/               # Model architectures
│   ├── self_modifying.py # Self-modifying layers
│   └── hope.py           # HOPE model
└── utils/                # Utilities
    └── training.py       # Training helpers
```

## References

- [Paper PDF](https://abehrouz.github.io/files/NL.pdf)
- [Google Research Blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- [OpenReview](https://openreview.net/forum?id=nbMeRvNb7A)

## License

MIT License
