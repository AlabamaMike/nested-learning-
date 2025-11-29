"""
Deep Optimizers for Nested Learning.

This module implements optimizers that view gradient descent with momentum as a
nested optimization problem, where the momentum term is an associative memory
that learns to compress the history of gradients.
"""

from nested_learning.optimizers.deep_momentum import DeepMomentumGD
from nested_learning.optimizers.delta_rule import DeltaRuleMomentum
from nested_learning.optimizers.preconditioned import PreconditionedMomentum

__all__ = [
    "DeepMomentumGD",
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
]
