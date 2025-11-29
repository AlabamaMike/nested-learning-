"""
Models for Nested Learning.

This module implements the model architectures for Nested Learning:
- Self-Modifying Layers: Layers that modify their weights during forward pass
- HOPE: Hierarchical Optimizing Processing Ensemble sequence model
"""

from nested_learning.models.self_modifying import (
    SelfModifyingLinear,
    SelfModifyingAttention,
    SelfModifyingMLP,
)
from nested_learning.models.hope import HOPE

__all__ = [
    "SelfModifyingLinear",
    "SelfModifyingAttention",
    "SelfModifyingMLP",
    "HOPE",
]
