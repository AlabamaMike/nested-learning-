"""
Nested Learning: A paradigm for multi-level optimization in deep learning.

This package implements the Nested Learning framework from the NeurIPS 2025 paper
"Nested Learning: The Illusion of Deep Learning Architectures" by Behrouz et al.
"""

from nested_learning.optimizers import (
    DeepMomentumGD,
    DeltaRuleMomentum,
    PreconditionedMomentum,
)
from nested_learning.memory import (
    AssociativeMemory,
    ContinuumMemorySystem,
    ContinuumMemoryBlock,
)
from nested_learning.models import (
    HOPE,
    SelfModifyingLinear,
    SelfModifyingAttention,
    SelfModifyingMLP,
)

__version__ = "0.1.0"
__all__ = [
    # Optimizers
    "DeepMomentumGD",
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
    # Memory
    "AssociativeMemory",
    "ContinuumMemorySystem",
    "ContinuumMemoryBlock",
    # Models
    "HOPE",
    "SelfModifyingLinear",
    "SelfModifyingAttention",
    "SelfModifyingMLP",
]
