"""Tests for Nested Learning memory modules."""

import pytest
import torch

from nested_learning.memory import (
    AssociativeMemory,
    ContinuumMemorySystem,
    ContinuumMemoryBlock,
)
from nested_learning.memory.associative import (
    SurpriseBasedMemory,
    HierarchicalAssociativeMemory,
)
from nested_learning.memory.continuum import FrequencyMemory, CrossFrequencyAttention


class TestAssociativeMemory:
    """Tests for Associative Memory."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = AssociativeMemory(memory_size=32, key_dim=64, value_dim=64)
        assert memory is not None

    def test_read(self):
        """Test memory read operation."""
        memory = AssociativeMemory(memory_size=32, key_dim=64, value_dim=64)
        query = torch.randn(4, 64)
        output, weights = memory.read(query, return_weights=True)

        assert output.shape == (4, 64)
        assert weights.shape == (4, 32)

    def test_write(self):
        """Test memory write operation."""
        memory = AssociativeMemory(memory_size=32, key_dim=64, value_dim=64)
        memory.train()

        key = torch.randn(4, 64)
        value = torch.randn(4, 64)
        memory.write(key, value)

    def test_forward(self):
        """Test forward pass."""
        memory = AssociativeMemory(memory_size=32, key_dim=64, value_dim=64)
        query = torch.randn(4, 64)

        output, surprise = memory(query)
        assert output.shape == query.shape

    def test_forward_with_target(self):
        """Test forward pass with target value."""
        memory = AssociativeMemory(memory_size=32, key_dim=64, value_dim=64)
        memory.train()

        query = torch.randn(4, 64)
        target = torch.randn(4, 64)

        output, surprise = memory(query, target)
        assert output.shape == query.shape
        assert surprise.shape == (4, 1)


class TestSurpriseBasedMemory:
    """Tests for Surprise-Based Memory."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = SurpriseBasedMemory(memory_size=32, key_dim=64, value_dim=64)
        assert memory is not None

    def test_read(self):
        """Test memory read operation."""
        memory = SurpriseBasedMemory(memory_size=32, key_dim=64, value_dim=64)
        query = torch.randn(4, 64)
        output = memory.read(query)
        assert output.shape == (4, 64)

    def test_write(self):
        """Test surprise-based write."""
        memory = SurpriseBasedMemory(
            memory_size=5,
            key_dim=64,
            value_dim=64,
            surprise_threshold=0.1,
        )
        memory.train()

        key = torch.randn(2, 64)
        value = torch.randn(2, 64)
        surprise = torch.tensor([[1.0], [0.05]])  # One high, one low

        memory.write(key, value, surprise)


class TestHierarchicalAssociativeMemory:
    """Tests for Hierarchical Associative Memory."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = HierarchicalAssociativeMemory(num_levels=3)
        assert memory is not None

    def test_forward(self):
        """Test forward pass through hierarchy."""
        memory = HierarchicalAssociativeMemory(
            num_levels=3,
            dims=[64, 64, 64],
        )
        x = torch.randn(4, 64)
        output = memory(x)
        assert output.shape == x.shape


class TestContinuumMemoryBlock:
    """Tests for Continuum Memory Block."""

    def test_initialization(self):
        """Test block initialization."""
        block = ContinuumMemoryBlock(d_model=64, frequencies=[1, 4, 16])
        assert block is not None
        assert len(block.blocks) == 3

    def test_forward(self):
        """Test forward pass."""
        block = ContinuumMemoryBlock(d_model=64, frequencies=[1, 4])
        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == x.shape


class TestContinuumMemorySystem:
    """Tests for full Continuum Memory System."""

    def test_initialization(self):
        """Test CMS initialization."""
        cms = ContinuumMemorySystem(d_model=64, n_layers=2, frequencies=[1, 4])
        assert cms is not None

    def test_forward(self):
        """Test forward pass."""
        cms = ContinuumMemorySystem(d_model=64, n_layers=2, frequencies=[1, 4])
        x = torch.randn(2, 10, 64)
        output = cms(x)
        assert output.shape == x.shape

    def test_forward_without_memory(self):
        """Test forward pass without memory modules."""
        cms = ContinuumMemorySystem(d_model=64, n_layers=2, frequencies=[1, 4])
        x = torch.randn(2, 10, 64)
        output = cms(x, use_memory=False)
        assert output.shape == x.shape


class TestFrequencyMemory:
    """Tests for Frequency Memory."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = FrequencyMemory(d_model=64, memory_size=16, update_frequency=4)
        assert memory is not None

    def test_forward(self):
        """Test forward pass."""
        memory = FrequencyMemory(d_model=64, memory_size=16, update_frequency=4)
        x = torch.randn(2, 10, 64)
        output = memory(x)
        assert output.shape == x.shape


class TestCrossFrequencyAttention:
    """Tests for Cross-Frequency Attention."""

    def test_initialization(self):
        """Test attention initialization."""
        attn = CrossFrequencyAttention(d_model=64, num_frequencies=3)
        assert attn is not None

    def test_forward(self):
        """Test forward pass."""
        attn = CrossFrequencyAttention(d_model=64, num_frequencies=3)
        x = torch.randn(2, 10, 64)
        freq_outputs = [torch.randn(2, 10, 64) for _ in range(3)]

        output = attn(x, freq_outputs)
        assert output.shape == x.shape
