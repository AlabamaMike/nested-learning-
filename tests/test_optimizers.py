"""Tests for Nested Learning optimizers."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nested_learning.optimizers import (
    DeepMomentumGD,
    DeltaRuleMomentum,
    PreconditionedMomentum,
)
from nested_learning.optimizers.deep_momentum import DeepMomentumGDSimple
from nested_learning.optimizers.delta_rule import AdaptiveDeltaRuleMomentum
from nested_learning.optimizers.preconditioned import HierarchicalPreconditionedMomentum


class TestDeltaRuleMomentum:
    """Tests for Delta Rule Momentum optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        param = torch.randn(10, requires_grad=True)
        optimizer = DeltaRuleMomentum([param], lr=0.01)
        assert optimizer is not None

    def test_step(self):
        """Test optimizer step."""
        param = torch.randn(10, requires_grad=True)
        initial_value = param.clone()

        optimizer = DeltaRuleMomentum([param], lr=0.01)
        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Parameter should have changed
        assert not torch.allclose(param, initial_value)

    def test_surprise_metric(self):
        """Test surprise magnitude computation."""
        param = torch.randn(10, requires_grad=True)
        optimizer = DeltaRuleMomentum([param], lr=0.01)

        # First step
        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Surprise should be available after first step
        surprise = optimizer.get_surprise_magnitude()
        assert surprise >= 0

    def test_multiple_params(self):
        """Test with multiple parameters."""
        params = [torch.randn(10, requires_grad=True) for _ in range(3)]
        optimizer = DeltaRuleMomentum(params, lr=0.01)

        loss = sum(p.sum() for p in params)
        loss.backward()
        optimizer.step()

    def test_convergence(self):
        """Test that optimizer converges on simple problem."""
        target = torch.zeros(10)
        param = torch.randn(10, requires_grad=True)
        optimizer = DeltaRuleMomentum([param], lr=0.1)

        initial_loss = F.mse_loss(param, target).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = F.mse_loss(param, target)
            loss.backward()
            optimizer.step()

        final_loss = F.mse_loss(param, target).item()
        assert final_loss < initial_loss


class TestAdaptiveDeltaRuleMomentum:
    """Tests for Adaptive Delta Rule Momentum."""

    def test_initialization(self):
        """Test optimizer initialization."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdaptiveDeltaRuleMomentum([param], lr=0.01)
        assert optimizer is not None

    def test_step(self):
        """Test optimizer step."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdaptiveDeltaRuleMomentum([param], lr=0.01)

        loss = param.sum()
        loss.backward()
        optimizer.step()


class TestDeepMomentumGDSimple:
    """Tests for simplified Deep Momentum GD."""

    def test_initialization(self):
        """Test optimizer initialization."""
        param = torch.randn(10, requires_grad=True)
        optimizer = DeepMomentumGDSimple([param], lr=0.01)
        assert optimizer is not None

    def test_step(self):
        """Test optimizer step."""
        param = torch.randn(10, requires_grad=True)
        optimizer = DeepMomentumGDSimple([param], lr=0.01)

        loss = param.sum()
        loss.backward()
        optimizer.step()


class TestPreconditionedMomentum:
    """Tests for Preconditioned Momentum optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        param = torch.randn(10, requires_grad=True)
        optimizer = PreconditionedMomentum([param], lr=0.01, memory_size=8)
        assert optimizer is not None

    def test_step(self):
        """Test optimizer step."""
        param = torch.randn(10, requires_grad=True)
        optimizer = PreconditionedMomentum([param], lr=0.01, memory_size=8)

        loss = param.sum()
        loss.backward()
        optimizer.step()

    def test_multiple_steps(self):
        """Test multiple optimization steps."""
        param = torch.randn(10, requires_grad=True)
        optimizer = PreconditionedMomentum([param], lr=0.01, memory_size=8)

        for _ in range(10):
            optimizer.zero_grad()
            loss = param.sum()
            loss.backward()
            optimizer.step()


class TestHierarchicalPreconditionedMomentum:
    """Tests for Hierarchical Preconditioned Momentum."""

    def test_initialization(self):
        """Test optimizer initialization."""
        param = torch.randn(10, requires_grad=True)
        optimizer = HierarchicalPreconditionedMomentum([param], lr=0.01)
        assert optimizer is not None

    def test_step(self):
        """Test optimizer step."""
        param = torch.randn(10, requires_grad=True)
        optimizer = HierarchicalPreconditionedMomentum([param], lr=0.01)

        loss = param.sum()
        loss.backward()
        optimizer.step()
