"""Tests for Nested Learning models."""

import pytest
import torch

from nested_learning.models import (
    HOPE,
    SelfModifyingLinear,
    SelfModifyingAttention,
    SelfModifyingMLP,
)
from nested_learning.models.self_modifying import DeltaRuleLayer
from nested_learning.models.hope import HOPEBlock, HOPEMemoryModule


class TestSelfModifyingLinear:
    """Tests for Self-Modifying Linear layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = SelfModifyingLinear(64, 64)
        assert layer is not None

    def test_forward(self):
        """Test forward pass."""
        layer = SelfModifyingLinear(64, 128)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 128)

    def test_forward_batched(self):
        """Test forward pass with batch dimensions."""
        layer = SelfModifyingLinear(64, 64)
        x = torch.randn(2, 10, 64)
        output = layer(x)
        assert output.shape == (2, 10, 64)

    def test_modification_occurs(self):
        """Test that modification is input-dependent."""
        layer = SelfModifyingLinear(64, 64)

        x1 = torch.randn(4, 64)
        x2 = torch.randn(4, 64)

        # Different inputs should produce different outputs
        # beyond what the base linear would produce
        out1 = layer(x1)
        out2 = layer(x2)

        # Outputs should be different
        assert not torch.allclose(out1, out2)


class TestSelfModifyingAttention:
    """Tests for Self-Modifying Attention."""

    def test_initialization(self):
        """Test attention initialization."""
        attn = SelfModifyingAttention(d_model=64, n_heads=4)
        assert attn is not None

    def test_forward(self):
        """Test forward pass."""
        attn = SelfModifyingAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == x.shape

    def test_forward_with_attention(self):
        """Test forward pass returning attention weights."""
        attn = SelfModifyingAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        output, weights = attn(x, return_attention=True)

        assert output.shape == x.shape
        assert weights.shape == (2, 4, 10, 10)  # [batch, heads, seq, seq]

    def test_causal_masking(self):
        """Test that causal masking is applied."""
        attn = SelfModifyingAttention(d_model=64, n_heads=4)
        x = torch.randn(1, 10, 64)
        _, weights = attn(x, return_attention=True)

        # Upper triangular should be zero (causal)
        for i in range(10):
            for j in range(i + 1, 10):
                assert weights[0, 0, i, j].item() < 1e-6


class TestSelfModifyingMLP:
    """Tests for Self-Modifying MLP."""

    def test_initialization(self):
        """Test MLP initialization."""
        mlp = SelfModifyingMLP(d_model=64)
        assert mlp is not None

    def test_forward(self):
        """Test forward pass."""
        mlp = SelfModifyingMLP(d_model=64, expansion_factor=4)
        x = torch.randn(2, 10, 64)
        output = mlp(x)
        assert output.shape == x.shape


class TestDeltaRuleLayer:
    """Tests for Delta Rule Layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = DeltaRuleLayer(64, 64)
        assert layer is not None

    def test_forward(self):
        """Test forward pass."""
        layer = DeltaRuleLayer(64, 128)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 128)

    def test_error_magnitude(self):
        """Test error magnitude computation."""
        layer = DeltaRuleLayer(64, 64)
        x = torch.randn(4, 64)
        error = layer.get_error_magnitude(x)
        assert error.item() >= 0


class TestHOPEBlock:
    """Tests for HOPE Block."""

    def test_initialization(self):
        """Test block initialization."""
        block = HOPEBlock(d_model=64, n_heads=4)
        assert block is not None

    def test_forward(self):
        """Test forward pass."""
        block = HOPEBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_forward_with_frequency(self):
        """Test forward pass with frequency > 1."""
        block = HOPEBlock(d_model=64, n_heads=4, frequency=4)
        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == x.shape


class TestHOPEMemoryModule:
    """Tests for HOPE Memory Module."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = HOPEMemoryModule(d_model=64)
        assert memory is not None

    def test_forward(self):
        """Test forward pass."""
        memory = HOPEMemoryModule(d_model=64)
        x = torch.randn(2, 10, 64)
        output = memory(x)
        assert output.shape == x.shape

    def test_forward_with_value(self):
        """Test forward pass with value for writing."""
        memory = HOPEMemoryModule(d_model=64)
        memory.train()

        query = torch.randn(2, 10, 64)
        value = torch.randn(2, 10, 64)
        output = memory(query, value)
        assert output.shape == query.shape


class TestHOPE:
    """Tests for HOPE model."""

    def test_initialization(self):
        """Test model initialization."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
        )
        assert model is not None

    def test_forward(self):
        """Test forward pass."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
            max_seq_len=32,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids)

        assert 'logits' in outputs
        assert outputs['logits'].shape == (2, 16, 100)

    def test_forward_with_labels(self):
        """Test forward pass with labels."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, labels=input_ids)

        assert 'loss' in outputs
        assert outputs['loss'].item() > 0

    def test_forward_with_hidden_states(self):
        """Test forward pass returning hidden states."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, return_hidden_states=True)

        assert 'hidden_states' in outputs
        # Should have n_layers + 1 hidden states (input + each layer)
        assert len(outputs['hidden_states']) == 3

    def test_generate(self):
        """Test text generation."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
            max_seq_len=32,
        )

        prompt = torch.tensor([[1, 2, 3]])
        generated = model.generate(prompt, max_length=5)

        assert generated.shape[1] == prompt.shape[1] + 5

    def test_num_params(self):
        """Test parameter counting."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
        )

        params = model.get_num_params()
        assert params > 0

        # Non-embedding should be less than total
        non_emb_params = model.get_num_params(non_embedding=True)
        total_params = model.get_num_params(non_embedding=False)
        assert non_emb_params < total_params

    def test_backward(self):
        """Test backward pass."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']

        loss.backward()

        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads


class TestHOPEVariants:
    """Tests for HOPE model variants."""

    def test_without_memory(self):
        """Test HOPE without memory module."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
            use_memory=False,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids)
        assert outputs['logits'].shape == (2, 16, 100)

    def test_without_cms(self):
        """Test HOPE without CMS."""
        model = HOPE(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
            use_cms=False,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids)
        assert outputs['logits'].shape == (2, 16, 100)

    def test_minimal_config(self):
        """Test HOPE with minimal configuration."""
        model = HOPE(
            d_model=32,
            n_heads=2,
            n_layers=1,
            vocab_size=50,
            use_memory=False,
            use_cms=False,
        )

        input_ids = torch.randint(0, 50, (1, 8))
        outputs = model(input_ids)
        assert outputs['logits'].shape == (1, 8, 50)
