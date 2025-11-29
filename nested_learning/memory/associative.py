"""
Associative Memory Module for Nested Learning.

This implements the core insight from Nested Learning that all neural network
components can be viewed as associative memories that compress their context flow.

For a single-layer network, the optimal weights satisfy:
    W* = arg min_W ⟨Wx, ∇L(W; x)⟩

This shows that weights compress the relationship between inputs and their
error signals. The AssociativeMemory module makes this explicit by providing
read/write operations that learn from input-output pairs.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class AssociativeMemory(nn.Module):
    """
    Associative Memory with learnable key-value storage.

    This module implements a differentiable associative memory that can:
    1. Store key-value pairs
    2. Retrieve values based on query similarity
    3. Update its storage based on prediction error (delta rule)

    The memory operates as a learned compression of the mapping from
    inputs (keys) to outputs (values), which is the core abstraction
    underlying all neural network layers in the Nested Learning view.

    Args:
        memory_size: Number of memory slots (default: 128)
        key_dim: Dimension of keys (default: 64)
        value_dim: Dimension of values (default: 64)
        num_heads: Number of attention heads for retrieval (default: 4)
        use_delta_rule: Use delta rule for online updates (default: True)
        temperature: Softmax temperature for retrieval (default: 1.0)
    """

    def __init__(
        self,
        memory_size: int = 128,
        key_dim: int = 64,
        value_dim: int = 64,
        num_heads: int = 4,
        use_delta_rule: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.use_delta_rule = use_delta_rule
        self.temperature = temperature
        self.head_dim = key_dim // num_heads

        # Memory storage
        self.register_buffer(
            'keys',
            torch.randn(memory_size, key_dim) * 0.02,
        )
        self.register_buffer(
            'values',
            torch.zeros(memory_size, value_dim),
        )

        # Query/Key/Value projections for multi-head attention
        self.query_proj = nn.Linear(key_dim, key_dim)
        self.key_proj_in = nn.Linear(key_dim, key_dim)
        self.value_proj_out = nn.Linear(value_dim, value_dim)

        # Delta rule learning rate (learnable)
        self.delta_lr = nn.Parameter(torch.tensor(0.1))

        # Surprise gate for gating updates based on prediction error
        self.surprise_gate = nn.Sequential(
            nn.Linear(value_dim, value_dim // 2),
            nn.GELU(),
            nn.Linear(value_dim // 2, 1),
            nn.Sigmoid(),
        )

    def read(
        self,
        query: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Read from memory using the given query.

        Args:
            query: Query tensor of shape [batch, key_dim]
            return_weights: Whether to return attention weights

        Returns:
            Retrieved value and optionally the attention weights
        """
        batch_size = query.shape[0]

        # Project query
        q = self.query_proj(query)  # [batch, key_dim]

        # Reshape for multi-head attention
        q = rearrange(q, 'b (h d) -> b h d', h=self.num_heads)

        # Project stored keys
        k = self.key_proj_in(self.keys)  # [memory_size, key_dim]
        k = rearrange(k, 'm (h d) -> h m d', h=self.num_heads)

        # Compute attention
        scores = torch.einsum('bhd,hmd->bhm', q, k) / (self.head_dim ** 0.5)
        weights = F.softmax(scores / self.temperature, dim=-1)  # [batch, heads, memory]

        # Average attention across heads
        weights_avg = weights.mean(dim=1)  # [batch, memory]

        # Retrieve values
        v = self.values  # [memory, value_dim]
        retrieved = torch.einsum('bm,mv->bv', weights_avg, v)

        # Project output
        output = self.value_proj_out(retrieved)

        if return_weights:
            return output, weights_avg
        return output, None

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ):
        """
        Write to memory using the delta rule.

        The write operation updates memory based on the prediction error:
            keys += lr * gate * (key - weighted_keys)
            values += lr * gate * (value - weighted_values)

        Args:
            key: Key tensor of shape [batch, key_dim]
            value: Value tensor of shape [batch, value_dim]
            gate: Optional gate tensor of shape [batch, 1]
        """
        if not self.use_delta_rule:
            return

        with torch.no_grad():
            batch_size = key.shape[0]

            # Get current prediction
            predicted, weights = self.read(key, return_weights=True)

            # Compute prediction error
            error = value - predicted  # [batch, value_dim]

            # Compute surprise-based gate if not provided
            if gate is None:
                gate = self.surprise_gate(error.abs())  # [batch, 1]

            # Apply delta rule update
            lr = torch.sigmoid(self.delta_lr)

            # Update values: move stored values toward target
            # Weighted by attention and gated by surprise
            value_delta = torch.einsum('bm,bv->mv', weights * gate, error)
            value_delta = value_delta / (batch_size + 1e-8)
            self.values.add_(value_delta, alpha=lr.item())

            # Update keys: move toward query keys
            key_proj = self.key_proj_in(key)
            key_delta = torch.einsum('bm,bd->md', weights * gate, key_proj - self.keys)
            key_delta = key_delta / (batch_size + 1e-8)
            self.keys.add_(key_delta, alpha=lr.item() * 0.1)

            # Normalize keys
            self.keys.data = F.normalize(self.keys.data, dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        target_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional write-back.

        Args:
            query: Query tensor [batch, key_dim]
            target_value: Optional target value for delta rule update

        Returns:
            Retrieved value and surprise/prediction error
        """
        retrieved, _ = self.read(query)

        if target_value is not None:
            error = target_value - retrieved
            surprise = error.abs().mean(dim=-1, keepdim=True)

            if self.training:
                self.write(query, target_value)

            return retrieved, surprise

        return retrieved, torch.zeros(query.shape[0], 1, device=query.device)


class SurpriseBasedMemory(nn.Module):
    """
    Memory module that prioritizes storage based on surprise.

    Inspired by Titans architecture, this memory stores experiences based
    on how "surprising" they are (high prediction error). This creates
    an implicit curriculum where the memory focuses on hard examples.

    Args:
        memory_size: Number of memory slots (default: 256)
        key_dim: Dimension of keys (default: 64)
        value_dim: Dimension of values (default: 64)
        surprise_threshold: Minimum surprise to trigger write (default: 0.5)
    """

    def __init__(
        self,
        memory_size: int = 256,
        key_dim: int = 64,
        value_dim: int = 64,
        surprise_threshold: float = 0.5,
    ):
        super().__init__()

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.surprise_threshold = surprise_threshold

        # Memory storage
        self.register_buffer('keys', torch.randn(memory_size, key_dim) * 0.02)
        self.register_buffer('values', torch.zeros(memory_size, value_dim))
        self.register_buffer('surprise_scores', torch.zeros(memory_size))
        self.register_buffer('write_ptr', torch.tensor(0))

        # Projections
        self.query_proj = nn.Linear(key_dim, key_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from memory."""
        q = self.query_proj(query)  # [batch, key_dim]
        k = self.key_proj(self.keys)  # [memory, key_dim]

        # Attention
        scores = torch.matmul(q, k.t()) / (self.key_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        # Weight by stored surprise (higher surprise = more weight)
        surprise_weights = F.softmax(self.surprise_scores, dim=0)
        weights = weights * surprise_weights.unsqueeze(0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Retrieve
        retrieved = torch.matmul(weights, self.values)
        return self.value_proj(retrieved)

    def write(self, key: torch.Tensor, value: torch.Tensor, surprise: torch.Tensor):
        """
        Write to memory if surprise exceeds threshold.

        Uses a least-surprised eviction policy: replaces the memory slot
        with the lowest surprise score.
        """
        with torch.no_grad():
            batch_size = key.shape[0]

            for i in range(batch_size):
                s = surprise[i].item()

                if s > self.surprise_threshold:
                    # Find slot to replace (lowest surprise)
                    min_idx = self.surprise_scores.argmin().item()

                    # Only replace if new surprise is higher
                    if s > self.surprise_scores[min_idx].item():
                        self.keys[min_idx] = key[i]
                        self.values[min_idx] = value[i]
                        self.surprise_scores[min_idx] = s

            # Decay surprise scores over time
            self.surprise_scores.mul_(0.99)

    def forward(
        self,
        query: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional write."""
        retrieved = self.read(query)

        if target is not None:
            error = target - retrieved
            surprise = error.norm(dim=-1, keepdim=True)

            if self.training:
                self.write(query, target, surprise)

            return retrieved, surprise

        return retrieved, torch.zeros(query.shape[0], 1, device=query.device)


class HierarchicalAssociativeMemory(nn.Module):
    """
    Hierarchical Associative Memory with multiple levels.

    This implements a multi-level memory hierarchy where each level
    operates at a different abstraction level:
    - Level 0: Token-level patterns (fine-grained)
    - Level 1: Phrase-level patterns (medium)
    - Level 2: Document-level patterns (coarse)

    Each level compresses information from the level below, creating
    a pyramid of increasingly abstract representations.

    Args:
        num_levels: Number of hierarchy levels (default: 3)
        memory_sizes: Memory size per level (default: [64, 32, 16])
        dims: Dimension per level (default: [64, 128, 256])
    """

    def __init__(
        self,
        num_levels: int = 3,
        memory_sizes: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if memory_sizes is None:
            memory_sizes = [64, 32, 16][:num_levels]
        if dims is None:
            dims = [64, 128, 256][:num_levels]

        self.num_levels = num_levels
        self.dims = dims

        # Create memory at each level
        self.memories = nn.ModuleList()
        for i in range(num_levels):
            self.memories.append(
                AssociativeMemory(
                    memory_size=memory_sizes[i],
                    key_dim=dims[i],
                    value_dim=dims[i],
                )
            )

        # Inter-level projections
        self.up_projs = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.up_projs.append(nn.Linear(dims[i], dims[i + 1]))
            self.down_projs.append(nn.Linear(dims[i + 1], dims[i]))

        # Compression gates
        self.compress_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i] // 4),
                nn.GELU(),
                nn.Linear(dims[i] // 4, 1),
                nn.Sigmoid(),
            )
            for i in range(num_levels)
        ])

    def forward(
        self,
        x: torch.Tensor,
        level: int = 0,
        propagate_up: bool = True,
        propagate_down: bool = True,
    ) -> torch.Tensor:
        """
        Process input through hierarchical memory.

        Args:
            x: Input tensor [batch, dim]
            level: Starting level (default: 0)
            propagate_up: Propagate to higher levels
            propagate_down: Propagate back to lower levels

        Returns:
            Output tensor with hierarchical context
        """
        # Store activations at each level
        activations = [None] * self.num_levels
        activations[level] = x

        # Read from current level
        current = x
        output, _ = self.memories[level](current)

        # Propagate up the hierarchy
        if propagate_up:
            for i in range(level, self.num_levels - 1):
                # Project up
                current = self.up_projs[i](output)

                # Read from higher level
                higher_output, _ = self.memories[i + 1](current)

                # Gate compression
                gate = self.compress_gates[i + 1](higher_output)
                output = gate * higher_output + (1 - gate) * current
                activations[i + 1] = output

        # Propagate down the hierarchy
        if propagate_down:
            for i in range(min(level + 1, self.num_levels - 1), 0, -1):
                if activations[i] is not None:
                    # Project down
                    current = self.down_projs[i - 1](activations[i])

                    # Combine with lower level
                    if activations[i - 1] is not None:
                        gate = self.compress_gates[i - 1](current)
                        activations[i - 1] = (
                            gate * current + (1 - gate) * activations[i - 1]
                        )

        # Return output at original level
        return activations[level] if activations[level] is not None else output
