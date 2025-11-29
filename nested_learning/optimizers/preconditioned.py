"""
Preconditioned Momentum Optimizer.

This optimizer implements the associative memory view of momentum with
key-value preconditioning. Instead of treating momentum as a simple
accumulator, it views momentum as a learned mapping:

    K -> V : gradient_keys -> update_values

This allows the optimizer to learn different update strategies for
different "types" of gradients based on their characteristics.

The key insight from Nested Learning is:
    W* = arg min_W ⟨Wx, ∇L(W; x)⟩

which shows that the optimal weights compress the relationship between
inputs and their error signals into a lower-dimensional representation.
"""

from typing import Optional, Callable, Iterable, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class PreconditionedMomentum(Optimizer):
    """
    Preconditioned Momentum Optimizer.

    This optimizer maintains a learned key-value memory that maps gradient
    patterns to update directions. The memory is updated online using the
    delta rule, and the optimizer learns to precondition gradients based
    on their similarity to stored patterns.

    Update rules:
        k_t = encode(gradient_t)  # Gradient key
        v_t = memory.read(k_t)     # Retrieve update pattern
        update_t = v_t + alpha * (gradient_t - v_t)  # Blend with actual gradient
        memory.write(k_t, update_t)  # Update memory

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        memory_size: Number of slots in the key-value memory (default: 32)
        key_dim: Dimension of keys (default: 64)
        alpha: Blend factor between memory and gradient (default: 0.5)
        memory_lr: Learning rate for memory updates (default: 0.1)
        weight_decay: L2 regularization (default: 0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        memory_size: int = 32,
        key_dim: int = 64,
        alpha: float = 0.5,
        memory_lr: float = 0.1,
        weight_decay: float = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if memory_size < 1:
            raise ValueError(f"Invalid memory size: {memory_size}")

        defaults = dict(
            lr=lr,
            memory_size=memory_size,
            key_dim=key_dim,
            alpha=alpha,
            memory_lr=memory_lr,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _init_memory(self, param: torch.Tensor, state: Dict[str, Any], group: Dict[str, Any]):
        """Initialize the key-value memory for a parameter."""
        device = param.device
        memory_size = group['memory_size']
        key_dim = group['key_dim']
        param_dim = param.numel()

        # Key encoder: gradient -> key
        # Use random projection for large parameters
        if param_dim > key_dim * 10:
            state['key_proj'] = torch.randn(
                param_dim, key_dim, device=device
            ) / (param_dim ** 0.5)
        else:
            state['key_proj'] = None

        # Key-value memory
        state['keys'] = torch.randn(memory_size, key_dim, device=device) * 0.01
        state['values'] = torch.zeros(memory_size, param_dim, device=device)

        # Memory usage tracking
        state['usage'] = torch.zeros(memory_size, device=device)

    def _encode_key(self, grad: torch.Tensor, state: Dict[str, Any], key_dim: int) -> torch.Tensor:
        """Encode a gradient into a key vector."""
        flat_grad = grad.flatten()

        if state['key_proj'] is not None:
            key = flat_grad @ state['key_proj']
        else:
            # Pad or truncate to key_dim
            if flat_grad.shape[0] >= key_dim:
                key = flat_grad[:key_dim]
            else:
                key = F.pad(flat_grad, (0, key_dim - flat_grad.shape[0]))

        # Normalize key
        key = F.normalize(key, dim=0)
        return key

    def _read_memory(
        self,
        key: torch.Tensor,
        state: Dict[str, Any],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Read from memory using soft attention."""
        keys = state['keys']  # [M, K]
        values = state['values']  # [M, P]

        # Compute attention weights
        similarities = torch.matmul(keys, key)  # [M]
        weights = F.softmax(similarities / temperature, dim=0)  # [M]

        # Retrieve value
        value = torch.matmul(weights, values)  # [P]

        return value, weights

    def _write_memory(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        weights: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
    ):
        """Write to memory using soft attention."""
        keys = state['keys']  # [M, K]
        values = state['values']  # [M, P]
        usage = state['usage']  # [M]

        # Update keys (move toward query key)
        key_update = torch.outer(weights, key) - weights.unsqueeze(1) * keys
        keys.add_(key_update, alpha=lr)

        # Update values (move toward new value)
        value_update = torch.outer(weights, value) - weights.unsqueeze(1) * values
        values.add_(value_update, alpha=lr)

        # Update usage for potential memory management
        usage.add_(weights, alpha=0.1)

        # Re-normalize keys
        state['keys'] = F.normalize(keys, dim=1)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            memory_lr = group['memory_lr']
            weight_decay = group['weight_decay']
            key_dim = group['key_dim']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize memory on first step
                if 'keys' not in state:
                    self._init_memory(p, state, group)
                    state['step'] = 0

                state['step'] += 1
                flat_grad = grad.flatten()

                # Encode gradient as key
                key = self._encode_key(grad, state, key_dim)

                # Read from memory
                value, weights = self._read_memory(key, state)

                # Blend memory value with actual gradient
                update = alpha * value + (1 - alpha) * flat_grad

                # Write back to memory
                self._write_memory(key, flat_grad, weights, state, memory_lr)

                # Apply update
                p.data.add_(update.view_as(p.data), alpha=-lr)

        return loss


class HierarchicalPreconditionedMomentum(Optimizer):
    """
    Hierarchical Preconditioned Momentum with multi-level memory.

    This extends PreconditionedMomentum with multiple memory levels operating
    at different timescales:
    - Level 0: Fast memory, updates every step (immediate gradient patterns)
    - Level 1: Medium memory, updates every N steps (local trends)
    - Level 2: Slow memory, updates every M steps (global patterns)

    This creates a hierarchical associative memory that captures gradient
    patterns at multiple temporal scales.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        num_levels: Number of memory hierarchy levels (default: 3)
        memory_sizes: Memory size per level (default: [16, 32, 64])
        update_frequencies: Update frequency per level (default: [1, 8, 64])
        key_dim: Dimension of keys (default: 64)
        weight_decay: L2 regularization (default: 0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        num_levels: int = 3,
        memory_sizes: Optional[List[int]] = None,
        update_frequencies: Optional[List[int]] = None,
        key_dim: int = 64,
        weight_decay: float = 0,
    ):
        if memory_sizes is None:
            memory_sizes = [16, 32, 64][:num_levels]
        if update_frequencies is None:
            update_frequencies = [1, 8, 64][:num_levels]

        defaults = dict(
            lr=lr,
            num_levels=num_levels,
            memory_sizes=memory_sizes,
            update_frequencies=update_frequencies,
            key_dim=key_dim,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _init_hierarchical_memory(
        self,
        param: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
    ):
        """Initialize multi-level memory hierarchy."""
        device = param.device
        key_dim = group['key_dim']
        param_dim = param.numel()
        num_levels = group['num_levels']
        memory_sizes = group['memory_sizes']

        # Key encoder
        if param_dim > key_dim * 10:
            state['key_proj'] = torch.randn(
                param_dim, key_dim, device=device
            ) / (param_dim ** 0.5)
        else:
            state['key_proj'] = None

        # Initialize each memory level
        state['memories'] = []
        for level in range(num_levels):
            mem_size = memory_sizes[level]
            state['memories'].append({
                'keys': torch.randn(mem_size, key_dim, device=device) * 0.01,
                'values': torch.zeros(mem_size, param_dim, device=device),
                'usage': torch.zeros(mem_size, device=device),
            })

        # Gradient accumulator for each level
        state['accumulators'] = [
            torch.zeros(param_dim, device=device)
            for _ in range(num_levels)
        ]
        state['accumulator_counts'] = [0] * num_levels

    def _encode_key(self, grad: torch.Tensor, state: Dict[str, Any], key_dim: int) -> torch.Tensor:
        """Encode gradient as key."""
        flat_grad = grad.flatten()

        if state['key_proj'] is not None:
            key = flat_grad @ state['key_proj']
        else:
            if flat_grad.shape[0] >= key_dim:
                key = flat_grad[:key_dim]
            else:
                key = F.pad(flat_grad, (0, key_dim - flat_grad.shape[0]))

        return F.normalize(key, dim=0)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            key_dim = group['key_dim']
            num_levels = group['num_levels']
            update_frequencies = group['update_frequencies']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize on first step
                if 'memories' not in state:
                    self._init_hierarchical_memory(p, state, group)
                    state['step'] = 0

                state['step'] += 1
                step = state['step']
                flat_grad = grad.flatten()

                # Encode gradient as key
                key = self._encode_key(grad, state, key_dim)

                # Aggregate updates from all levels
                total_update = torch.zeros_like(flat_grad)
                level_weights = [1.0 / (2 ** i) for i in range(num_levels)]
                weight_sum = sum(level_weights)

                for level in range(num_levels):
                    freq = update_frequencies[level]
                    memory = state['memories'][level]
                    accumulator = state['accumulators'][level]

                    # Accumulate gradient
                    accumulator.add_(flat_grad)
                    state['accumulator_counts'][level] += 1

                    # Update memory at this level's frequency
                    if step % freq == 0:
                        count = state['accumulator_counts'][level]
                        avg_grad = accumulator / count

                        # Read from memory
                        keys = memory['keys']
                        values = memory['values']
                        similarities = torch.matmul(keys, key)
                        weights = F.softmax(similarities, dim=0)
                        value = torch.matmul(weights, values)

                        # Write to memory
                        memory_lr = 0.1 / (level + 1)  # Slower update for higher levels
                        key_update = torch.outer(weights, key) - weights.unsqueeze(1) * keys
                        value_update = torch.outer(weights, avg_grad) - weights.unsqueeze(1) * values
                        keys.add_(key_update, alpha=memory_lr)
                        values.add_(value_update, alpha=memory_lr)
                        memory['keys'] = F.normalize(keys, dim=1)

                        # Reset accumulator
                        accumulator.zero_()
                        state['accumulator_counts'][level] = 0
                    else:
                        # Just read from memory
                        keys = memory['keys']
                        values = memory['values']
                        similarities = torch.matmul(keys, key)
                        weights = F.softmax(similarities, dim=0)
                        value = torch.matmul(weights, values)

                    # Add weighted contribution from this level
                    total_update.add_(value, alpha=level_weights[level] / weight_sum)

                # Blend with actual gradient
                alpha = 0.5
                final_update = alpha * total_update + (1 - alpha) * flat_grad

                # Apply update
                p.data.add_(final_update.view_as(p.data), alpha=-lr)

        return loss
