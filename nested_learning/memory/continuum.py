"""
Continuum Memory System (CMS) for Nested Learning.

The CMS is a key contribution of Nested Learning, providing a multi-frequency
memory architecture where different components update at different rates:

- High-frequency (f1): Every token - captures syntax and immediate dependencies
- Medium-frequency (f2): Every 16-64 tokens - captures semantic patterns
- Low-frequency (f3): Every 256-1024 tokens - captures discourse structure
- Very low-frequency: Per-sequence - captures document-level understanding

The multi-frequency update rule:
    θ^(f_ℓ)_{i+1} = θ^(f_ℓ)_i − (∑_{t=i-C^(ℓ)}^i η^(ℓ)_t ∇L(θ^(f_ℓ)_t; x_t))
    if i ≡ 0 (mod C^(ℓ))

Where C^(ℓ) represents the chunk size for frequency level ℓ.

The output is computed as stacked MLPs at different frequencies:
    y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))
"""

from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FrequencyBlock(nn.Module):
    """
    A single frequency-specific block in the CMS.

    This block accumulates gradients over a specific number of steps
    before applying updates, creating different temporal dynamics
    at different layers.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for the MLP
        output_dim: Output dimension
        frequency: Update frequency (1 = every step, N = every N steps)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        frequency: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.frequency = frequency

        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

        # For gradient accumulation
        self.register_buffer('step_counter', torch.tensor(0))
        self.accumulated_grads: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the frequency block."""
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # Residual connection if dimensions match
        if self.input_dim == self.output_dim:
            h = h + x

        return self.norm(h)

    def accumulate_gradient(self):
        """Accumulate gradients for this block's parameters."""
        for name, param in self.named_parameters():
            if param.grad is not None:
                if name not in self.accumulated_grads:
                    self.accumulated_grads[name] = torch.zeros_like(param.grad)
                self.accumulated_grads[name].add_(param.grad)

    def should_update(self) -> bool:
        """Check if this block should update based on frequency."""
        self.step_counter += 1
        return self.step_counter.item() % self.frequency == 0

    def apply_accumulated_gradient(self, lr: float):
        """Apply accumulated gradients and reset."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.accumulated_grads:
                    # Average the accumulated gradients
                    avg_grad = self.accumulated_grads[name] / self.frequency
                    param.add_(avg_grad, alpha=-lr)

        # Reset accumulators
        self.accumulated_grads.clear()


class ContinuumMemoryBlock(nn.Module):
    """
    A complete CMS block with multiple frequency levels.

    This implements the stacked MLP architecture from the paper:
        y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))

    Each MLP updates at a different frequency, creating a spectrum of
    temporal dynamics from fast adaptation to slow consolidation.

    Args:
        d_model: Model dimension
        frequencies: List of update frequencies (default: [1, 16, 64, 256])
        expansion_factor: MLP expansion factor (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        frequencies: Optional[List[int]] = None,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if frequencies is None:
            frequencies = [1, 16, 64, 256]

        self.d_model = d_model
        self.frequencies = frequencies
        self.num_levels = len(frequencies)

        # Create frequency blocks
        self.blocks = nn.ModuleList()
        hidden_dim = d_model * expansion_factor

        for freq in frequencies:
            self.blocks.append(
                FrequencyBlock(
                    input_dim=d_model,
                    hidden_dim=hidden_dim,
                    output_dim=d_model,
                    frequency=freq,
                    dropout=dropout,
                )
            )

        # Frequency-specific gating
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
            for _ in frequencies
        ])

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all frequency levels.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Process through each frequency level
        output = x
        contributions = []

        for i, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            # Forward through block
            block_output = block(output)

            # Compute gate for this frequency
            g = gate(block_output)

            # Store gated contribution
            contributions.append(g * block_output)

            # Update output for next level
            output = block_output

        # Combine all frequency contributions
        combined = sum(contributions) / len(contributions)

        return self.norm(combined + x)

    def update_frequencies(self, lr: float = 1e-3):
        """
        Update parameters based on accumulated gradients.

        Call this during training to apply the multi-frequency update rule.
        """
        for block in self.blocks:
            if block.should_update():
                block.apply_accumulated_gradient(lr)

    def accumulate_all_gradients(self):
        """Accumulate gradients for all frequency blocks."""
        for block in self.blocks:
            block.accumulate_gradient()


class ContinuumMemorySystem(nn.Module):
    """
    Complete Continuum Memory System.

    This is the full CMS implementation that combines:
    1. Multi-frequency MLP blocks
    2. Frequency-specific memory modules
    3. Inter-frequency communication

    The system creates a "continuum" of memory from fast-changing
    working memory to slow-changing knowledge storage.

    Args:
        d_model: Model dimension
        n_layers: Number of CMS blocks (default: 4)
        frequencies: Update frequencies (default: [1, 16, 64, 256])
        memory_size: Size of memory at each frequency (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        frequencies: Optional[List[int]] = None,
        memory_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if frequencies is None:
            frequencies = [1, 16, 64, 256]

        self.d_model = d_model
        self.n_layers = n_layers
        self.frequencies = frequencies

        # CMS blocks
        self.blocks = nn.ModuleList([
            ContinuumMemoryBlock(d_model, frequencies, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Frequency-specific memory (key-value storage)
        self.memories = nn.ModuleList()
        for freq in frequencies:
            self.memories.append(
                FrequencyMemory(
                    d_model=d_model,
                    memory_size=memory_size // (frequencies.index(freq) + 1),
                    update_frequency=freq,
                )
            )

        # Cross-frequency attention
        self.cross_freq_attn = CrossFrequencyAttention(
            d_model=d_model,
            num_frequencies=len(frequencies),
        )

    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the CMS.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            use_memory: Whether to use frequency-specific memories

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Process through CMS blocks
        output = x
        for block in self.blocks:
            output = block(output)

        # Apply frequency-specific memories
        if use_memory:
            memory_outputs = []
            for i, memory in enumerate(self.memories):
                mem_out = memory(output)
                memory_outputs.append(mem_out)

            # Cross-frequency attention
            output = self.cross_freq_attn(output, memory_outputs)

        return output

    def get_frequency_states(self) -> List[torch.Tensor]:
        """Get the current state of each frequency-specific memory."""
        states = []
        for memory in self.memories:
            states.append(memory.get_memory_state())
        return states


class FrequencyMemory(nn.Module):
    """
    Memory module that operates at a specific frequency.

    This memory accumulates information over multiple steps and
    only updates at its designated frequency, creating stable
    representations that change slowly.

    Args:
        d_model: Model dimension
        memory_size: Number of memory slots
        update_frequency: How often to update memory
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 64,
        update_frequency: int = 1,
    ):
        super().__init__()

        self.d_model = d_model
        self.memory_size = memory_size
        self.update_frequency = update_frequency

        # Memory storage
        self.register_buffer('memory', torch.zeros(memory_size, d_model))
        self.register_buffer('memory_mask', torch.zeros(memory_size))
        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('accumulator', torch.zeros(memory_size, d_model))
        self.register_buffer('accumulator_count', torch.zeros(memory_size))

        # Query/Key/Value projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

        # Memory write gate
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Query memory and optionally write.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Memory-augmented output
        """
        batch_size, seq_len, _ = x.shape

        # Query projection
        q = self.query(x)  # [batch, seq, d_model]

        # Key and value from memory
        k = self.key(self.memory)  # [memory, d_model]
        v = self.value(self.memory)  # [memory, d_model]

        # Attention over memory
        scores = torch.matmul(q, k.t()) / (self.d_model ** 0.5)

        # Mask empty memory slots
        scores = scores.masked_fill(self.memory_mask == 0, float('-inf'))

        # Softmax (handle all -inf case)
        if self.memory_mask.sum() > 0:
            weights = F.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
        else:
            weights = torch.zeros_like(scores)

        # Retrieve from memory
        retrieved = torch.matmul(weights, v)  # [batch, seq, d_model]

        # Output projection with residual
        output = self.output(retrieved) + x

        # Accumulate for writing (during training)
        if self.training:
            self._accumulate_for_write(x)

        return output

    def _accumulate_for_write(self, x: torch.Tensor):
        """Accumulate information for memory write."""
        self.step_counter += 1

        # Pool over batch and sequence
        pooled = x.mean(dim=[0, 1])  # [d_model]

        # Find least-used slot
        if self.memory_mask.sum() < self.memory_size:
            # Use next empty slot
            idx = int((self.memory_mask == 0).nonzero()[0].item())
        else:
            # Use slot with lowest accumulator count
            idx = int(self.accumulator_count.argmin().item())

        # Accumulate
        self.accumulator[idx] = self.accumulator[idx] + pooled
        self.accumulator_count[idx] = self.accumulator_count[idx] + 1

        # Check if we should commit to memory
        if self.step_counter.item() % self.update_frequency == 0:
            self._commit_to_memory()

    def _commit_to_memory(self):
        """Commit accumulated values to memory."""
        with torch.no_grad():
            # Find slots with accumulated values
            active_slots = (self.accumulator_count > 0).nonzero(as_tuple=True)[0]

            for idx in active_slots:
                avg_value = self.accumulator[idx] / self.accumulator_count[idx]

                # Compute write gate
                combined = torch.cat([self.memory[idx], avg_value])
                gate = self.write_gate(combined.unsqueeze(0)).squeeze()

                # Update memory
                self.memory[idx] = gate * avg_value + (1 - gate) * self.memory[idx]
                self.memory_mask[idx] = 1.0

            # Reset accumulators
            self.accumulator.zero_()
            self.accumulator_count.zero_()

    def get_memory_state(self) -> torch.Tensor:
        """Return current memory state."""
        return self.memory.clone()


class CrossFrequencyAttention(nn.Module):
    """
    Attention mechanism for combining information across frequencies.

    This allows different frequency levels to communicate, enabling
    fast-changing components to access slow-changing knowledge and
    slow components to incorporate recent patterns.

    Args:
        d_model: Model dimension
        num_frequencies: Number of frequency levels
        n_heads: Number of attention heads (default: 4)
    """

    def __init__(
        self,
        d_model: int,
        num_frequencies: int,
        n_heads: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_frequencies = num_frequencies
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections for main input
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Frequency embeddings
        self.freq_embed = nn.Embedding(num_frequencies, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        freq_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine main input with frequency-specific outputs.

        Args:
            x: Main input [batch, seq_len, d_model]
            freq_outputs: List of outputs from each frequency level

        Returns:
            Combined output
        """
        batch_size, seq_len, _ = x.shape

        # Query from main input
        q = self.q_proj(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)

        # Stack frequency outputs with embeddings
        freq_stack = []
        for i, freq_out in enumerate(freq_outputs):
            # Add frequency embedding
            freq_emb = self.freq_embed(
                torch.tensor(i, device=x.device)
            ).unsqueeze(0).unsqueeze(0)
            freq_out = freq_out + freq_emb
            freq_stack.append(freq_out)

        # Concatenate frequency outputs
        freq_concat = torch.cat(freq_stack, dim=1)  # [batch, num_freq * seq, d_model]

        # Key and value from frequency outputs
        k = self.k_proj(freq_concat)
        v = self.v_proj(freq_concat)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)

        # Reshape and project
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        out = self.out_proj(attn_out)

        return self.norm(out + x)
