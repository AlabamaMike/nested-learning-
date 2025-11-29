"""
HOPE: Hierarchical Optimizing Processing Ensemble.

HOPE is the main sequence model from the Nested Learning paper that combines:
1. Self-modifying attention and MLP layers
2. Continuum Memory System (CMS) for multi-frequency learning
3. Recursive self-optimization capabilities

Key features:
- Unbounded in-context learning through self-referential optimization
- Multi-frequency parameter updates for stable knowledge consolidation
- Surprise-based memory prioritization (from Titans)
- Self-modification of weights during forward pass

HOPE achieves superior performance on:
- Language modeling (lower perplexity vs. Titans, Samba, Transformers)
- Long-context reasoning (maintains accuracy at 1M+ tokens)
- Continual learning (avoids catastrophic forgetting)
"""

from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nested_learning.models.self_modifying import (
    SelfModifyingAttention,
    SelfModifyingMLP,
)
from nested_learning.memory.continuum import ContinuumMemoryBlock, FrequencyMemory
from nested_learning.memory.associative import AssociativeMemory, SurpriseBasedMemory


class HOPEBlock(nn.Module):
    """
    A single HOPE transformer block.

    Combines self-modifying attention and MLP with optional
    frequency-specific processing.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        expansion_factor: MLP expansion factor (default: 4)
        modification_rank: Rank of self-modifications (default: None)
        frequency: Update frequency for this block (default: 1)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_factor: int = 4,
        modification_rank: Optional[int] = None,
        frequency: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.frequency = frequency

        # Pre-normalization
        self.attn_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)

        # Self-modifying attention
        self.attention = SelfModifyingAttention(
            d_model=d_model,
            n_heads=n_heads,
            modification_rank=modification_rank,
            dropout=dropout,
        )

        # Self-modifying MLP
        self.mlp = SelfModifyingMLP(
            d_model=d_model,
            expansion_factor=expansion_factor,
            modification_rank=modification_rank,
            dropout=dropout,
        )

        # Optional frequency-specific processing
        if frequency > 1:
            self.freq_processor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.freq_processor = None

        # For gradient accumulation at this frequency
        self.register_buffer('step_counter', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the HOPE block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual
        h = self.attn_norm(x)
        attn_out, _ = self.attention(h, attention_mask)
        x = x + attn_out

        # MLP with residual
        h = self.mlp_norm(x)
        mlp_out = self.mlp(h)
        x = x + mlp_out

        # Frequency-specific processing
        if self.freq_processor is not None:
            self.step_counter += 1
            if self.step_counter.item() % self.frequency == 0:
                x = x + 0.1 * self.freq_processor(x)

        return x


class HOPEMemoryModule(nn.Module):
    """
    Memory module for HOPE combining associative and surprise-based memory.

    This module maintains multiple memory systems:
    1. Associative memory for general pattern storage
    2. Surprise-based memory for unexpected/important patterns
    3. Working memory for recent context

    Args:
        d_model: Model dimension
        memory_size: Size of each memory type (default: 128)
        working_memory_size: Size of working memory (default: 32)
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 128,
        working_memory_size: int = 32,
    ):
        super().__init__()

        self.d_model = d_model

        # Associative memory for general patterns
        self.associative = AssociativeMemory(
            memory_size=memory_size,
            key_dim=d_model,
            value_dim=d_model,
        )

        # Surprise-based memory for important/unexpected patterns
        self.surprise_memory = SurpriseBasedMemory(
            memory_size=memory_size,
            key_dim=d_model,
            value_dim=d_model,
        )

        # Working memory (FIFO buffer)
        self.register_buffer(
            'working_memory',
            torch.zeros(working_memory_size, d_model)
        )
        self.register_buffer('wm_ptr', torch.tensor(0))

        # Memory combination
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Query all memory systems and combine results.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            value: Optional value for memory write

        Returns:
            Memory-augmented output
        """
        batch_size, seq_len, _ = query.shape

        # Reshape for memory operations
        flat_query = query.reshape(-1, self.d_model)

        # Query associative memory
        assoc_out, _ = self.associative.read(flat_query)

        # Query surprise memory
        surprise_out = self.surprise_memory.read(flat_query)

        # Query working memory
        wm_scores = torch.matmul(flat_query, self.working_memory.t())
        wm_weights = F.softmax(wm_scores / (self.d_model ** 0.5), dim=-1)
        wm_out = torch.matmul(wm_weights, self.working_memory)

        # Compute gating weights
        combined = torch.cat([assoc_out, surprise_out, wm_out], dim=-1)
        gates = self.memory_gate(combined)  # [batch*seq, 3]

        # Weighted combination
        memory_out = (
            gates[:, 0:1] * assoc_out +
            gates[:, 1:2] * surprise_out +
            gates[:, 2:3] * wm_out
        )

        # Reshape back
        memory_out = memory_out.reshape(batch_size, seq_len, self.d_model)

        # Write to memories if value provided
        if value is not None and self.training:
            flat_value = value.reshape(-1, self.d_model)
            self.associative.write(flat_query, flat_value)

            # Compute surprise for surprise-based memory
            error = flat_value - assoc_out
            surprise = error.norm(dim=-1, keepdim=True)
            self.surprise_memory.write(flat_query, flat_value, surprise)

            # Update working memory (FIFO)
            self._update_working_memory(flat_value)

        return self.output_proj(memory_out) + query

    def _update_working_memory(self, values: torch.Tensor):
        """Update working memory with recent values."""
        with torch.no_grad():
            # Take mean over batch for working memory update
            avg_value = values.mean(dim=0)
            ptr = self.wm_ptr.item()
            self.working_memory[ptr] = avg_value
            self.wm_ptr = (self.wm_ptr + 1) % self.working_memory.shape[0]


class HOPE(nn.Module):
    """
    HOPE: Hierarchical Optimizing Processing Ensemble.

    The main sequence model from Nested Learning that combines all concepts:
    - Self-modifying attention and MLP layers
    - Continuum Memory System (multi-frequency updates)
    - Associative and surprise-based memory
    - Unbounded in-context learning

    Args:
        d_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 6)
        vocab_size: Vocabulary size (default: 32000)
        max_seq_len: Maximum sequence length (default: 8192)
        frequencies: Update frequencies for CMS (default: [1, 16, 64, 256])
        expansion_factor: MLP expansion factor (default: 4)
        modification_rank: Rank for self-modification (default: None)
        dropout: Dropout probability (default: 0.1)
        use_memory: Whether to use memory modules (default: True)
        use_cms: Whether to use Continuum Memory System (default: True)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        vocab_size: int = 32000,
        max_seq_len: int = 8192,
        frequencies: Optional[List[int]] = None,
        expansion_factor: int = 4,
        modification_rank: Optional[int] = None,
        dropout: float = 0.1,
        use_memory: bool = True,
        use_cms: bool = True,
    ):
        super().__init__()

        if frequencies is None:
            frequencies = [1, 16, 64, 256]

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.frequencies = frequencies
        self.use_memory = use_memory
        self.use_cms = use_cms

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Optional frequency embedding
        self.freq_embedding = nn.Embedding(len(frequencies), d_model)

        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)

        # HOPE blocks with different frequencies
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            # Assign frequency based on layer depth
            # Early layers: high frequency (fast adaptation)
            # Later layers: low frequency (stable knowledge)
            freq_idx = min(i * len(frequencies) // n_layers, len(frequencies) - 1)
            freq = frequencies[freq_idx]

            self.blocks.append(
                HOPEBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    expansion_factor=expansion_factor,
                    modification_rank=modification_rank,
                    frequency=freq,
                    dropout=dropout,
                )
            )

        # Memory module
        if use_memory:
            self.memory = HOPEMemoryModule(d_model)
        else:
            self.memory = None

        # Continuum Memory System
        if use_cms:
            self.cms = ContinuumMemoryBlock(
                d_model=d_model,
                frequencies=frequencies,
                dropout=dropout,
            )
        else:
            self.cms = None

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding weights
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HOPE.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            labels: Optional labels for computing loss
            return_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary with logits, optional loss, and optional hidden states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

        # Token + position embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.input_norm(x)

        # Store hidden states if requested
        hidden_states = [x] if return_hidden_states else None

        # Process through HOPE blocks
        for i, block in enumerate(self.blocks):
            x = block(x, attention_mask)

            if return_hidden_states:
                hidden_states.append(x)

        # Apply memory module
        if self.memory is not None:
            x = self.memory(x)

        # Apply CMS
        if self.cms is not None:
            x = self.cms(x)

        # Output
        x = self.output_norm(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = {'logits': logits}
        if loss is not None:
            output['loss'] = loss
        if return_hidden_states:
            output['hidden_states'] = hidden_states

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(generated[:, -self.max_seq_len:])
                logits = outputs['logits'][:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: Exclude embedding parameters (default: True)

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params


class HOPESmall(HOPE):
    """Small HOPE model (~125M parameters)."""

    def __init__(self, vocab_size: int = 32000, **kwargs):
        super().__init__(
            d_model=768,
            n_heads=12,
            n_layers=12,
            vocab_size=vocab_size,
            **kwargs,
        )


class HOPEMedium(HOPE):
    """Medium HOPE model (~350M parameters)."""

    def __init__(self, vocab_size: int = 32000, **kwargs):
        super().__init__(
            d_model=1024,
            n_heads=16,
            n_layers=24,
            vocab_size=vocab_size,
            **kwargs,
        )


class HOPELarge(HOPE):
    """Large HOPE model (~760M parameters)."""

    def __init__(self, vocab_size: int = 32000, **kwargs):
        super().__init__(
            d_model=1280,
            n_heads=20,
            n_layers=36,
            vocab_size=vocab_size,
            **kwargs,
        )
