"""
Self-Modifying Layers for Nested Learning.

These layers implement the "self-modification" capability from the HOPE architecture,
where parameters can change during the forward pass based on the input. This creates
an unbounded in-context learning capability through self-referential optimization.

Key insight: Traditional layers have static weights W that are only updated during
backpropagation. Self-modifying layers compute dynamic weight updates ΔW based on
the input and apply them during the forward pass:

    output = (W + ΔW(input)) @ input

This allows the model to adapt its computation on-the-fly, similar to how attention
creates input-dependent transformations, but applied to the weights themselves.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


class SelfModifyingLinear(nn.Module):
    """
    Linear layer with self-modifying weights.

    The weight modification is computed using a delta rule update:
        ΔW = η * outer(error, input)

    where the error signal is computed from the input and a learned
    "target" representation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        modification_rank: Rank of the weight modification (default: None = full rank)
        modification_strength: Strength of weight modification (default: 0.1)
        use_bias: Whether to use bias (default: True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        modification_rank: Optional[int] = None,
        modification_strength: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.modification_strength = modification_strength

        if modification_rank is None:
            modification_rank = min(in_features, out_features) // 4
        self.modification_rank = modification_rank

        # Base weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Delta rule components
        # These learn to predict the "error" that drives weight modification
        self.error_predictor = nn.Sequential(
            nn.Linear(in_features, modification_rank),
            nn.GELU(),
            nn.Linear(modification_rank, out_features),
        )

        # Low-rank modification matrices
        self.mod_down = nn.Linear(in_features, modification_rank, bias=False)
        self.mod_up = nn.Linear(modification_rank, out_features, bias=False)

        # Learnable modification strength
        self.mod_gate = nn.Parameter(torch.tensor(modification_strength))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.mod_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-modification.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Base linear transformation
        base_output = F.linear(x, self.weight, self.bias)

        # Compute error signal for delta rule
        error = self.error_predictor(x)  # [..., out_features]

        # Low-rank weight modification
        # ΔW effectively = mod_up @ mod_down
        mod_hidden = self.mod_down(x)  # [..., rank]
        modification = self.mod_up(mod_hidden)  # [..., out_features]

        # Gate the modification
        gate = torch.sigmoid(self.mod_gate)

        # Combine base output with input-dependent modification
        # This is equivalent to applying (W + ΔW) @ x
        output = base_output + gate * (error * modification)

        return output

    def get_effective_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the effective weight matrix for a given input.

        This is useful for analysis and debugging.
        """
        # Compute average modification over batch
        with torch.no_grad():
            mod_hidden = self.mod_down(x)
            # Average over batch and sequence dimensions
            avg_hidden = mod_hidden.mean(dim=list(range(len(x.shape) - 1)))
            delta_W = torch.outer(self.mod_up.weight.mean(dim=0), avg_hidden)

        return self.weight + torch.sigmoid(self.mod_gate) * delta_W


class SelfModifyingAttention(nn.Module):
    """
    Multi-head attention with self-modifying Query/Key/Value projections.

    This extends standard attention by allowing the Q/K/V projection weights
    to be modified based on the input, enabling more adaptive attention patterns.

    The modification follows the Titans/HOPE approach where "surprise" (prediction
    error) drives weight updates during the forward pass.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        modification_rank: Rank of weight modifications (default: d_model // 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        modification_rank: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if modification_rank is None:
            modification_rank = d_model // 8
        self.modification_rank = modification_rank

        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Self-modification components for each projection
        self.q_modifier = WeightModifier(d_model, d_model, modification_rank)
        self.k_modifier = WeightModifier(d_model, d_model, modification_rank)
        self.v_modifier = WeightModifier(d_model, d_model, modification_rank)

        # Surprise computation (how unexpected is the input?)
        self.surprise_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with self-modifying projections.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Compute surprise for gating modifications
        surprise = self.surprise_net(x)  # [batch, seq, 1]

        # Apply self-modifying Q/K/V projections
        q = self.q_proj(x) + surprise * self.q_modifier(x)
        k = self.k_proj(x) + surprise * self.k_modifier(x)
        v = self.v_proj(x) + surprise * self.v_modifier(x)

        # Reshape for multi-head attention
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attn_output = torch.matmul(attention_weights, v)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')

        # Output projection
        output = self.o_proj(attn_output)

        if return_attention:
            return output, attention_weights
        return output, None


class WeightModifier(nn.Module):
    """
    Computes low-rank weight modifications based on input.

    This is a helper module that computes ΔW for self-modifying layers
    using a low-rank decomposition for efficiency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Low-rank factors for weight modification
        self.factor_in = nn.Linear(in_features, rank, bias=False)
        self.factor_out = nn.Linear(rank, out_features, bias=False)

        # Initialize to near-zero for stability
        nn.init.normal_(self.factor_in.weight, std=0.01)
        nn.init.zeros_(self.factor_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute modification: equivalent to (factor_out @ factor_in) @ x"""
        hidden = self.factor_in(x)
        return self.factor_out(hidden)


class SelfModifyingMLP(nn.Module):
    """
    MLP with self-modifying weights.

    Both the up-projection and down-projection can be modified based on
    the input, allowing the MLP to adapt its transformation on-the-fly.

    Args:
        d_model: Model dimension
        expansion_factor: MLP expansion factor (default: 4)
        modification_rank: Rank of weight modifications (default: d_model // 8)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: "gelu")
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        modification_rank: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = d_model * expansion_factor

        if modification_rank is None:
            modification_rank = d_model // 8
        self.modification_rank = modification_rank

        # Main MLP
        self.up_proj = nn.Linear(d_model, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, d_model)

        # Self-modification
        self.up_modifier = WeightModifier(d_model, self.hidden_dim, modification_rank)
        self.down_modifier = WeightModifier(self.hidden_dim, d_model, modification_rank)

        # Surprise/gate computation
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-modifying weights.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Compute modification gate
        g = self.gate(x)

        # Up projection with modification
        h = self.up_proj(x) + g * self.up_modifier(x)
        h = self.activation(h)
        h = self.dropout(h)

        # Down projection with modification (gate computed on hidden state)
        g_hidden = torch.sigmoid(h.mean(dim=-1, keepdim=True))
        out = self.down_proj(h) + g_hidden * self.down_modifier(h)
        out = self.dropout(out)

        return out


class DeltaRuleLayer(nn.Module):
    """
    Layer that implements explicit delta rule weight updates during forward pass.

    This layer maintains a "target" representation and updates weights to
    minimize the prediction error between output and target. This is the
    most explicit implementation of the Nested Learning insight that
    all layers learn by compressing their context flow.

    Update rule during forward:
        prediction = W @ x
        error = target(x) - prediction
        ΔW = η * outer(error, x)
        output = (W + ΔW) @ x

    Args:
        in_features: Input dimension
        out_features: Output dimension
        learning_rate: Delta rule learning rate (default: 0.1)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate

        # Main weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Target predictor (what the output "should" be)
        self.target_net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
        )

        # Learnable learning rate
        self.lr = nn.Parameter(torch.tensor(learning_rate))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with delta rule weight modification.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Get batch dimensions
        batch_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.in_features)

        # Compute prediction with current weights
        prediction = F.linear(flat_x, self.weight, self.bias)

        # Compute target
        target = self.target_net(flat_x)

        # Compute error
        error = target - prediction.detach()

        # Compute weight modification via delta rule
        # ΔW = η * mean(outer(error, x))
        # This is averaged over the batch for stability
        lr = torch.sigmoid(self.lr)
        delta_W = lr * torch.einsum('bi,bj->ij', error, flat_x) / flat_x.shape[0]

        # Apply modified weights
        effective_weight = self.weight + delta_W
        output = F.linear(flat_x, effective_weight, self.bias)

        # Reshape to original batch shape
        return output.reshape(*batch_shape, self.out_features)

    def get_error_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Get the prediction error magnitude for monitoring."""
        with torch.no_grad():
            prediction = F.linear(x, self.weight, self.bias)
            target = self.target_net(x)
            return (target - prediction).abs().mean()
