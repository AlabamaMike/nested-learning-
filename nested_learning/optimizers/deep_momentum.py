"""
Deep Momentum Gradient Descent (DMGD).

This implements the core insight from Nested Learning: gradient descent with momentum
is a two-level nested optimization problem where the momentum term is an associative
memory that learns to compress the history of gradients.

Standard momentum update:
    m_{t+1} = α·m_t − η·∇L(W_t; x_t)
    W_{t+1} = W_t + m_{t+1}

Reformulated as optimization (key insight):
    m_{t+1} = arg min_m ⟨m, ∇L(W_t; x_t)⟩ + η||m - m_t||²

This reveals momentum as a learned associative memory mapping gradient patterns
to update directions. DMGD replaces the linear memory with an MLP for richer
compression of gradient statistics through nonlinearities.
"""

import math
from typing import Dict, Any, Optional, Callable, Iterable, List

import torch
import torch.nn as nn
from torch.optim import Optimizer


class MomentumMLP(nn.Module):
    """
    An MLP that serves as the momentum memory module.

    This replaces the linear momentum term with a learnable nonlinear function
    that can capture complex gradient dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Build the MLP
        layers: List[nn.Module] = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            current_dim = hidden_dim

        # Output layer projects back to gradient dimension
        layers.append(nn.Linear(current_dim, input_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepMomentumGD(Optimizer):
    """
    Deep Momentum Gradient Descent optimizer.

    This optimizer implements the Nested Learning view of momentum, where the
    momentum term is replaced by an MLP that learns to compress gradient history
    into effective update directions.

    The update rule becomes:
        u_t = concatenate(gradient_t, momentum_state_t)
        m_{t+1} = momentum_mlp(u_t)
        W_{t+1} = W_t - lr * m_{t+1}

    The momentum MLP is trained via an internal loss that measures how well
    the predicted update correlates with the true gradient direction.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum_lr: Learning rate for the momentum MLP (default: 1e-4)
        momentum_decay: Decay factor for momentum state (default: 0.9)
        hidden_dim: Hidden dimension of momentum MLP (default: 64)
        num_layers: Number of layers in momentum MLP (default: 2)
        weight_decay: L2 regularization (default: 0)
        use_internal_loss: Whether to train momentum MLP with internal loss (default: True)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum_lr: float = 1e-4,
        momentum_decay: float = 0.9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        weight_decay: float = 0,
        use_internal_loss: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum_lr < 0.0:
            raise ValueError(f"Invalid momentum learning rate: {momentum_lr}")
        if not 0.0 <= momentum_decay < 1.0:
            raise ValueError(f"Invalid momentum decay: {momentum_decay}")

        defaults = dict(
            lr=lr,
            momentum_lr=momentum_lr,
            momentum_decay=momentum_decay,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            weight_decay=weight_decay,
            use_internal_loss=use_internal_loss,
        )
        super().__init__(params, defaults)

        # Initialize momentum MLPs for each parameter group
        self._init_momentum_mlps()

    def _init_momentum_mlps(self):
        """Initialize the momentum MLP for each parameter."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Flatten dimension for MLP input
                    flat_dim = p.numel()

                    # For very large parameters, use a compressed representation
                    if flat_dim > 10000:
                        # Use random projection for compression
                        compressed_dim = min(1000, flat_dim // 10)
                        state['compressed'] = True
                        state['compress_proj'] = torch.randn(
                            flat_dim, compressed_dim, device=p.device
                        ) / math.sqrt(flat_dim)
                        state['decompress_proj'] = torch.randn(
                            compressed_dim, flat_dim, device=p.device
                        ) / math.sqrt(compressed_dim)
                        input_dim = compressed_dim * 2  # gradient + momentum state
                    else:
                        state['compressed'] = False
                        input_dim = flat_dim * 2  # gradient + momentum state

                    # Create momentum MLP
                    state['momentum_mlp'] = MomentumMLP(
                        input_dim=input_dim,
                        hidden_dim=group['hidden_dim'],
                        num_layers=group['num_layers'],
                    ).to(p.device)

                    # Momentum MLP optimizer
                    state['mlp_optimizer'] = torch.optim.Adam(
                        state['momentum_mlp'].parameters(),
                        lr=group['momentum_lr'],
                    )

                    # Initialize momentum state
                    if state['compressed']:
                        state['momentum_state'] = torch.zeros(
                            compressed_dim, device=p.device
                        )
                    else:
                        state['momentum_state'] = torch.zeros_like(p.data.flatten())

    def _compress(self, x: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """Compress a tensor using random projection."""
        if state['compressed']:
            flat = x.flatten()
            return flat @ state['compress_proj']
        return x.flatten()

    def _decompress(self, x: torch.Tensor, state: Dict[str, Any], shape: torch.Size) -> torch.Tensor:
        """Decompress a tensor back to original shape."""
        if state['compressed']:
            flat = x @ state['decompress_proj']
            return flat.view(shape)
        return x.view(shape)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum_decay = group['momentum_decay']
            weight_decay = group['weight_decay']
            use_internal_loss = group['use_internal_loss']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Get compressed gradient
                compressed_grad = self._compress(grad, state)
                momentum_state = state['momentum_state']

                # Concatenate gradient and momentum state as input to MLP
                mlp_input = torch.cat([compressed_grad, momentum_state])

                # Get update from momentum MLP
                momentum_mlp = state['momentum_mlp']

                # Forward pass through momentum MLP
                with torch.enable_grad():
                    mlp_input_param = mlp_input.clone().detach().requires_grad_(True)
                    momentum_output = momentum_mlp(mlp_input_param.unsqueeze(0)).squeeze(0)

                # Extract the update direction (first half of output)
                half_dim = momentum_output.shape[0] // 2
                update = momentum_output[:half_dim]

                # Train the momentum MLP with internal loss if enabled
                if use_internal_loss and p.grad is not None:
                    internal_loss = self._compute_internal_loss(
                        update, compressed_grad, momentum_state
                    )

                    # Update momentum MLP
                    mlp_optimizer = state['mlp_optimizer']
                    mlp_optimizer.zero_grad()
                    internal_loss.backward()
                    mlp_optimizer.step()

                # Update momentum state with decay
                state['momentum_state'] = (
                    momentum_decay * momentum_state +
                    (1 - momentum_decay) * update.detach()
                )

                # Apply update to parameters
                update_full = self._decompress(update.detach(), state, p.shape)
                p.data.add_(update_full, alpha=-lr)

        return loss

    def _compute_internal_loss(
        self,
        update: torch.Tensor,
        gradient: torch.Tensor,
        momentum_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the internal loss for training the momentum MLP.

        The internal loss encourages the update to:
        1. Align with the gradient direction (correlation loss)
        2. Maintain smoothness with previous momentum (stability loss)

        L_internal = ⟨update, gradient⟩ + η||update - momentum_state||²
        """
        # Correlation loss: inner product with gradient (want to minimize)
        correlation_loss = torch.dot(update, gradient)

        # Stability loss: distance from previous momentum
        stability_loss = torch.sum((update - momentum_state) ** 2)

        # Combined loss (minimizing correlation while maintaining stability)
        return correlation_loss + 0.1 * stability_loss


class DeepMomentumGDSimple(Optimizer):
    """
    A simplified version of Deep Momentum GD that uses a single shared MLP
    for all parameters, reducing memory overhead.

    This is more practical for large models where having per-parameter MLPs
    would be prohibitively expensive.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient (default: 0.9)
        hidden_dim: Hidden dimension of the shared momentum MLP (default: 256)
        weight_decay: L2 regularization (default: 0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        hidden_dim: int = 256,
        weight_decay: float = 0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            hidden_dim=hidden_dim,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Shared momentum transformation (learned scale and shift)
        self._step_count = 0
        self._device = None

        # These will be initialized on first step
        self._momentum_transform = None

    def _ensure_init(self, device: torch.device):
        """Initialize the momentum transformation on first use."""
        if self._momentum_transform is None:
            hidden_dim = self.param_groups[0]['hidden_dim']

            # Simple nonlinear momentum transformation
            # Maps: (gradient_stats) -> (scale, shift) for momentum
            self._momentum_transform = nn.Sequential(
                nn.Linear(4, hidden_dim),  # Input: [grad_norm, mom_norm, cos_sim, step]
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),  # Output: [scale, shift]
                nn.Sigmoid(),
            ).to(device)

            # Initialize to approximate standard momentum
            with torch.no_grad():
                self._momentum_transform[-2].weight.fill_(0)
                self._momentum_transform[-2].bias.fill_(0.5)

            self._device = device

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group['lr']
            base_momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                self._ensure_init(p.device)

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']

                # Compute statistics for adaptive momentum
                grad_norm = grad.norm().item()
                buf_norm = buf.norm().item()

                # Cosine similarity between gradient and momentum
                if grad_norm > 1e-8 and buf_norm > 1e-8:
                    cos_sim = (grad * buf).sum().item() / (grad_norm * buf_norm)
                else:
                    cos_sim = 0.0

                # Normalized step count
                step_norm = min(1.0, self._step_count / 1000)

                # Get adaptive momentum parameters
                stats = torch.tensor(
                    [grad_norm, buf_norm, cos_sim, step_norm],
                    device=p.device,
                )

                with torch.enable_grad():
                    params = self._momentum_transform(stats.unsqueeze(0)).squeeze(0)

                scale = params[0].item()
                shift = params[1].item()

                # Adaptive momentum coefficient
                adaptive_momentum = base_momentum * scale + (1 - base_momentum) * shift

                # Update momentum buffer
                buf.mul_(adaptive_momentum).add_(grad, alpha=1 - adaptive_momentum)

                # Apply update
                p.data.add_(buf, alpha=-lr)

        return loss
