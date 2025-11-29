"""
Delta Rule Momentum Optimizer.

This optimizer implements the delta rule view of momentum from Nested Learning,
which reformulates momentum as an associative memory using L2 regression loss
instead of dot product similarity.

Standard momentum uses:
    m_{t+1} = α·m_t − η·∇L

Delta Rule momentum uses:
    m_{t+1} = m_t - η·(m_t - ∇L)

This is equivalent to an online learning rule where the momentum vector
"learns" to predict the gradient using L2 loss, making it more robust to
noisy or imperfect gradients.

The key insight is that momentum can be viewed as associative memory that
compresses gradient history, and using L2 loss provides better compression
than simple exponential averaging.
"""

from typing import Optional, Callable, Iterable, List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


class DeltaRuleMomentum(Optimizer):
    """
    Delta Rule Momentum Optimizer.

    This optimizer views momentum as an associative memory trained with the
    delta rule (L2 regression). The momentum state learns to predict gradients,
    and the prediction error drives both the parameter update and momentum update.

    Update rules:
        error_t = gradient_t - momentum_t
        momentum_{t+1} = momentum_t + beta * error_t
        W_{t+1} = W_t - lr * (momentum_{t+1} + surprise_weight * error_t)

    The "surprise" term (error_t) allows the optimizer to react quickly to
    unexpected gradients while maintaining smooth updates.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        beta: Delta rule learning rate for momentum (default: 0.1)
        surprise_weight: Weight for the surprise/error term (default: 0.5)
        weight_decay: L2 regularization (default: 0)
        use_surprise: Whether to include surprise term in update (default: True)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.1,
        surprise_weight: float = 0.5,
        weight_decay: float = 0,
        use_surprise: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        if surprise_weight < 0.0:
            raise ValueError(f"Invalid surprise weight: {surprise_weight}")

        defaults = dict(
            lr=lr,
            beta=beta,
            surprise_weight=surprise_weight,
            weight_decay=weight_decay,
            use_surprise=use_surprise,
        )
        super().__init__(params, defaults)

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
            beta = group['beta']
            surprise_weight = group['surprise_weight']
            weight_decay = group['weight_decay']
            use_surprise = group['use_surprise']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize state on first step
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['step'] = 0

                state['step'] += 1
                momentum = state['momentum']

                # Compute prediction error (surprise)
                error = grad - momentum

                # Update momentum with delta rule
                momentum.add_(error, alpha=beta)

                # Compute update: base momentum + weighted surprise
                if use_surprise:
                    update = momentum + surprise_weight * error
                else:
                    update = momentum

                # Apply update to parameters
                p.data.add_(update, alpha=-lr)

        return loss

    def get_surprise_magnitude(self) -> float:
        """
        Get the average surprise magnitude across all parameters.

        This is useful for monitoring how "surprising" recent gradients are
        to the optimizer, which can indicate distribution shift.
        """
        total_surprise = 0.0
        total_params = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'momentum' in state:
                    error = p.grad - state['momentum']
                    total_surprise += error.abs().mean().item()
                    total_params += 1

        return total_surprise / max(total_params, 1)


class AdaptiveDeltaRuleMomentum(Optimizer):
    """
    Adaptive Delta Rule Momentum with learned surprise sensitivity.

    This extends DeltaRuleMomentum by making the beta and surprise_weight
    parameters adaptive based on the gradient statistics. Parameters that
    show high variance in gradients get higher beta (faster adaptation),
    while stable parameters get lower beta.

    This creates a multi-timescale learning system where different parameters
    operate at different frequencies based on their gradient dynamics.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        beta_min: Minimum beta value (default: 0.01)
        beta_max: Maximum beta value (default: 0.5)
        surprise_min: Minimum surprise weight (default: 0.1)
        surprise_max: Maximum surprise weight (default: 1.0)
        ema_decay: Decay for gradient variance estimation (default: 0.99)
        weight_decay: L2 regularization (default: 0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta_min: float = 0.01,
        beta_max: float = 0.5,
        surprise_min: float = 0.1,
        surprise_max: float = 1.0,
        ema_decay: float = 0.99,
        weight_decay: float = 0,
    ):
        defaults = dict(
            lr=lr,
            beta_min=beta_min,
            beta_max=beta_max,
            surprise_min=surprise_min,
            surprise_max=surprise_max,
            ema_decay=ema_decay,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_min = group['beta_min']
            beta_max = group['beta_max']
            surprise_min = group['surprise_min']
            surprise_max = group['surprise_max']
            ema_decay = group['ema_decay']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize state on first step
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['grad_mean'] = torch.zeros_like(p.data)
                    state['grad_var'] = torch.ones_like(p.data)
                    state['step'] = 0

                state['step'] += 1
                momentum = state['momentum']
                grad_mean = state['grad_mean']
                grad_var = state['grad_var']

                # Update gradient statistics (running mean and variance)
                grad_mean.mul_(ema_decay).add_(grad, alpha=1 - ema_decay)
                grad_sq = (grad - grad_mean) ** 2
                grad_var.mul_(ema_decay).add_(grad_sq, alpha=1 - ema_decay)

                # Compute normalized variance (coefficient of variation)
                grad_std = (grad_var + 1e-8).sqrt()
                cv = (grad_std / (grad_mean.abs() + 1e-8)).mean().item()
                cv = min(max(cv, 0), 1)  # Clamp to [0, 1]

                # Adaptive beta: higher variance -> higher beta
                beta = beta_min + (beta_max - beta_min) * cv

                # Adaptive surprise weight: higher variance -> higher surprise weight
                surprise_weight = surprise_min + (surprise_max - surprise_min) * cv

                # Compute prediction error (surprise)
                error = grad - momentum

                # Update momentum with delta rule
                momentum.add_(error, alpha=beta)

                # Compute update with adaptive surprise
                update = momentum + surprise_weight * error

                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss
