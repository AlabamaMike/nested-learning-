"""
Training Utilities for Nested Learning.

This module provides training infrastructure for the multi-frequency update
scheme used in Nested Learning. Key components include:

1. MultiFrequencyTrainer: Handles different update frequencies for different layers
2. GradientAccumulator: Accumulates gradients over multiple steps
3. Frequency scheduling utilities
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class FrequencyConfig:
    """Configuration for a frequency level."""

    frequency: int  # Update every N steps
    learning_rate: float  # Learning rate for this frequency
    warmup_steps: int = 0  # Warmup steps for this frequency
    weight_decay: float = 0.0  # Weight decay for this frequency


def create_frequency_schedule(
    frequencies: List[int],
    base_lr: float = 1e-4,
    lr_decay_factor: float = 0.5,
) -> List[FrequencyConfig]:
    """
    Create a frequency schedule with appropriate learning rates.

    Higher frequency (faster updating) layers typically need lower learning
    rates for stability, while lower frequency layers can use higher rates.

    Args:
        frequencies: List of update frequencies
        base_lr: Base learning rate
        lr_decay_factor: Factor to decay LR for higher frequencies

    Returns:
        List of FrequencyConfig objects
    """
    configs = []
    for i, freq in enumerate(frequencies):
        # Lower frequencies get higher learning rates
        lr = base_lr * (lr_decay_factor ** i)
        warmup = freq * 10  # Warmup proportional to frequency

        configs.append(FrequencyConfig(
            frequency=freq,
            learning_rate=lr,
            warmup_steps=warmup,
        ))

    return configs


class GradientAccumulator:
    """
    Accumulates gradients over multiple steps for a parameter group.

    This is essential for implementing multi-frequency updates where
    some layers only update every N steps.
    """

    def __init__(self, params: List[torch.nn.Parameter], frequency: int = 1):
        self.params = list(params)
        self.frequency = frequency
        self.step_count = 0

        # Initialize gradient accumulators
        self.accumulated_grads: Dict[int, torch.Tensor] = {}
        for i, p in enumerate(self.params):
            if p.requires_grad:
                self.accumulated_grads[i] = torch.zeros_like(p.data)

    def accumulate(self):
        """Accumulate current gradients."""
        for i, p in enumerate(self.params):
            if p.grad is not None and i in self.accumulated_grads:
                self.accumulated_grads[i].add_(p.grad.data)

        self.step_count += 1

    def should_update(self) -> bool:
        """Check if we should apply accumulated gradients."""
        return self.step_count % self.frequency == 0

    def get_accumulated_grads(self) -> Dict[int, torch.Tensor]:
        """Get accumulated gradients (averaged)."""
        result = {}
        for i, grad in self.accumulated_grads.items():
            result[i] = grad / self.frequency
        return result

    def apply_and_reset(self, optimizer: Optimizer, lr_scale: float = 1.0):
        """Apply accumulated gradients and reset."""
        if not self.should_update():
            return

        # Set accumulated grads as the parameter gradients
        for i, p in enumerate(self.params):
            if i in self.accumulated_grads:
                if p.grad is None:
                    p.grad = self.accumulated_grads[i].clone() / self.frequency
                else:
                    p.grad.data = self.accumulated_grads[i].clone() / self.frequency

                # Scale learning rate if needed
                if lr_scale != 1.0:
                    p.grad.data.mul_(lr_scale)

        # Reset accumulators
        for i in self.accumulated_grads:
            self.accumulated_grads[i].zero_()

    def reset(self):
        """Reset all accumulators."""
        for i in self.accumulated_grads:
            self.accumulated_grads[i].zero_()
        self.step_count = 0


class MultiFrequencyTrainer:
    """
    Trainer that handles multi-frequency parameter updates.

    This trainer implements the core Nested Learning insight that different
    layers should update at different frequencies:
    - High-frequency layers (early layers): React quickly to new data
    - Low-frequency layers (later layers): Consolidate stable knowledge

    Args:
        model: The model to train
        optimizer: Base optimizer (will be wrapped)
        frequencies: List of update frequencies
        frequency_assignments: Dict mapping module names to frequency indices
        scheduler: Optional learning rate scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        frequencies: Optional[List[int]] = None,
        frequency_assignments: Optional[Dict[str, int]] = None,
        scheduler: Optional[Any] = None,
    ):
        if frequencies is None:
            frequencies = [1, 8, 64, 256]

        self.model = model
        self.base_optimizer = optimizer
        self.frequencies = frequencies
        self.scheduler = scheduler
        self.global_step = 0

        # Create gradient accumulators for each frequency
        self.accumulators: Dict[int, GradientAccumulator] = {}

        # Assign parameters to frequencies
        if frequency_assignments is None:
            self._auto_assign_frequencies()
        else:
            self._assign_frequencies(frequency_assignments)

    def _auto_assign_frequencies(self):
        """Automatically assign frequencies based on module depth."""
        # Collect all modules with parameters
        modules_with_params = []
        for name, module in self.model.named_modules():
            params = list(module.parameters(recurse=False))
            if params:
                modules_with_params.append((name, module, params))

        if not modules_with_params:
            return

        # Assign frequencies based on depth
        n_modules = len(modules_with_params)
        n_freqs = len(self.frequencies)

        for i, (name, module, params) in enumerate(modules_with_params):
            # Earlier modules get higher frequencies (lower indices)
            freq_idx = min(i * n_freqs // n_modules, n_freqs - 1)
            freq = self.frequencies[freq_idx]

            if freq not in self.accumulators:
                self.accumulators[freq] = GradientAccumulator(params, freq)
            else:
                # Add to existing accumulator
                self.accumulators[freq].params.extend(params)
                for j, p in enumerate(params):
                    if p.requires_grad:
                        idx = len(self.accumulators[freq].accumulated_grads)
                        self.accumulators[freq].accumulated_grads[idx] = (
                            torch.zeros_like(p.data)
                        )

    def _assign_frequencies(self, assignments: Dict[str, int]):
        """Assign frequencies based on provided mapping."""
        for name, module in self.model.named_modules():
            if name in assignments:
                freq_idx = assignments[name]
                freq = self.frequencies[freq_idx]
                params = list(module.parameters(recurse=False))

                if params:
                    if freq not in self.accumulators:
                        self.accumulators[freq] = GradientAccumulator(params, freq)
                    else:
                        self.accumulators[freq].params.extend(params)

    def step(self, loss: torch.Tensor):
        """
        Perform a single training step.

        This handles:
        1. Backward pass
        2. Gradient accumulation at each frequency
        3. Parameter updates at appropriate frequencies

        Args:
            loss: The loss tensor to backpropagate
        """
        # Backward pass
        loss.backward()

        # Accumulate gradients at each frequency
        for freq, accumulator in self.accumulators.items():
            accumulator.accumulate()

            # Apply updates if this frequency should update
            if accumulator.should_update():
                accumulator.apply_and_reset(self.base_optimizer)

        # Base optimizer step (for frequency-1 params or non-assigned params)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        # Update scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

    def zero_grad(self):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        return {
            'global_step': self.global_step,
            'optimizer_state': self.base_optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        self.global_step = state_dict['global_step']
        self.base_optimizer.load_state_dict(state_dict['optimizer_state'])
        if self.scheduler and state_dict['scheduler_state']:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])


class CosineAnnealingWithWarmup:
    """
    Learning rate scheduler with warmup and cosine annealing.

    Implements:
    - Linear warmup from 0 to base_lr
    - Cosine annealing from base_lr to min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * scale

    def state_dict(self) -> Dict[str, Any]:
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']


class NestedLearningCallback:
    """
    Callback for monitoring Nested Learning training.

    Tracks metrics specific to Nested Learning:
    - Per-frequency gradient magnitudes
    - Surprise levels
    - Memory utilization
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.metrics: Dict[str, List[float]] = {
            'loss': [],
            'grad_norm': [],
            'surprise': [],
        }

    def on_step(
        self,
        loss: float,
        model: nn.Module,
        trainer: Optional[MultiFrequencyTrainer] = None,
    ):
        """Called after each training step."""
        self.step_count += 1
        self.metrics['loss'].append(loss)

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.metrics['grad_norm'].append(total_norm ** 0.5)

        # Log periodically
        if self.step_count % self.log_interval == 0:
            avg_loss = sum(self.metrics['loss'][-self.log_interval:]) / self.log_interval
            avg_grad = sum(self.metrics['grad_norm'][-self.log_interval:]) / self.log_interval

            print(
                f"Step {self.step_count}: "
                f"loss={avg_loss:.4f}, grad_norm={avg_grad:.4f}"
            )

    def get_metrics(self) -> Dict[str, List[float]]:
        """Return all collected metrics."""
        return self.metrics


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: Optimizer,
    scheduler: Optional[Any] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Single training step utility function.

    Args:
        model: The model to train
        batch: Dictionary with 'input_ids' and optionally 'labels'
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Loss value
    """
    model.train()

    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch.get('labels', batch['input_ids']),
    )
    loss = outputs['loss']

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Scheduler step
    if scheduler is not None:
        scheduler.step()

    return loss.item()


def evaluate(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']

            batch_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
    }
