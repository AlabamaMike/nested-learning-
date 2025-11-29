#!/usr/bin/env python3
"""
Demo: Comparing Deep Optimizers vs Standard Optimizers

This script compares the performance of Nested Learning optimizers
against standard optimizers on a simple optimization task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List

# Import our optimizers
import sys
sys.path.insert(0, '..')
from nested_learning.optimizers import DeepMomentumGD, DeltaRuleMomentum, PreconditionedMomentum


def create_test_problem(dim: int = 100, seed: int = 42):
    """Create a simple quadratic optimization problem."""
    torch.manual_seed(seed)

    # Create a positive definite matrix for quadratic loss
    A = torch.randn(dim, dim)
    A = A @ A.t() + 0.1 * torch.eye(dim)  # Make positive definite

    b = torch.randn(dim)

    # Optimal solution
    x_opt = torch.linalg.solve(A, b)

    def loss_fn(x):
        return 0.5 * x @ A @ x - b @ x

    return loss_fn, x_opt, A, b


def run_optimizer(
    optimizer_class,
    loss_fn,
    x_init: torch.Tensor,
    n_steps: int = 500,
    **kwargs
) -> List[float]:
    """Run an optimizer and return loss history."""
    x = x_init.clone().requires_grad_(True)

    # Create optimizer
    if optimizer_class in [torch.optim.SGD, torch.optim.Adam]:
        optimizer = optimizer_class([x], **kwargs)
    else:
        optimizer = optimizer_class([x], **kwargs)

    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def compare_optimizers():
    """Compare different optimizers on the test problem."""
    print("=" * 60)
    print("Comparing Nested Learning Optimizers vs Standard Optimizers")
    print("=" * 60)

    # Create test problem
    dim = 50
    loss_fn, x_opt, A, b = create_test_problem(dim)
    x_init = torch.randn(dim)

    optimal_loss = loss_fn(x_opt).item()
    print(f"\nProblem dimension: {dim}")
    print(f"Optimal loss: {optimal_loss:.6f}")

    # Optimizers to compare
    optimizers = {
        'SGD': (torch.optim.SGD, {'lr': 0.01}),
        'SGD+Momentum': (torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        'Adam': (torch.optim.Adam, {'lr': 0.01}),
        'DeltaRuleMomentum': (DeltaRuleMomentum, {'lr': 0.01, 'beta': 0.1}),
    }

    results: Dict[str, List[float]] = {}

    print("\nRunning optimizers...")
    for name, (opt_class, kwargs) in optimizers.items():
        print(f"  {name}...", end=" ", flush=True)
        try:
            losses = run_optimizer(opt_class, loss_fn, x_init, n_steps=300, **kwargs)
            results[name] = losses
            print(f"final loss: {losses[-1]:.6f}")
        except Exception as e:
            print(f"failed: {e}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name, alpha=0.8)

    plt.axhline(y=optimal_loss, color='k', linestyle='--', label='Optimal')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison on Quadratic Problem')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150)
    print(f"\nPlot saved to optimizer_comparison.png")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, losses in results.items():
        final_gap = losses[-1] - optimal_loss
        print(f"{name:25s}: final gap = {final_gap:.6f}")


def demo_deep_momentum_adaptation():
    """
    Show how Deep Momentum GD adapts its momentum based on gradient patterns.
    """
    print("\n" + "=" * 60)
    print("Demo: Deep Momentum Adaptation")
    print("=" * 60)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )

    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)

    # Use Delta Rule Momentum optimizer
    optimizer = DeltaRuleMomentum(model.parameters(), lr=0.01, beta=0.1)

    print("\nTraining with Delta Rule Momentum (showing surprise levels):\n")

    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        output = model(X)
        loss = F.mse_loss(output, y)

        # Backward pass
        loss.backward()

        # Get surprise magnitude
        surprise = optimizer.get_surprise_magnitude()

        # Optimizer step
        optimizer.step()

        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, surprise={surprise:.4f}")

    print("\nKey insight: Surprise decreases as momentum learns gradient patterns!")
    print("Lower surprise = momentum is better at predicting gradients.")


def demo_preconditioned_memory():
    """
    Show how Preconditioned Momentum uses key-value memory.
    """
    print("\n" + "=" * 60)
    print("Demo: Preconditioned Momentum with Key-Value Memory")
    print("=" * 60)

    # Create a simple parameter
    param = torch.randn(20, requires_grad=True)
    target = torch.zeros(20)

    optimizer = PreconditionedMomentum(
        [param],
        lr=0.1,
        memory_size=16,
        alpha=0.7,  # Blend factor between memory and gradient
    )

    print("\nOptimizing with key-value memory:\n")

    for step in range(10):
        optimizer.zero_grad()

        loss = F.mse_loss(param, target)
        loss.backward()

        # The optimizer stores gradient patterns in memory
        # and uses them to precondition future updates
        optimizer.step()

        if step % 2 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    print("\nKey insight: Memory stores gradient patterns and retrieves")
    print("relevant updates for similar gradients, improving convergence!")


if __name__ == "__main__":
    compare_optimizers()
    demo_deep_momentum_adaptation()
    demo_preconditioned_memory()
