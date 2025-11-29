#!/usr/bin/env python3
"""
Demo: Core Nested Learning Concepts

This script demonstrates the key ideas from the Nested Learning paper:
1. Momentum as associative memory
2. Multi-frequency parameter updates
3. Self-modifying layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def demo_momentum_as_memory():
    """
    Demonstrate how momentum can be viewed as an associative memory.

    Key insight: Standard momentum is equivalent to solving:
        m_{t+1} = arg min_m ⟨m, ∇L⟩ + η||m - m_t||²

    This shows momentum is "learning" to predict good update directions
    by compressing gradient history.
    """
    print("=" * 60)
    print("Demo 1: Momentum as Associative Memory")
    print("=" * 60)

    # Create a simple optimization problem
    torch.manual_seed(42)
    target = torch.randn(10)

    # Parameter to optimize
    w = torch.randn(10, requires_grad=True)

    # Standard momentum
    momentum = torch.zeros(10)
    alpha = 0.9  # Momentum coefficient
    lr = 0.01

    print("\nOptimizing with momentum (showing momentum learns gradient patterns):\n")

    for step in range(20):
        # Compute gradient
        loss = F.mse_loss(w, target)
        grad = torch.autograd.grad(loss, w)[0]

        # Standard momentum update
        momentum = alpha * momentum - lr * grad

        # The key insight: momentum is compressing gradient history
        # It learns to predict the direction we should move

        # Correlation between momentum and current gradient
        correlation = F.cosine_similarity(momentum, -grad, dim=0).item()

        if step % 5 == 0:
            print(f"Step {step:2d}: loss={loss.item():.4f}, "
                  f"momentum-gradient correlation={correlation:.4f}")

        # Update parameter
        with torch.no_grad():
            w.add_(momentum)

    print("\nKey insight: High correlation means momentum has 'learned' ")
    print("the gradient pattern - it predicts where we should go!")


def demo_multi_frequency_updates():
    """
    Demonstrate multi-frequency parameter updates.

    Key insight: Different layers can update at different frequencies,
    creating a hierarchy from fast adaptation to slow consolidation.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Frequency Parameter Updates")
    print("=" * 60)

    # Create layers with different update frequencies
    frequencies = [1, 4, 16]  # Steps between updates

    class MultiFreqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fast_layer = nn.Linear(10, 10)   # Updates every step
            self.medium_layer = nn.Linear(10, 10)  # Updates every 4 steps
            self.slow_layer = nn.Linear(10, 10)   # Updates every 16 steps

        def forward(self, x):
            x = F.relu(self.fast_layer(x))
            x = F.relu(self.medium_layer(x))
            return self.slow_layer(x)

    model = MultiFreqModel()
    target = torch.randn(10)

    # Track gradient accumulators for each frequency
    grad_accum = {
        'fast': torch.zeros_like(model.fast_layer.weight.data),
        'medium': torch.zeros_like(model.medium_layer.weight.data),
        'slow': torch.zeros_like(model.slow_layer.weight.data),
    }

    update_counts = {'fast': 0, 'medium': 0, 'slow': 0}
    lr = 0.01

    print("\nTraining with different update frequencies:\n")

    for step in range(17):
        x = torch.randn(1, 10)
        output = model(x)
        loss = F.mse_loss(output.squeeze(), target)
        loss.backward()

        # Accumulate gradients
        grad_accum['fast'] += model.fast_layer.weight.grad.data
        grad_accum['medium'] += model.medium_layer.weight.grad.data
        grad_accum['slow'] += model.slow_layer.weight.grad.data

        updates_this_step = []

        # Fast layer: update every step
        if step % frequencies[0] == 0:
            model.fast_layer.weight.data -= lr * grad_accum['fast']
            grad_accum['fast'].zero_()
            update_counts['fast'] += 1
            updates_this_step.append('fast')

        # Medium layer: update every 4 steps
        if step % frequencies[1] == 0:
            model.medium_layer.weight.data -= lr * grad_accum['medium'] / frequencies[1]
            grad_accum['medium'].zero_()
            update_counts['medium'] += 1
            updates_this_step.append('medium')

        # Slow layer: update every 16 steps
        if step % frequencies[2] == 0:
            model.slow_layer.weight.data -= lr * grad_accum['slow'] / frequencies[2]
            grad_accum['slow'].zero_()
            update_counts['slow'] += 1
            updates_this_step.append('slow')

        model.zero_grad()

        if updates_this_step:
            print(f"Step {step:2d}: Updated layers: {updates_this_step}")

    print(f"\nTotal updates - Fast: {update_counts['fast']}, "
          f"Medium: {update_counts['medium']}, Slow: {update_counts['slow']}")
    print("\nKey insight: Slow layers consolidate knowledge over longer periods!")


def demo_self_modification():
    """
    Demonstrate self-modifying layers.

    Key insight: Layers can modify their own weights during forward pass,
    enabling unbounded in-context learning.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Self-Modifying Layers")
    print("=" * 60)

    class SelfModifyingLayer(nn.Module):
        """A layer that modifies its weights based on input."""

        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(dim))

            # Modification network
            self.mod_net = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim * dim),
            )
            # Initialize to produce small modifications
            nn.init.zeros_(self.mod_net[-1].weight)
            nn.init.zeros_(self.mod_net[-1].bias)

        def forward(self, x):
            # Standard linear transform
            base_output = F.linear(x, self.weight, self.bias)

            # Compute input-dependent weight modification
            delta_w = self.mod_net(x).view(-1, x.size(-1), x.size(-1))

            # Apply modified weights
            # This is like (W + ΔW(x)) @ x
            modified_output = torch.bmm(delta_w, x.unsqueeze(-1)).squeeze(-1)

            return base_output + 0.1 * modified_output

    layer = SelfModifyingLayer(10)

    print("\nShowing how different inputs cause different weight modifications:\n")

    # Different inputs cause different effective weights
    for i in range(3):
        x = torch.randn(1, 10)

        # Get base output
        base_out = F.linear(x, layer.weight, layer.bias)

        # Get modified output
        full_out = layer(x)

        # Compute effective modification magnitude
        mod_magnitude = (full_out - base_out).norm().item()

        print(f"Input {i+1}: modification magnitude = {mod_magnitude:.4f}")

    print("\nKey insight: The layer adapts its computation based on each input!")
    print("This enables unbounded in-context learning.")


def demo_surprise_based_learning():
    """
    Demonstrate surprise-based memory updates.

    Key insight: Memory should prioritize storing "surprising" experiences
    (high prediction error) rather than routine ones.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Surprise-Based Memory")
    print("=" * 60)

    class SurpriseMemory:
        """Simple surprise-based memory."""

        def __init__(self, capacity=5):
            self.capacity = capacity
            self.memory = []
            self.surprise_scores = []

        def predict(self, x):
            """Predict value for input based on stored memories."""
            if not self.memory:
                return torch.zeros_like(x)

            # Find most similar memory
            similarities = [F.cosine_similarity(x, m[0], dim=0).item()
                          for m in self.memory]
            best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            return self.memory[best_idx][1]

        def store(self, key, value, surprise):
            """Store only if surprise exceeds threshold."""
            if len(self.memory) < self.capacity:
                self.memory.append((key, value))
                self.surprise_scores.append(surprise)
            else:
                # Replace least surprising memory
                min_idx = min(range(len(self.surprise_scores)),
                            key=lambda i: self.surprise_scores[i])
                if surprise > self.surprise_scores[min_idx]:
                    self.memory[min_idx] = (key, value)
                    self.surprise_scores[min_idx] = surprise

    memory = SurpriseMemory(capacity=5)

    print("\nStoring experiences based on surprise level:\n")

    # Generate some "experiences"
    for i in range(10):
        key = torch.randn(10)
        true_value = torch.randn(10)

        # Predict
        predicted = memory.predict(key)

        # Compute surprise (prediction error)
        surprise = (true_value - predicted).norm().item()

        # Store based on surprise
        old_len = len(memory.memory)
        memory.store(key.clone(), true_value.clone(), surprise)
        stored = len(memory.memory) > old_len or old_len == memory.capacity

        print(f"Experience {i+1}: surprise={surprise:.2f}, "
              f"stored={stored}, memory_size={len(memory.memory)}")

    print("\nKey insight: Memory focuses on unexpected/important patterns!")
    print(f"Final memory contains {len(memory.memory)} high-surprise experiences.")


if __name__ == "__main__":
    demo_momentum_as_memory()
    demo_multi_frequency_updates()
    demo_self_modification()
    demo_surprise_based_learning()

    print("\n" + "=" * 60)
    print("Summary: Nested Learning Key Concepts")
    print("=" * 60)
    print("""
1. MOMENTUM AS MEMORY: Gradient descent with momentum is actually
   an associative memory learning to predict good update directions.

2. MULTI-FREQUENCY UPDATES: Different layers update at different
   rates, from fast adaptation to slow knowledge consolidation.

3. SELF-MODIFICATION: Layers can modify their own weights based on
   input, enabling unbounded in-context learning.

4. SURPRISE-BASED STORAGE: Memory prioritizes unexpected experiences,
   focusing learning on what matters most.

Together, these create the Nested Learning paradigm - a unified view
of ML models as nested optimization problems!
""")
