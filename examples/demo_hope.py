#!/usr/bin/env python3
"""
Demo: HOPE Model Architecture

This script demonstrates the HOPE (Hierarchical Optimizing Processing Ensemble)
model from the Nested Learning paper.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')

from nested_learning import HOPE, DeepMomentumGD
from nested_learning.utils.training import MultiFrequencyTrainer, train_step


def demo_hope_architecture():
    """Demonstrate the HOPE model architecture."""
    print("=" * 60)
    print("Demo: HOPE Model Architecture")
    print("=" * 60)

    # Create a small HOPE model
    model = HOPE(
        d_model=256,
        n_heads=4,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=512,
        frequencies=[1, 8, 32],
        use_memory=True,
        use_cms=True,
    )

    print(f"\nHOPE Model Configuration:")
    print(f"  - Model dimension: {model.d_model}")
    print(f"  - Attention heads: {model.n_heads}")
    print(f"  - Layers: {model.n_layers}")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Frequencies: {model.frequencies}")
    print(f"  - Parameters: {model.get_num_params():,}")

    # Generate dummy input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, return_hidden_states=True)

    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")


def demo_self_modification():
    """Show self-modification in action."""
    print("\n" + "=" * 60)
    print("Demo: Self-Modification in HOPE")
    print("=" * 60)

    model = HOPE(
        d_model=128,
        n_heads=2,
        n_layers=2,
        vocab_size=100,
        max_seq_len=128,
        frequencies=[1, 4],
        use_memory=False,
        use_cms=False,
    )

    # Access a self-modifying attention layer
    attn_layer = model.blocks[0].attention

    print("\nExamining self-modifying attention layer:")
    print(f"  - Model dimension: {attn_layer.d_model}")
    print(f"  - Number of heads: {attn_layer.n_heads}")
    print(f"  - Modification rank: {attn_layer.modification_rank}")

    # Show how different inputs produce different modifications
    x1 = torch.randn(1, 10, model.d_model)
    x2 = torch.randn(1, 10, model.d_model)

    with torch.no_grad():
        # Compute surprise for each input
        surprise1 = attn_layer.surprise_net(x1).mean().item()
        surprise2 = attn_layer.surprise_net(x2).mean().item()

    print(f"\nSurprise levels for different inputs:")
    print(f"  - Input 1: {surprise1:.4f}")
    print(f"  - Input 2: {surprise2:.4f}")
    print("\nHigher surprise = more weight modification applied!")


def demo_multi_frequency_training():
    """Demonstrate multi-frequency training."""
    print("\n" + "=" * 60)
    print("Demo: Multi-Frequency Training")
    print("=" * 60)

    model = HOPE(
        d_model=128,
        n_heads=2,
        n_layers=4,
        vocab_size=100,
        max_seq_len=64,
        frequencies=[1, 4, 16],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = MultiFrequencyTrainer(
        model=model,
        optimizer=optimizer,
        frequencies=[1, 4, 16],
    )

    print("\nTraining with multi-frequency updates:")
    print("  - Frequency 1: Updates every step (early layers)")
    print("  - Frequency 4: Updates every 4 steps (middle layers)")
    print("  - Frequency 16: Updates every 16 steps (later layers)")

    # Generate dummy data
    def get_batch():
        return {'input_ids': torch.randint(0, 100, (4, 32))}

    losses = []
    print("\n" + "-" * 40)

    for step in range(20):
        batch = get_batch()
        outputs = model(batch['input_ids'], labels=batch['input_ids'])
        loss = outputs['loss']

        # Multi-frequency training step
        trainer.step(loss)
        losses.append(loss.item())

        if step % 4 == 0:
            print(f"Step {step:2d}: loss = {loss.item():.4f}")

    print("-" * 40)
    print(f"\nLoss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")


def demo_generation():
    """Demonstrate text generation with HOPE."""
    print("\n" + "=" * 60)
    print("Demo: Text Generation with HOPE")
    print("=" * 60)

    model = HOPE(
        d_model=128,
        n_heads=2,
        n_layers=2,
        vocab_size=100,
        max_seq_len=64,
    )

    # Start with a prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]])  # 5-token prompt
    print(f"\nPrompt tokens: {prompt.tolist()[0]}")

    # Generate
    generated = model.generate(
        prompt,
        max_length=10,
        temperature=0.8,
        top_k=20,
    )

    print(f"Generated tokens: {generated.tolist()[0]}")
    print(f"\nGenerated {generated.shape[1] - prompt.shape[1]} new tokens!")


def demo_memory_module():
    """Demonstrate the HOPE memory module."""
    print("\n" + "=" * 60)
    print("Demo: HOPE Memory Module")
    print("=" * 60)

    from nested_learning.models.hope import HOPEMemoryModule

    memory = HOPEMemoryModule(
        d_model=64,
        memory_size=32,
        working_memory_size=8,
    )

    print("\nHOPE Memory Module:")
    print("  - Associative memory for general patterns")
    print("  - Surprise-based memory for important patterns")
    print("  - Working memory for recent context")

    # Query the memory
    query = torch.randn(2, 10, 64)  # [batch, seq, dim]

    with torch.no_grad():
        output = memory(query)

    print(f"\nInput shape: {query.shape}")
    print(f"Output shape: {output.shape}")

    # In training mode, memory gets updated
    memory.train()
    value = torch.randn(2, 10, 64)

    output = memory(query, value)
    print("\nMemory updated with new patterns!")


def main():
    """Run all demos."""
    demo_hope_architecture()
    demo_self_modification()
    demo_multi_frequency_training()
    demo_generation()
    demo_memory_module()

    print("\n" + "=" * 60)
    print("HOPE Model Summary")
    print("=" * 60)
    print("""
HOPE combines all Nested Learning concepts:

1. SELF-MODIFYING LAYERS: Attention and MLP layers modify their
   weights based on input surprise, enabling adaptive computation.

2. MULTI-FREQUENCY UPDATES: Different layers update at different
   rates, from fast (syntax) to slow (knowledge) learning.

3. MEMORY SYSTEMS: Combines associative memory, surprise-based
   memory, and working memory for comprehensive context handling.

4. CONTINUUM MEMORY: The CMS creates a spectrum of temporal
   dynamics for stable continual learning.

Result: A model that achieves:
- Lower perplexity than Transformers/Titans on language modeling
- Superior long-context performance (1M+ tokens)
- Better continual learning without catastrophic forgetting
""")


if __name__ == "__main__":
    main()
