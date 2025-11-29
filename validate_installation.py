#!/usr/bin/env python3
"""
Validate Nested Learning Installation

Run this script to verify that all components are properly installed
and working correctly.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from nested_learning import (
            # Optimizers
            DeepMomentumGD,
            DeltaRuleMomentum,
            PreconditionedMomentum,
            # Memory
            AssociativeMemory,
            ContinuumMemorySystem,
            ContinuumMemoryBlock,
            # Models
            HOPE,
            SelfModifyingLinear,
            SelfModifyingAttention,
            SelfModifyingMLP,
        )
        print("  All imports successful!")
        return True
    except ImportError as e:
        print(f"  Import failed: {e}")
        traceback.print_exc()
        return False


def test_deep_momentum():
    """Test Deep Momentum GD optimizer."""
    print("\nTesting Deep Momentum GD optimizer...")

    try:
        import torch
        from nested_learning import DeepMomentumGD

        # Create a simple parameter
        param = torch.randn(10, requires_grad=True)
        optimizer = DeepMomentumGD([param], lr=0.01)

        # Simulate a training step
        loss = param.sum()
        loss.backward()
        optimizer.step()

        print("  Deep Momentum GD: OK")
        return True
    except Exception as e:
        print(f"  Deep Momentum GD failed: {e}")
        traceback.print_exc()
        return False


def test_delta_rule():
    """Test Delta Rule Momentum optimizer."""
    print("\nTesting Delta Rule Momentum optimizer...")

    try:
        import torch
        from nested_learning import DeltaRuleMomentum

        param = torch.randn(10, requires_grad=True)
        optimizer = DeltaRuleMomentum([param], lr=0.01)

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Test surprise metric
        surprise = optimizer.get_surprise_magnitude()
        print(f"  Surprise magnitude: {surprise:.4f}")
        print("  Delta Rule Momentum: OK")
        return True
    except Exception as e:
        print(f"  Delta Rule Momentum failed: {e}")
        traceback.print_exc()
        return False


def test_associative_memory():
    """Test Associative Memory module."""
    print("\nTesting Associative Memory...")

    try:
        import torch
        from nested_learning import AssociativeMemory

        memory = AssociativeMemory(
            memory_size=32,
            key_dim=64,
            value_dim=64,
        )

        # Query the memory
        query = torch.randn(4, 64)
        output, _ = memory.read(query)

        print(f"  Input shape: {query.shape}")
        print(f"  Output shape: {output.shape}")
        print("  Associative Memory: OK")
        return True
    except Exception as e:
        print(f"  Associative Memory failed: {e}")
        traceback.print_exc()
        return False


def test_continuum_memory():
    """Test Continuum Memory System."""
    print("\nTesting Continuum Memory System...")

    try:
        import torch
        from nested_learning import ContinuumMemoryBlock

        cms = ContinuumMemoryBlock(
            d_model=64,
            frequencies=[1, 4, 16],
        )

        x = torch.randn(2, 10, 64)
        output = cms(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print("  Continuum Memory System: OK")
        return True
    except Exception as e:
        print(f"  Continuum Memory System failed: {e}")
        traceback.print_exc()
        return False


def test_self_modifying_linear():
    """Test Self-Modifying Linear layer."""
    print("\nTesting Self-Modifying Linear...")

    try:
        import torch
        from nested_learning import SelfModifyingLinear

        layer = SelfModifyingLinear(64, 64)
        x = torch.randn(4, 64)
        output = layer(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print("  Self-Modifying Linear: OK")
        return True
    except Exception as e:
        print(f"  Self-Modifying Linear failed: {e}")
        traceback.print_exc()
        return False


def test_self_modifying_attention():
    """Test Self-Modifying Attention."""
    print("\nTesting Self-Modifying Attention...")

    try:
        import torch
        from nested_learning import SelfModifyingAttention

        attn = SelfModifyingAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print("  Self-Modifying Attention: OK")
        return True
    except Exception as e:
        print(f"  Self-Modifying Attention failed: {e}")
        traceback.print_exc()
        return False


def test_hope_model():
    """Test HOPE model."""
    print("\nTesting HOPE Model...")

    try:
        import torch
        from nested_learning import HOPE

        model = HOPE(
            d_model=128,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
            max_seq_len=64,
            frequencies=[1, 4],
            use_memory=True,
            use_cms=True,
        )

        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, labels=input_ids)

        print(f"  Input shape: {input_ids.shape}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Parameters: {model.get_num_params():,}")
        print("  HOPE Model: OK")
        return True
    except Exception as e:
        print(f"  HOPE Model failed: {e}")
        traceback.print_exc()
        return False


def test_training_utils():
    """Test training utilities."""
    print("\nTesting Training Utilities...")

    try:
        import torch
        from nested_learning.utils.training import (
            create_frequency_schedule,
            GradientAccumulator,
            MultiFrequencyTrainer,
        )

        # Test frequency schedule
        schedule = create_frequency_schedule([1, 8, 64], base_lr=1e-3)
        print(f"  Frequency schedule created with {len(schedule)} levels")

        # Test gradient accumulator
        param = torch.randn(10, requires_grad=True)
        accumulator = GradientAccumulator([param], frequency=4)
        print(f"  Gradient accumulator created")

        print("  Training Utilities: OK")
        return True
    except Exception as e:
        print(f"  Training Utilities failed: {e}")
        traceback.print_exc()
        return False


def test_forward_backward():
    """Test full forward/backward pass."""
    print("\nTesting Forward/Backward Pass...")

    try:
        import torch
        from nested_learning import HOPE

        model = HOPE(
            d_model=64,
            n_heads=2,
            n_layers=2,
            vocab_size=50,
            max_seq_len=32,
        )

        # Forward pass
        input_ids = torch.randint(0, 50, (2, 16))
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed: {has_grads}")
        print("  Forward/Backward Pass: OK")
        return True
    except Exception as e:
        print(f"  Forward/Backward Pass failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Nested Learning Installation Validation")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Deep Momentum GD", test_deep_momentum),
        ("Delta Rule Momentum", test_delta_rule),
        ("Associative Memory", test_associative_memory),
        ("Continuum Memory System", test_continuum_memory),
        ("Self-Modifying Linear", test_self_modifying_linear),
        ("Self-Modifying Attention", test_self_modifying_attention),
        ("HOPE Model", test_hope_model),
        ("Training Utilities", test_training_utils),
        ("Forward/Backward Pass", test_forward_backward),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n{name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Installation is valid.")
        return 0
    else:
        print("\nSome tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
