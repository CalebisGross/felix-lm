"""Tests for adaptive merge module."""

import torch

from felix_lm.v2.adaptive_merge import AdaptiveMergeLayer


def test_output_shapes():
    """Output streams should match input shapes."""
    B, T, d, N = 2, 16, 64, 4
    layer = AdaptiveMergeLayer(d)

    streams = [torch.randn(B, T, d) for _ in range(N)]
    merged, agreement, strength = layer(streams)

    assert len(merged) == N
    for s in merged:
        assert s.shape == (B, T, d)
    assert agreement.shape == (B, T)
    assert strength.shape == (B, T)


def test_identical_streams_high_agreement():
    """Identical streams should yield agreement=1.0 and high merge strength."""
    B, T, d = 2, 16, 64
    layer = AdaptiveMergeLayer(d, temp_init=5.0, bias_init=0.0)

    base = torch.randn(B, T, d)
    streams = [base.clone() for _ in range(4)]

    _, agreement, strength = layer(streams)

    assert agreement.mean() > 0.99, f"Expected agreement ~1.0, got {agreement.mean():.3f}"
    assert strength.mean() > 0.5, f"Expected high merge strength, got {strength.mean():.3f}"


def test_orthogonal_streams_low_agreement():
    """Orthogonal streams should yield low agreement and low merge strength."""
    B, T, d = 2, 16, 64
    layer = AdaptiveMergeLayer(d, temp_init=5.0, bias_init=-3.0)

    # Create orthogonal streams using QR decomposition
    torch.manual_seed(42)
    streams = []
    for i in range(4):
        s = torch.randn(B, T, d)
        # Make each stream distinct by scaling different dimensions
        mask = torch.zeros(d)
        mask[i * 16 : (i + 1) * 16] = 1.0
        streams.append(s * mask)

    _, agreement, strength = layer(streams)

    assert agreement.mean() < 0.3, f"Expected low agreement, got {agreement.mean():.3f}"


def test_gradients_flow():
    """Temperature and bias should receive gradients."""
    B, T, d = 2, 16, 64
    layer = AdaptiveMergeLayer(d)

    streams = [torch.randn(B, T, d, requires_grad=True) for _ in range(4)]
    merged, _, _ = layer(streams)

    loss = sum(s.sum() for s in merged)
    loss.backward()

    assert layer.temperature.grad is not None, "Temperature has no gradient"
    assert layer.bias.grad is not None, "Bias has no gradient"

    for i, s in enumerate(streams):
        assert s.grad is not None, f"Stream {i} has no gradient"


def test_merge_strength_bounded():
    """Merge strength should always be in [0, 1]."""
    B, T, d = 2, 16, 64
    layer = AdaptiveMergeLayer(d)

    streams = [torch.randn(B, T, d) for _ in range(4)]
    _, _, strength = layer(streams)

    assert (strength >= 0).all()
    assert (strength <= 1).all()


def test_initial_merge_strength():
    """With default bias_init=-1.0 and temp_init=1.0, initial strength should be moderate."""
    B, T, d = 2, 16, 64
    layer = AdaptiveMergeLayer(d, temp_init=1.0, bias_init=-1.0)

    streams = [torch.randn(B, T, d) for _ in range(4)]
    _, _, strength = layer(streams)

    mean_strength = strength.mean().item()
    # Random streams have low agreement, so strength should be below 0.5
    assert mean_strength < 0.5, f"Expected low initial strength, got {mean_strength:.3f}"
