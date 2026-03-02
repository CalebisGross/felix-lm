"""Tests for attention variants."""

import torch
import pytest

from felix_lm.attention import FullCausalAttention, LinearAttention, SlidingWindowAttention
from felix_lm.rope import build_rope_cache


@pytest.fixture
def rope_cache():
    return build_rope_cache(32, 16, 0, 13, 2)


class TestFullCausalAttention:
    def test_output_shape(self, rope_cache):
        attn = FullCausalAttention(dim=64, num_heads=4)
        x = torch.randn(2, 32, 64)
        out = attn(x, *rope_cache)
        assert out.shape == (2, 32, 64)

    def test_causal_masking(self, rope_cache):
        """Changing future tokens should not affect past outputs."""
        attn = FullCausalAttention(dim=64, num_heads=4)
        attn.eval()

        x1 = torch.randn(1, 32, 64)
        x2 = x1.clone()
        x2[:, 16:, :] = torch.randn(1, 16, 64)  # change future tokens

        out1 = attn(x1, *rope_cache)
        out2 = attn(x2, *rope_cache)

        # Past outputs (positions 0-15) should be identical
        torch.testing.assert_close(out1[:, :16, :], out2[:, :16, :])


class TestSlidingWindowAttention:
    def test_output_shape(self, rope_cache):
        attn = SlidingWindowAttention(dim=64, num_heads=4, window_size=8)
        x = torch.randn(2, 32, 64)
        out = attn(x, *rope_cache)
        assert out.shape == (2, 32, 64)

    def test_window_masking(self, rope_cache):
        """Tokens beyond the window should not affect output."""
        attn = SlidingWindowAttention(dim=64, num_heads=4, window_size=4)
        attn.eval()

        x1 = torch.randn(1, 32, 64)
        x2 = x1.clone()
        # Change token at position 0 — should NOT affect position 20 (window=4)
        x2[:, 0, :] = torch.randn(1, 64)

        out1 = attn(x1, *rope_cache)
        out2 = attn(x2, *rope_cache)

        # Position 20 with window=4 only sees positions 17-20
        # So changing position 0 should have no effect
        torch.testing.assert_close(out1[:, 20, :], out2[:, 20, :])


class TestLinearAttention:
    def test_output_shape(self, rope_cache):
        attn = LinearAttention(dim=64, num_heads=4)
        x = torch.randn(2, 32, 64)
        out = attn(x, *rope_cache)
        assert out.shape == (2, 32, 64)

    def test_causal_masking(self, rope_cache):
        """Changing future tokens should not affect past outputs."""
        attn = LinearAttention(dim=64, num_heads=4)
        attn.eval()

        x1 = torch.randn(1, 32, 64)
        x2 = x1.clone()
        x2[:, 16:, :] = torch.randn(1, 16, 64)

        out1 = attn(x1, *rope_cache)
        out2 = attn(x2, *rope_cache)

        # Past outputs should be identical
        torch.testing.assert_close(out1[:, :16, :], out2[:, :16, :], atol=1e-5, rtol=1e-5)

    def test_no_nans(self, rope_cache):
        """Linear attention should not produce NaNs."""
        attn = LinearAttention(dim=64, num_heads=4)
        x = torch.randn(2, 32, 64)
        out = attn(x, *rope_cache)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self, rope_cache):
        attn = LinearAttention(dim=64, num_heads=4)
        x = torch.randn(2, 32, 64, requires_grad=True)
        out = attn(x, *rope_cache)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
