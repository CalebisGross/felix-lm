"""Tests for the gated merge operation."""

import torch
import pytest

from felix_lm.merge import GatedMerge, MergeLayer


class TestGatedMerge:
    def test_output_shape(self):
        merge = GatedMerge(d_in=64, d_out=128)
        h1 = torch.randn(2, 32, 64)
        h2 = torch.randn(2, 32, 64)
        out = merge(h1, h2)
        assert out.shape == (2, 32, 128)

    def test_gradient_flows_to_both_streams(self):
        merge = GatedMerge(d_in=64, d_out=128)
        h1 = torch.randn(2, 32, 64, requires_grad=True)
        h2 = torch.randn(2, 32, 64, requires_grad=True)
        out = merge(h1, h2)
        out.sum().backward()
        assert h1.grad is not None and h1.grad.abs().sum() > 0
        assert h2.grad is not None and h2.grad.abs().sum() > 0

    def test_gates_start_open(self):
        """With positive bias init, gates should be near sigmoid(1.0) = 0.73."""
        merge = GatedMerge(d_in=64, d_out=128, gate_bias_init=1.0)
        h1 = torch.zeros(1, 1, 64)
        h2 = torch.zeros(1, 1, 64)
        g = merge.get_gate_values(h1, h2)
        # sigmoid(1.0) ≈ 0.731
        assert g.mean().item() > 0.7

    def test_dimension_reduction(self):
        """Merge can also reduce dimension (d_out < 2*d_in)."""
        merge = GatedMerge(d_in=128, d_out=128)
        h1 = torch.randn(1, 16, 128)
        h2 = torch.randn(1, 16, 128)
        out = merge(h1, h2)
        assert out.shape == (1, 16, 128)


class TestMergeLayer:
    def test_without_cross_attention(self):
        layer = MergeLayer(d_in=64, d_out=128, num_heads=4,
                           use_cross_stream_attention=False)
        h1 = torch.randn(2, 32, 64)
        h2 = torch.randn(2, 32, 64)
        out = layer(h1, h2)
        assert out.shape == (2, 32, 128)

    def test_with_cross_attention(self):
        layer = MergeLayer(d_in=64, d_out=128, num_heads=4,
                           use_cross_stream_attention=True)
        h1 = torch.randn(2, 32, 64)
        h2 = torch.randn(2, 32, 64)
        out = layer(h1, h2)
        assert out.shape == (2, 32, 128)
