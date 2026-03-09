"""Tests for CentralPost hub module."""

import torch

from felix_lm.v2.central_post import CentralPostLayer


def test_output_shapes():
    """Streams and central_post maintain their shapes."""
    B, T, d_stream, d_post, N = 2, 16, 64, 64, 4
    layer = CentralPostLayer(d_stream, d_post, N)

    streams = [torch.randn(B, T, d_stream) for _ in range(N)]
    cp = torch.zeros(B, T, d_post)

    out_streams, out_cp = layer(streams, cp)

    assert len(out_streams) == N
    for s in out_streams:
        assert s.shape == (B, T, d_stream)
    assert out_cp.shape == (B, T, d_post)


def test_central_post_updates():
    """CentralPost should not stay at zeros after write."""
    B, T, d_stream, d_post, N = 2, 16, 64, 64, 4
    layer = CentralPostLayer(d_stream, d_post, N)

    streams = [torch.randn(B, T, d_stream) for _ in range(N)]
    cp = torch.zeros(B, T, d_post)

    _, out_cp = layer(streams, cp)

    # After write phase, central_post should have changed
    assert not torch.allclose(out_cp, torch.zeros_like(out_cp), atol=1e-6)


def test_gradients_flow():
    """Gradients should flow to all streams and through CentralPost."""
    B, T, d_stream, d_post, N = 2, 16, 64, 64, 4
    layer = CentralPostLayer(d_stream, d_post, N)

    streams = [torch.randn(B, T, d_stream, requires_grad=True) for _ in range(N)]
    cp = torch.zeros(B, T, d_post, requires_grad=True)

    out_streams, out_cp = layer(streams, cp)

    # Loss from both streams and central_post
    loss = sum(s.sum() for s in out_streams) + out_cp.sum()
    loss.backward()

    # All input streams should have gradients
    for i, s in enumerate(streams):
        assert s.grad is not None, f"Stream {i} has no gradient"
        assert s.grad.abs().sum() > 0, f"Stream {i} has zero gradient"

    # CentralPost input should have gradient
    assert cp.grad is not None
    assert cp.grad.abs().sum() > 0


def test_streams_modified_by_read():
    """Streams should be modified by reading from a non-zero CentralPost."""
    B, T, d_stream, d_post, N = 2, 16, 64, 64, 4
    layer = CentralPostLayer(d_stream, d_post, N)

    streams = [torch.randn(B, T, d_stream) for _ in range(N)]
    cp = torch.randn(B, T, d_post)  # Non-zero CentralPost

    out_streams, _ = layer(streams, cp)

    # At least one stream should differ from its input
    any_changed = False
    for orig, updated in zip(streams, out_streams):
        if not torch.allclose(orig, updated, atol=1e-6):
            any_changed = True
            break
    assert any_changed, "No streams were modified by CentralPost read"
