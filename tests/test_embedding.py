"""Tests for stream initialization."""

import torch
import torch.nn.functional as F
import pytest

from felix_lm.embedding import StreamInitialization


def test_output_shapes():
    init = StreamInitialization(vocab_size=1000, d_embed=128, num_streams=4, d_stream=64)
    tokens = torch.randint(0, 1000, (2, 32))
    e, streams = init(tokens)
    assert e.shape == (2, 32, 128)
    assert len(streams) == 4
    for s in streams:
        assert s.shape == (2, 32, 64)


def test_streams_are_diverse():
    """Different streams should produce different representations."""
    init = StreamInitialization(vocab_size=1000, d_embed=128, num_streams=4, d_stream=64)
    tokens = torch.randint(0, 1000, (1, 32))
    _, streams = init(tokens)

    # Check pairwise cosine similarity is not too high
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            sim = F.cosine_similarity(streams[i], streams[j], dim=-1).mean()
            assert sim.item() < 0.8, f"Streams {i} and {j} too similar: {sim.item()}"


def test_embedding_gradient_flows():
    init = StreamInitialization(vocab_size=1000, d_embed=128, num_streams=4, d_stream=64)
    tokens = torch.randint(0, 1000, (2, 32))
    _, streams = init(tokens)
    loss = sum(s.sum() for s in streams)
    loss.backward()
    assert init.token_embedding.weight.grad is not None
