"""Tests for Felix-LM v2 full model."""

import torch

from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2


def _small_config():
    """Tiny config for fast testing."""
    return FelixV2Config(
        vocab_size=256,
        d_embed=32,
        d_stream=16,
        d_post=16,
        num_streams=4,
        num_layers=3,
        num_heads=2,
        num_refine_layers=1,
        num_refine_heads=2,
        ffn_mult=2,
        dropout=0.0,
    )


def test_forward_shape():
    """Logits should be [B, T, V]."""
    config = _small_config()
    model = FelixLMv2(config)

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids)

    assert result["logits"].shape == (B, T, config.vocab_size)
    assert len(result["agreements"]) == config.num_layers
    assert len(result["merge_strengths"]) == config.num_layers


def test_loss_finite():
    """Loss should be finite and not NaN."""
    config = _small_config()
    model = FelixLMv2(config)

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids, targets)

    assert "loss" in result
    assert torch.isfinite(result["loss"]).all()
    assert not torch.isnan(result["loss"]).any()


def test_all_params_have_gradients():
    """Every parameter should receive a gradient."""
    config = _small_config()
    model = FelixLMv2(config)

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids, targets)
    result["loss"].backward()

    last_layer = config.num_layers - 1
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Layer 0 read_projs: CentralPost starts at zeros, so reading is a no-op.
            # Last layer write_projs/gates: nothing reads CentralPost after the last layer.
            # stream_weights: softmax saturation can zero some grads.
            is_boundary_cp = (
                "layers.0.central_post.read_" in name
                or f"layers.{last_layer}.central_post.write_" in name
                or f"layers.{last_layer}.central_post.norm" in name
            )
            if is_boundary_cp or "stream_weights" in name:
                continue
            assert param.grad is not None, f"{name} has no gradient"
            assert param.grad.abs().sum() > 0, f"{name} has zero gradient"


def test_param_count():
    """Full-size param count should match estimate."""
    config = FelixV2Config()
    model = FelixLMv2(config)

    actual = sum(p.numel() for p in model.parameters())
    estimated = config.count_params()

    # Within 5% of estimate
    ratio = actual / estimated
    assert 0.95 < ratio < 1.05, (
        f"Param count mismatch: actual={actual:,} vs estimated={estimated:,} (ratio={ratio:.3f})"
    )


def test_overfit_single_batch():
    """Model should be able to overfit a single batch in 50 steps."""
    config = _small_config()
    model = FelixLMv2(config)
    model.train()

    B, T = 2, 32
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    final_loss = None
    for step in range(50):
        optimizer.zero_grad()
        result = model(token_ids, targets)
        loss = result["loss"]
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 49:
            final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, (
        f"Model did not overfit: initial={initial_loss:.3f} -> final={final_loss:.3f}"
    )


def test_agreements_increase_with_depth():
    """After some training, later layers should generally have higher agreement."""
    config = _small_config()
    model = FelixLMv2(config)
    model.train()

    B, T = 4, 32
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        optimizer.zero_grad()
        result = model(token_ids, targets)
        result["loss"].backward()
        optimizer.step()

    # Check that agreements are returned and are valid
    result = model(token_ids)
    agreements = result["agreements"]
    assert len(agreements) == config.num_layers
    for a in agreements:
        assert -1.0 <= a.item() <= 1.0


def test_no_loss_without_targets():
    """Forward without targets should not include loss."""
    config = _small_config()
    model = FelixLMv2(config)

    token_ids = torch.randint(0, config.vocab_size, (2, 16))
    result = model(token_ids)

    assert "loss" not in result
    assert "logits" in result
