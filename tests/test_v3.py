"""Tests for Felix-LM v3 hub-and-spoke architecture."""

import torch

from felix_lm.v3.config import FelixV3Config
from felix_lm.v3.model import FelixLMv3


def _small_config(**overrides):
    """Tiny config for fast testing."""
    defaults = dict(
        vocab_size=256,
        d_embed=32,
        num_layers=4,
        num_heads=2,
        ffn_mult=2,
        num_spokes=4,
        spoke_rank=8,
        gate_schedule="progressive",
        dropout=0.0,
    )
    defaults.update(overrides)
    return FelixV3Config(**defaults)


def test_forward_shape():
    """Logits should be [B, T, V]."""
    config = _small_config()
    model = FelixLMv3(config)

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids)

    assert result["logits"].shape == (B, T, config.vocab_size)
    assert len(result["agreements"]) == config.num_layers
    assert len(result["gate_values"]) == config.num_layers


def test_loss_finite():
    """Loss should be finite and not NaN."""
    config = _small_config()
    model = FelixLMv3(config)

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids, targets)

    assert "loss" in result
    assert torch.isfinite(result["loss"]).all()
    assert not torch.isnan(result["loss"]).any()


def test_no_loss_without_targets():
    """Forward without targets should not include loss."""
    config = _small_config()
    model = FelixLMv3(config)

    token_ids = torch.randint(0, config.vocab_size, (2, 16))
    result = model(token_ids)

    assert "loss" not in result
    assert "logits" in result


def test_all_params_have_gradients():
    """Every parameter should receive a gradient."""
    config = _small_config()
    model = FelixLMv3(config)

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids, targets)
    result["loss"].backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
            # Spoke params upstream of W_up (gate_bias, norm, w_down) get zero grad
            # at init because W_up is initialized to zeros — the spoke contribution
            # is zero, so dL/d(upstream) = 0. Resolves after first optimizer step.
            # W_up itself DOES get gradient (it's the bottleneck that breaks symmetry).
            if "spokes" in name and "w_up" not in name:
                continue
            assert param.grad.abs().sum() > 0, f"{name} has zero gradient"


def test_agreement_range():
    """Agreement values should be in [-1, 1]."""
    config = _small_config()
    model = FelixLMv3(config)

    token_ids = torch.randint(0, config.vocab_size, (4, 32))
    result = model(token_ids)

    for a in result["agreements"]:
        assert -1.0 <= a.item() <= 1.0, f"Agreement {a.item()} out of range"


def test_gate_progressive_initialization():
    """Progressive gates should increase from early to late layers."""
    config = _small_config(gate_schedule="progressive", gate_init_start=-2.0, gate_init_end=2.0)
    model = FelixLMv3(config)

    gate_values = [torch.sigmoid(s.gate_bias).item() for s in model.spokes]
    # Early gates should be smaller than late gates
    assert gate_values[0] < gate_values[-1], f"Progressive gates not increasing: {gate_values}"
    # First gate should be near sigmoid(-2) ~ 0.12
    assert gate_values[0] < 0.2
    # Last gate should be near sigmoid(2) ~ 0.88
    assert gate_values[-1] > 0.8


def test_gate_uniform_initialization():
    """Uniform gates should all be sigmoid(0) = 0.5."""
    config = _small_config(gate_schedule="uniform")
    model = FelixLMv3(config)

    for s in model.spokes:
        gate_val = torch.sigmoid(s.gate_bias).item()
        assert abs(gate_val - 0.5) < 0.01, f"Uniform gate not 0.5: {gate_val}"


def test_no_spokes_mode():
    """With gate_schedule='none', model should be a plain transformer."""
    config = _small_config(gate_schedule="none")
    model = FelixLMv3(config)

    assert model.spokes is None

    B, T = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    result = model(token_ids, targets)

    assert result["logits"].shape == (B, T, config.vocab_size)
    assert torch.isfinite(result["loss"])
    assert len(result["agreements"]) == 0
    assert len(result["gate_values"]) == 0


def test_param_count():
    """Full-size param count should match estimate."""
    config = FelixV3Config()  # default 11M config
    model = FelixLMv3(config)

    actual = sum(p.numel() for p in model.parameters())
    estimated = config.count_params()

    ratio = actual / estimated
    assert 0.95 < ratio < 1.05, (
        f"Param count mismatch: actual={actual:,} vs estimated={estimated:,} (ratio={ratio:.3f})"
    )


def test_param_count_no_spokes():
    """Without spokes, param count should match a plain transformer."""
    config = FelixV3Config(gate_schedule="none")
    model = FelixLMv3(config)

    actual = sum(p.numel() for p in model.parameters())
    estimated = config.count_params()

    ratio = actual / estimated
    assert 0.95 < ratio < 1.05, (
        f"Param count mismatch: actual={actual:,} vs estimated={estimated:,} (ratio={ratio:.3f})"
    )


def test_overfit_single_batch():
    """Model should be able to overfit a single batch in 50 steps."""
    config = _small_config()
    model = FelixLMv3(config)
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


def test_spoke_overhead_small():
    """Spoke overhead should be <5% of total params at default config."""
    config = FelixV3Config()
    model = FelixLMv3(config)

    total = sum(p.numel() for p in model.parameters())
    spoke_params = sum(p.numel() for name, p in model.named_parameters() if "spokes" in name)

    overhead_pct = spoke_params / total * 100
    assert overhead_pct < 5.0, f"Spoke overhead too high: {overhead_pct:.1f}%"


def test_gradient_checkpointing():
    """Gradient checkpointing should produce same loss as without."""
    config = _small_config()
    model_no_ckpt = FelixLMv3(config)

    config_ckpt = _small_config(gradient_checkpointing=True)
    model_ckpt = FelixLMv3(config_ckpt)

    # Copy weights
    model_ckpt.load_state_dict(model_no_ckpt.state_dict())

    token_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    model_no_ckpt.train()
    model_ckpt.train()

    result1 = model_no_ckpt(token_ids, targets)
    result2 = model_ckpt(token_ids, targets)

    assert torch.allclose(result1["loss"], result2["loss"], atol=1e-5), (
        f"Loss mismatch: {result1['loss'].item()} vs {result2['loss'].item()}"
    )
