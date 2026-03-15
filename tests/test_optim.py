"""Tests for Muon + AdamW combined optimizer."""

import types

import torch
import torch.nn as nn

from felix_lm.optim import Muon, MuonAdamW, build_optimizer, newton_schulz_5
from felix_lm.v3.config import FelixV3Config
from felix_lm.v3.model import FelixLMv3


def test_newton_schulz_shape_preserved():
    """NS should preserve input shape for square, tall, and wide matrices."""
    for shape in [(64, 64), (128, 32), (32, 128)]:
        G = torch.randn(*shape)
        U = newton_schulz_5(G)
        assert U.shape == G.shape, f"Shape mismatch for {shape}: got {U.shape}"


def test_newton_schulz_improves_orthogonality():
    """More NS steps should produce more orthogonal output."""
    G = torch.randn(32, 32)

    U_few = newton_schulz_5(G, steps=2)
    U_many = newton_schulz_5(G, steps=20)

    # Compute orthogonality deviation (how far U @ U^T is from identity)
    dev_few = (U_few @ U_few.T - torch.eye(32)).abs().max().item()
    dev_many = (U_many @ U_many.T - torch.eye(32)).abs().max().item()

    assert dev_many < dev_few, (
        f"More steps should improve orthogonality: {dev_few:.4f} -> {dev_many:.4f}"
    )


def test_newton_schulz_finite():
    """NS output should always be finite (no NaN/Inf)."""
    for shape in [(64, 64), (128, 32), (32, 128)]:
        G = torch.randn(*shape)
        U = newton_schulz_5(G)
        assert torch.isfinite(U).all(), f"NS output has non-finite values for {shape}"


def test_muon_step():
    """One Muon step should reduce loss on a simple problem."""
    W = nn.Parameter(torch.randn(32, 32))
    target = torch.randn(32, 32)

    opt = Muon([W], lr=0.01, momentum=0.95)

    loss_before = ((W - target) ** 2).sum().item()
    loss = ((W - target) ** 2).sum()
    loss.backward()
    opt.step()

    loss_after = ((W.detach() - target) ** 2).sum().item()
    assert loss_after < loss_before, (
        f"Muon step did not reduce loss: {loss_before:.4f} -> {loss_after:.4f}"
    )


def test_build_optimizer_groups():
    """build_optimizer should correctly classify parameters."""
    config = FelixV3Config(
        vocab_size=256,
        d_embed=32,
        num_layers=4,
        num_heads=2,
        ffn_mult=2,
        num_spokes=4,
        spoke_rank=8,
        gate_schedule="uniform",
    )
    model = FelixLMv3(config)

    args = types.SimpleNamespace(
        optimizer="muon_adamw",
        lr=3e-4,
        muon_lr=0.02,
        embed_lr_mult=10.0,
        muon_momentum=0.95,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
    )
    optimizer = build_optimizer(model, args)

    assert isinstance(optimizer, MuonAdamW)

    # Should have Muon groups (2D weights) and AdamW groups (embed + other)
    muon_groups = optimizer.muon.param_groups
    adamw_groups = optimizer.adamw.param_groups

    muon_params = sum(p.numel() for g in muon_groups for p in g["params"])
    adamw_params = sum(p.numel() for g in adamw_groups for p in g["params"])

    # Muon should have the bulk of 2D params (attention + FFN projections)
    assert muon_params > 0, "No Muon params found"
    assert adamw_params > 0, "No AdamW params found"
    # Embedding alone is vocab_size * d_embed = 256 * 32 = 8192
    assert adamw_params >= 256 * 32, "Embedding should be in AdamW"


def test_muon_adamw_wrapper():
    """MuonAdamW wrapper should expose step, zero_grad, param_groups."""
    config = FelixV3Config(
        vocab_size=256,
        d_embed=32,
        num_layers=4,
        num_heads=2,
        ffn_mult=2,
        num_spokes=4,
        spoke_rank=8,
        gate_schedule="uniform",
    )
    model = FelixLMv3(config)

    args = types.SimpleNamespace(
        optimizer="muon_adamw",
        lr=3e-4,
        muon_lr=0.02,
        embed_lr_mult=10.0,
        muon_momentum=0.95,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
    )
    optimizer = build_optimizer(model, args)

    # param_groups should be non-empty
    assert len(optimizer.param_groups) > 0

    # Should be able to do a training step
    token_ids = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    result = model(token_ids, targets)
    result["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()


def test_muon_v3_no_nan():
    """10 training steps with MuonAdamW should not produce NaN."""
    config = FelixV3Config(
        vocab_size=256,
        d_embed=32,
        num_layers=4,
        num_heads=2,
        ffn_mult=2,
        num_spokes=4,
        spoke_rank=8,
        gate_schedule="uniform",
    )
    model = FelixLMv3(config)

    args = types.SimpleNamespace(
        optimizer="muon_adamw",
        lr=3e-4,
        muon_lr=0.02,
        embed_lr_mult=10.0,
        muon_momentum=0.95,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
    )
    optimizer = build_optimizer(model, args)

    token_ids = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    for _ in range(10):
        optimizer.zero_grad()
        result = model(token_ids, targets)
        loss = result["loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def test_muon_lr_scheduling():
    """LR scheduling via base_lr should work for all param groups."""
    config = FelixV3Config(
        vocab_size=256,
        d_embed=32,
        num_layers=4,
        num_heads=2,
        ffn_mult=2,
        gate_schedule="none",
    )
    model = FelixLMv3(config)

    args = types.SimpleNamespace(
        optimizer="muon_adamw",
        lr=3e-4,
        muon_lr=0.02,
        embed_lr_mult=10.0,
        muon_momentum=0.95,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
    )
    optimizer = build_optimizer(model, args)

    # Apply 0.5x schedule multiplier
    for pg in optimizer.param_groups:
        base = pg.get("base_lr", args.lr)
        pg["lr"] = base * 0.5

    # Verify each group got halved from its base
    for pg in optimizer.param_groups:
        base = pg["base_lr"]
        expected = base * 0.5
        assert abs(pg["lr"] - expected) < 1e-8, (
            f"LR scheduling failed: expected {expected}, got {pg['lr']}"
        )
