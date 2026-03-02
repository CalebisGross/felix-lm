"""Tests for the full Felix-LM model."""

import torch
import pytest

from felix_lm.config import make_m0_config, make_m2_config
from felix_lm.model import FelixLM
from felix_lm.utils import count_parameters


class TestFelixLM:
    @pytest.fixture
    def m2_model(self):
        config = make_m2_config()
        return FelixLM(config)

    def test_forward_output_shape(self, m2_model):
        tokens = torch.randint(0, 50257, (2, 64))
        result = m2_model(tokens)
        assert result["logits"].shape == (2, 64, 50257)

    def test_forward_with_targets(self, m2_model):
        tokens = torch.randint(0, 50257, (2, 64))
        targets = torch.randint(0, 50257, (2, 64))
        result = m2_model(tokens, targets)
        assert "loss" in result
        assert "per_stage_losses" in result
        assert len(result["per_stage_losses"]) == 3  # K=3 stages

    def test_backward_all_gradients(self, m2_model):
        """Every parameter should receive a gradient."""
        tokens = torch.randint(0, 50257, (2, 32))
        targets = torch.randint(0, 50257, (2, 32))
        result = m2_model(tokens, targets)
        result["loss"].backward()

        for name, p in m2_model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_stream_agreements_returned(self, m2_model):
        tokens = torch.randint(0, 50257, (2, 32))
        result = m2_model(tokens)
        # 2 merge points (K-1 = 2)
        assert len(result["stream_agreements"]) == 2

    def test_param_count_matches_estimate(self, m2_model):
        actual = count_parameters(m2_model)
        estimated = m2_model.config.count_params()
        # Within 5%
        assert abs(actual - estimated) / actual < 0.05

    def test_loss_is_finite(self, m2_model):
        tokens = torch.randint(0, 50257, (2, 32))
        targets = torch.randint(0, 50257, (2, 32))
        result = m2_model(tokens, targets)
        assert torch.isfinite(result["loss"])


class TestM0Baseline:
    def test_forward(self):
        config = make_m0_config()
        model = FelixLM(config)
        tokens = torch.randint(0, 50257, (2, 64))
        targets = torch.randint(0, 50257, (2, 64))
        result = model(tokens, targets)
        assert result["logits"].shape == (2, 64, 50257)
        assert torch.isfinite(result["loss"])
        # M0 has no merges, so no stream agreements
        assert len(result["stream_agreements"]) == 0


class TestSmokeTest:
    def test_overfit_single_batch(self):
        """The model should be able to memorize a tiny batch."""
        config = make_m2_config()
        config.dropout = 0.0  # no dropout for overfitting test
        model = FelixLM(config)
        model.train()

        tokens = torch.randint(0, 1000, (1, 32))
        targets = torch.randint(0, 1000, (1, 32))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(50):
            result = model(tokens, targets)
            loss = result["loss"]
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f"Loss didn't decrease enough: {initial_loss:.2f} -> {final_loss:.2f}"
        )
