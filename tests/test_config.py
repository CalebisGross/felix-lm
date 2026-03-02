"""Tests for the configuration system."""

import pytest

from felix_lm.config import FelixConfig, StageConfig, make_m0_config, make_m2_config


def test_m2_config_structure():
    config = make_m2_config()
    assert config.num_stages == 3
    assert config.total_layers == 13
    assert config.stages[0].num_streams == 4
    assert config.stages[1].num_streams == 2
    assert config.stages[2].num_streams == 1
    assert config.final_dim == 128


def test_m2_attention_types():
    config = make_m2_config()
    assert config.stages[0].attention_type == "linear"
    assert config.stages[1].attention_type == "sliding_window"
    assert config.stages[2].attention_type == "full_causal"


def test_m2_param_count():
    config = make_m2_config()
    estimated = config.count_params()
    assert 9_000_000 <= estimated <= 12_000_000


def test_m0_config_structure():
    config = make_m0_config()
    assert config.num_stages == 1
    assert config.stages[0].num_streams == 1
    assert config.stages[0].attention_type == "full_causal"


def test_m0_param_count():
    config = make_m0_config()
    estimated = config.count_params()
    assert 9_000_000 <= estimated <= 12_000_000


def test_supervision_weights_sum_to_one():
    config = make_m2_config()
    weights = config.get_supervision_weights()
    assert len(weights) == config.num_stages
    assert abs(sum(weights) - 1.0) < 1e-6


def test_supervision_weights_final_stage_heaviest():
    config = make_m2_config()
    weights = config.get_supervision_weights()
    assert weights[-1] > max(weights[:-1])


def test_from_felix_params():
    config = FelixConfig.from_felix_params(top_radius=4.0, turns=2, height=13)
    assert config.num_stages == 3
    assert config.stages[0].num_streams == 4
    assert config.stages[-1].num_streams == 1
    assert config.total_layers == 13


def test_head_dim():
    config = make_m2_config()
    for s in config.stages:
        assert s.dim % s.num_heads == 0
        assert s.head_dim == s.dim // s.num_heads
