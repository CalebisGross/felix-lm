"""Utility functions for Felix-LM."""

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_component(model: torch.nn.Module) -> dict[str, int]:
    """Count parameters grouped by top-level component."""
    counts: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        component = name.split(".")[0]
        counts[component] = counts.get(component, 0) + p.numel()
    return counts
