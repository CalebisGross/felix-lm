"""Muon + AdamW combined optimizer.

Muon (Momentum + Orthogonalization) for 2D weight matrices,
AdamW for everything else (embeddings, norms, biases, scalars).

Reference: Karpathy's autoresearch / Jordan et al.
"""

import math

import torch
from torch.optim import AdamW, Optimizer


def newton_schulz_5(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Approximate the orthogonal polar factor of G via Newton-Schulz iterations.

    Uses a 5th-order (quintic) polynomial per step for fast convergence.
    Returns U where G = U @ S (polar decomposition), U is orthogonal.
    Runs in float32 for numerical stability.

    Reference: Karpathy's autoresearch / "Polar Express" coefficients.
    Always iterates in wide form (cols >= rows) for these specific coefficients.
    """
    original_dtype = G.dtype
    X = G.float()

    # Normalize so spectral norm is ~1 (required for NS convergence)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X / (X.norm() + 1e-7)

    # Always iterate in wide form (the coefficients were optimized for this)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    if transposed:
        X = X.T

    return X.to(original_dtype)


class Muon(Optimizer):
    """Muon optimizer: Nesterov momentum on orthogonalized gradients.

    Only for 2D parameters (weight matrices). Uses Newton-Schulz
    to project gradients onto the Stiefel manifold before the momentum update.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Orthogonalize the gradient
                g_orth = newton_schulz_5(g)

                # Nesterov momentum: buf = mu * buf + g_orth, update = mu * buf + g_orth
                buf.mul_(mu).add_(g_orth)
                p.add_(buf.mul(mu).add(g_orth), alpha=-lr)


class MuonAdamW:
    """Combined optimizer: Muon for 2D weights, AdamW for everything else.

    Exposes a unified interface (.step(), .zero_grad(), .param_groups)
    so the training loop doesn't need to know about the split.
    """

    def __init__(
        self,
        muon_params: list[dict],
        adamw_params: list[dict],
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_weight_decay: float = 0.1,
    ):
        self.muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)
        self.adamw = AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=adamw_betas,
            weight_decay=adamw_weight_decay,
        )

    @property
    def param_groups(self) -> list[dict]:
        return self.muon.param_groups + self.adamw.param_groups

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_optimizer(model: torch.nn.Module, args) -> MuonAdamW:
    """Build MuonAdamW by classifying parameters into optimizer groups.

    2D weight matrices (attention projections, FFN weights) -> Muon
    Everything else (embeddings, norms, biases, scalars, spoke gates) -> AdamW
    """
    muon_params = []
    embed_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_embedding = "embedding" in name and "weight" in name
        is_2d = param.dim() == 2

        if is_embedding:
            embed_params.append(param)
        elif is_2d:
            muon_params.append(param)
        else:
            other_params.append(param)

    # muP-style width correction for Muon LR
    d_embed = getattr(model, "config", None)
    if d_embed is not None:
        d_embed = getattr(d_embed, "d_embed", 768)
    else:
        d_embed = 768
    muon_lr = args.muon_lr * math.sqrt(768 / d_embed)

    # Embedding gets higher LR (updated sparsely, needs aggressive updates)
    embed_lr = args.lr * args.embed_lr_mult

    # Build param groups
    muon_groups = [{"params": muon_params, "lr": muon_lr, "base_lr": muon_lr}]

    adamw_groups = []
    if embed_params:
        adamw_groups.append({"params": embed_params, "lr": embed_lr, "base_lr": embed_lr})
    if other_params:
        adamw_groups.append({"params": other_params, "lr": args.lr, "base_lr": args.lr})

    n_muon = sum(p.numel() for p in muon_params)
    n_embed = sum(p.numel() for p in embed_params)
    n_other = sum(p.numel() for p in other_params)
    print("  Optimizer: MuonAdamW")
    print(f"    Muon params:  {n_muon:>10,} (lr={muon_lr:.4f})")
    print(f"    Embed params: {n_embed:>10,} (lr={embed_lr:.4f})")
    print(f"    Other params: {n_other:>10,} (lr={args.lr:.4f})")

    return MuonAdamW(
        muon_params=muon_groups,
        adamw_params=adamw_groups,
        muon_lr=muon_lr,
        muon_momentum=args.muon_momentum,
        adamw_lr=args.lr,
        adamw_betas=(args.beta1, args.beta2),
        adamw_weight_decay=args.weight_decay,
    )
