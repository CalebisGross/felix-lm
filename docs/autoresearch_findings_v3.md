# Felix-LM v3 Autoresearch: Hub-and-Spoke Findings

**38 experiments, 11M and 100M scale, March 2026**

## The Question

Can lightweight "spoke" probes — low-rank read/write operations branching off a standard transformer backbone — improve perplexity over the backbone alone?

## The Answer (Short Version)

Yes. At both 11M and 100M parameters, a transformer with 4 spoke probes at rank 32 beats the same transformer without spokes. The improvement grows with scale: -0.72 PPL at 11M (1.5%), -1.19 PPL at 100M (2.7%).

This is the first Felix architecture variant that definitively improves over a plain transformer at matched hyperparameters and training steps.

## What We Built

### v3 Architecture: Hub-and-Spoke

v2 tried to make parallel streams work. 82 experiments proved that duplicating transformer blocks across streams is a parameter tax that never pays for itself. v3 inverts the design:

**The hub (backbone) is a standard transformer.** It gets all the parameter budget, all the depth, all the attention capacity. This is the CentralPost — the central shared state.

**The spokes are lightweight low-rank probes.** Each spoke reads the hub state through a learned projection (d -> r), applies SiLU, projects back (r -> d), and gates the result into the hub as a residual. Spokes are the "agents" — they explore different aspects of the representation and report back.

```
For each layer:
  h = h + Attention(RMSNorm(h))         # hub (standard transformer)
  h = h + FFN(RMSNorm(h))               # hub (standard transformer)
  h_norm = RMSNorm(h)                    # spoke preprocessing
  For each spoke s:
    view_s = SiLU(h_norm @ W_down_s)    # project to low-rank [d -> r]
    update_s = view_s @ W_up_s           # project back [r -> d]
  h = h + sigmoid(gate_bias) * mean(updates)  # gated residual
```

The key design choices, all validated by ablation:
- **W_up initialized to zeros** — spokes start as identity and bootstrap gradually
- **Uniform gate schedule** — all layers start at sigmoid(0) = 0.5
- **4 spokes at rank 32** — best efficiency tradeoff (5.6% param overhead)
- **Agreement as diagnostic** — cross-spoke cosine similarity computed with no_grad

### How Felix Principles Map

| Felix Principle | v3 Realization |
|-----------------|----------------|
| CentralPost as hub | The transformer backbone IS the CentralPost |
| Agents as explorers | Spokes are lightweight agents that read/write the hub |
| Parallel exploration | S spokes process in parallel with different projections |
| Progressive convergence | Gate values can be learned per-layer |
| Agreement as confidence | Cross-spoke cosine similarity on low-rank views |
| Helical trajectory | Depth-extended RoPE on the backbone |

## Experiment Phases

### Phase 1: Gate Schedule (exp 1-3)

Tested whether spokes help at all, and how to initialize the gates.

| Exp | Config | Val PPL | Delta vs none |
|-----|--------|---------|---------------|
| 1 | v3_none (no spokes) | 47.90 | baseline |
| 2 | v3_base (progressive gates) | 48.61 | +0.71 |
| 3 | v3_uniform (uniform gates) | 47.64 | **-0.26** |

**Finding:** Progressive gates (small early, large late) hurt because they starve early-layer spokes. Uniform gates (all start at 0.5) let spokes contribute equally and learn their own schedule.

### Phase 2: Rank Scaling (exp 4-6)

How much capacity does each spoke need?

| Exp | Config | Val PPL | Delta | Overhead |
|-----|--------|---------|-------|----------|
| 4 | v3_r8 (rank 8) | 48.82 | +0.92 | 1.4% |
| 3 | v3_uniform (rank 16) | 47.64 | -0.26 | 2.8% |
| 5 | v3_r32 (rank 32) | 47.18 | **-0.72** | 5.6% |
| 6 | v3_r64 (rank 64) | 47.06 | **-0.84** | 11.2% |

**Finding:** Rank scales monotonically. Diminishing returns set in after rank 32 — going from r32 to r64 gains only 0.12 PPL for 5.6% more params, while r16 to r32 gained 0.46 for 2.8%. Rank 32 is the efficiency sweet spot.

### Phase 3: Spoke Count (exp 7-8)

How many diverse probes are needed?

| Exp | Config | Val PPL | Delta |
|-----|--------|---------|-------|
| 7 | v3_2spoke | 48.62 | +0.72 |
| 3 | v3_uniform (4 spokes) | 47.64 | -0.26 |
| 8 | v3_8spoke (killed) | ~156 at step 500 | too slow |

**Finding:** 4 spokes is optimal. 2 spokes don't provide enough diversity. 8 spokes are 2.5x slower due to compile overhead with the loop, and weren't promising at step 500.

### Phase 4: Initialization (exp 9)

Does random W_up init help spokes bootstrap faster?

| Exp | Config | Val PPL | Delta vs none |
|-----|--------|---------|---------------|
| 3 | v3_uniform (zeros W_up) | 47.64 | -0.26 |
| 9 | v3_uniform (random W_up, std=0.01) | 47.94 | +0.04 |

**Finding:** Zeros init is better. Random init introduces noise early that the model must overcome. Zeros lets spokes start silent and gradually bootstrap, which is less disruptive.

### Phase 5: Depth Comparison (exp 10)

Are spokes a better use of params than extra transformer layers?

| Exp | Config | Params | Val PPL |
|-----|--------|--------|---------|
| 5 | v3_r32 (20 layers + spokes) | 12.34M | **47.18** |
| 10 | v3_deep22 (22 layers, no spokes) | 12.21M | 47.69 |

**Finding:** Spokes beat extra depth by 0.51 PPL at matched params. 4 spokes at rank 32 are worth more than 2 additional transformer layers.

### Phase 6: Extended Training (exp 11-12)

Does the spoke advantage hold with more training?

| Exp | Config | Steps | Val PPL |
|-----|--------|-------|---------|
| 5 | v3_r32 | 2500 | 47.18 |
| 11 | v3_r32 | 5000 | 41.89 |
| 12 | v2_baseline | 5000 | 39.63 |

**Finding:** At 5000 matched steps, the plain transformer still wins overall (39.63 vs 41.89). The v3 baseline (47.90) is worse than v2_baseline (44.81) because v3 uses nn.Embedding directly while v2 uses StreamInitialization with extra projections. The spoke improvement is real but doesn't overcome this embedding gap.

### Phase 7: 100M Scale (exp 13-14)

The critical test — does the spoke advantage hold or grow at scale?

| Exp | Config | Params | Val PPL |
|-----|--------|--------|---------|
| 13 | v3_100m_none | 109.6M | 43.88 |
| 14 | v3_100m_r32 | 112.3M | **42.69** |

**Finding:** The spoke advantage **grows with scale**:
- 11M: -0.72 PPL (1.5% improvement)
- 100M: -1.19 PPL (2.7% improvement)

At 100M, embedding is only ~23% of params (vs 55% at 11M), so more of the budget goes to compute where spokes can contribute. This suggests the advantage will continue growing at 500M+.

### Phase 8: Embedding Projection (exp 15-16, 11M)

v3's baseline (47.90) was worse than v2's (44.81) because v2 uses StreamInitialization with extra projections while v3 uses raw nn.Embedding. Can we close this gap?

| Exp | Config | Params | Val PPL | Delta vs v3_none |
|-----|--------|--------|---------|-------------------|
| 15 | v3_proj (proj only, no spokes) | 11.70M | 46.57 | -1.33 |
| 16 | v3_proj_r32 (proj + 4 spokes r=32) | 12.36M | **45.98** | **-1.92** |

**Finding:** Embedding projection and spokes **compound nearly additively**. Projection alone gives -1.33 PPL, spokes alone give -0.72 PPL, and together they give -1.92 PPL (94% of the expected -2.05 if perfectly additive). This means the two improvements target different bottlenecks: projection improves the initial representation quality, while spokes improve per-layer processing. The combined v3_proj_r32 at 45.98 PPL is now within 1.17 PPL of v2_baseline (44.81) — the embedding gap is almost closed.

### Phase 9: 100M LR Sweep & Skepticism Check (exp 18-21a)

The spoke advantage at 100M was tested under organized skepticism.

| Config | LR 2e-3 | LR 3e-3 | Delta from LR |
|--------|---------|---------|---------------|
| v3_100m_none (baseline) | 43.88 | **41.30** | -2.58 |
| v3_100m_r32 (spokes) | 42.69 | 41.66 | -1.03 |
| Spoke advantage | -1.19 | **+0.36** | **gone** |

Additional: v3_100m_r64 = 43.22 PPL (worse than r32, higher rank hurts at 100M).

**Finding:** The entire spoke advantage at 100M was a learning rate artifact. At LR 2e-3 (which was suboptimal), spokes appeared to help by -1.19 PPL. At LR 3e-3 (the better LR), the baseline beats spokes by 0.36 PPL. This is the same pattern as v2 EXP-81: Felix architectures act as implicit regularizers that partially compensate for too-low LR. Once LR is tuned, the plain transformer wins because it has less overhead.

This does NOT invalidate the 11M results, where LR 2e-2 was already the optimized LR from 58 experiments of v2 sweep. But it means the 100M scaling story ("advantage grows with scale") was wrong.

## Key Findings

### What Works (at 11M with tuned LR)

1. **Hub-and-spoke is the right inversion.** v2 put diversity in the expensive path (stream blocks). v3 puts diversity in the cheapest possible path (low-rank probes). The expensive backbone is shared.

2. **Spokes beat extra depth at 11M.** At matched params, 4 spokes at rank 32 outperform 2 additional transformer layers. The diverse feedback from multiple cheap probes is more valuable than monolithic depth.

3. **Zeros init is essential.** Spokes must start as identity and bootstrap gradually. Random init hurts.

4. **Uniform gates beat progressive.** Let the model learn its own gate schedule rather than imposing one.

### What Scales (Corrected)

5. **The spoke advantage IS real at 100M** — but shrinks. At LR 3e-3 (fair comparison): spokes 40.98 vs baseline 41.48 = -0.50 PPL. Earlier conclusion that it was a "LR artifact" was caused by a checkpoint overwrite (Muon run clobbered the baseline). The advantage shrinks from -0.72 (11M, 1.5%) to -0.50 (100M, 1.2%) but persists.

6. **Always sweep LR for the baseline before claiming an architecture win.** This lesson is still correct — the original -1.19 PPL gap at LR 2e-3 was inflated. Fair comparison requires matched LR.

### Beyond PPL

7. **Spokes are better calibrated** — ECE 0.0067 vs 0.0082 at 100M (18% better). The model knows what it doesn't know.

8. **Spokes trade easy tokens for hard tokens** — worse on easiest 20%, better on hardest 20%. PPL weights all equally, but hard tokens carry more information.

9. **The model discovers progressive convergence** — from uniform gate init, learns explore-early/converge-late schedule. Different at each scale (sharp binary at 11M, smooth ramp at 100M).

10. **Representations are completely different** — cosine sim ~0.01 between baseline and spoke hidden states at every layer. Same predictions, different internal structure.

### What Doesn't Work

1. **Progressive gate schedule** — starves early layers, worse than uniform
2. **Random W_up init** — adds noise, zeros is better
3. **Rank < 16** — not enough capacity per spoke to be useful
4. **2 spokes** — not enough diversity
5. **8 spokes** — too slow, marginal benefit over 4

### The Optimal v3 Config

```
FelixV3Config(
    num_spokes=4,
    spoke_rank=64,            # r64 with spoke-LR 2x (r32 at 11M)
    gate_schedule="uniform",  # all gates start at sigmoid(0) = 0.5
    embed_proj=True,          # linear projection after embedding
    # W_up initialized to zeros (default in SpokeLayer)
    # spoke-lr-mult=2.0 at 100M (uniform at 11M)
)
```

At 11M: v3_proj_r32 = **45.98 PPL** (-1.92 vs bare baseline, -0.59 vs proj-only)
At 100M: v3_proj_r64 + spoke-LR 2x = **39.15 PPL** (-2.33 vs bare baseline, -1.00 vs proj-only)

### Key Discovery: Spoke-Specific Learning Rate

Spokes are small params (2-5% of total) that need faster learning at scale.
The optimal spoke LR multiplier depends on scale:

| Scale | Backbone LR | Spoke LR mult | Spoke PPL | vs proj-only |
|-------|-------------|---------------|-----------|--------------|
| 11M | 2e-2 | 1x (uniform) | 45.98 | -0.59 |
| 100M | 3e-3 | 2x | 39.15 | -1.00 |

At 11M, backbone LR is already high enough for spokes. At 100M, spokes need 2x
to keep up with the larger backbone. This insight unlocked rank scaling that was
previously hidden (r64 failed at uniform LR, succeeds at 2x).

## Comparison Across Felix Versions

| Version | Approach | 11M (tuned LR) | 100M (tuned LR) | Verdict |
|---------|----------|-----------------|------------------|---------|
| v1 (MSPM) | Hard merge stages | 118.41 PPL (loses) | N/A at matched LR | Loses |
| v2 (Adaptive) | Soft merge + CentralPost | +2 to +6 PPL tax | LR artifact | Loses |
| **v3 (Hub-and-Spoke)** | Cheap spoke probes | **-0.72 PPL (wins)** | **-0.50 PPL (wins)** | **Wins at both scales + qualitative gains** |

v3 is the first Felix variant that beats a plain transformer at both 11M and 100M under fair conditions (matched LR). The PPL advantage is modest (-0.50 at 100M), but the qualitative analysis reveals deeper differences: better calibration, hard-token specialization, and learned convergence schedules. See `docs/qualitative_findings.md` for the full analysis.

## What's Next

1. **100M with embed_proj** — proj + spokes compound at 11M. Test at 100M where spokes already give -1.19
2. **500M+ scale test** — the advantage grew from 11M to 100M. Does it keep growing?
3. **Larger rank at scale** — at 100M, rank 32 is only 2.4% overhead. Rank 64 or 128 might be optimal
4. **Spoke attention** — replace linear down/up with single-head cross-attention for richer probes
5. **Different activation** — SwiGLU-style spokes (two down projections, gated)
6. **Spoke dropout** — randomly drop spokes during training for regularization

## Technical Notes

- All experiments on AMD RX 7800 XT (16GB VRAM), PyTorch 2.9.1+ROCm
- WikiText-103, seq_len=512, batch=8, grad_accum=8 (effective batch 64)
- 11M: LR 2e-2, 100M: LR 2e-3 (both from v2 sweep)
- torch.compile enabled for all runs
- Agreement computed with no_grad (diagnostic only, not part of loss)
- Code: `felix_lm/v3/` (config.py, spokes.py, model.py), 70 tests passing
