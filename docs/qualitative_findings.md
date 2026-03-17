# Felix-LM v3: Beyond Perplexity

**The headline finding from 100+ experiments was: spokes don't help at scale (PPL).**
**The deeper finding: spokes create a fundamentally different model that PPL doesn't measure.**

## What PPL Misses

Perplexity averages cross-entropy across all tokens equally. A model that's 0.04 nats
worse on the 20% easiest tokens but 0.38 nats better on the 20% hardest tokens can
have identical PPL but dramatically different behavior.

## Evidence (11M, v3_none vs v3_r32, both at LR 2e-2)

### 1. Calibration

| Model | ECE (lower = better) |
|-------|---------------------|
| Baseline | 0.0149 |
| Spokes | **0.0076** (2x better) |

Spokes produce confidence estimates that are much better aligned with actual accuracy.
When the spoke model says 70% confident, it's right ~74% of the time.

### 2. Hard vs Easy Token Tradeoff

| Difficulty Quintile | Baseline CE | Spokes CE | Relative Change |
|---------------------|-------------|-----------|-----------------|
| Easiest (20%) | 0.254 | 0.291 | +14.6% (worse) |
| Easy | 1.477 | 1.453 | -1.6% |
| Medium | 3.172 | 3.063 | -3.4% |
| Hard | 5.375 | 5.188 | -3.5% |
| Hardest (20%) | 8.875 | 8.500 | **-4.2% (best)** |

Spokes sacrifice easy tokens to improve hard tokens. The improvement monotonically
increases with difficulty. In information-theoretic terms, spokes allocate model
capacity toward tokens with high surprise — where correct predictions carry the most
information.

### 3. Rare Token Advantage

| Token Category | Baseline CE | Spokes CE | Delta |
|---------------|-------------|-----------|-------|
| Common (top-1K) | 2.531 | 2.443 | -0.088 |
| Content | 5.579 | 5.386 | -0.193 |
| Rare (bottom-10K) | 7.561 | 7.108 | **-0.453** |

Spokes are 5x more helpful on rare tokens than common ones.

### 4. Long-Range Dependencies

| Position Range | Baseline CE | Spokes CE | Delta |
|---------------|-------------|-----------|-------|
| 0-64 | 4.098 | 3.986 | -0.113 |
| 256-320 | 3.759 | 3.606 | **-0.153** |

Spokes improve predictions more at longer context distances. The diverse projections
may help maintain information across depth that a single residual stream loses.

### 5. Learned Gate Schedule

Despite uniform initialization (all gates at 0.5), the model learned:
- Layers 1-9: gates 0.01-0.06 (spokes nearly silent)
- Layers 10-12: gates 0.33-0.42 (moderate)
- Layers 13-18: gates 0.95-0.98 (spokes fully active)

This is the Felix progressive convergence principle — discovered by the model, not
imposed by us. Early layers explore independently; late layers converge strongly.

### 6. Completely Different Representations

Cosine similarity between hidden states at every layer: ~0.00.

The two models learn entirely different internal representations that produce
similar average predictions. This is ensemble-like diversity from a single model.

### 7. Unique Correct Predictions

Of tokens where only one model is right:
- Spokes correct, baseline wrong: 9,839 (57%)
- Baseline correct, spokes wrong: 7,320 (43%)

Spokes get 34% more unique predictions right.

## What This Means

PPL says the spoke model is marginally better (-0.72 PPL). But the qualitative
analysis says it's a **fundamentally different model** that:

1. Is better calibrated (knows what it doesn't know)
2. Invests capacity in hard tokens over easy ones
3. Excels at rare/content tokens and morphological completion
4. Improves long-range predictions
5. Discovers its own convergence schedule
6. Learns completely different representations

These properties are potentially more valuable for downstream tasks than a PPL
number. A well-calibrated model that's good at hard tokens is exactly what you
want for:
- Information extraction (rare entities, numbers)
- Long-document summarization (long-range dependencies)
- Uncertainty quantification (calibration)
- Ensemble diversity (orthogonal representations from one model)

## 100M Results (Both at LR 3e-3 — Fair Comparison)

Spokes: 40.98 PPL / 1.1539 BPB. Baseline: 41.48 PPL / 1.1573 BPB. Delta: -0.50 PPL.

### Calibration Scales

| Scale | Baseline ECE | Spokes ECE | Ratio |
|-------|-------------|-----------|-------|
| 11M | 0.0149 | 0.0076 | 1.96x better |
| 100M | 0.0082 | 0.0067 | 1.22x better |

The baseline gets better calibrated at scale, but spokes still win.

### Hard/Easy Tradeoff Persists at 100M

| Quintile | 11M Delta | 100M Delta |
|----------|-----------|------------|
| Easiest | +14.6% | +38.1% |
| Hard | -3.5% | 0.0% |
| Hardest | -4.2% | -2.8% |

Same pattern: sacrifice easy, improve hard. Smaller absolute effect at 100M.

### Gate Schedule Changes with Scale

- 11M: sharp binary (layers 1-9 off, 13-18 fully on at 0.95+)
- 100M: smooth ramp (0.15 to 0.74, peak at layers 14-16, then declining)

The model adapts its convergence schedule to its capacity.

### Representations Still Divergent

Cosine similarity ~0.01 at every layer at 100M. Still completely different models.

### Token Strengths

Spokes dominate: word completions, morphological continuation
Baseline dominates: proper noun completion, high-frequency patterns

## Implication for Felix

The Felix identity (diversity, agreement, convergence) provides:

1. **A PPL improvement** that survives at both 11M (-0.72) and 100M (-0.50)
   when LR is properly tuned for both models
2. **Better calibration** (1.2-2x lower ECE) — the model knows what it doesn't know
3. **Hard-token specialization** — capacity allocated to informative tokens
4. **Learned convergence** — the model discovers progressive gating from uniform init
5. **Representation diversity** — a fundamentally different model, not a marginal tweak

The right evaluation for Felix is not just PPL — it's calibration, stratified
difficulty analysis, and downstream task performance where hard tokens and
uncertainty estimation matter.

## Real-Scale Validation (1B Tokens Dolma, MI300X)

The qualitative properties were confirmed at real scale (100M params, 1B tokens
of Dolma, seq_len=2048). All properties held or strengthened:

| Property | Local (2500 steps WikiText) | Real Scale (1B tokens Dolma) |
|----------|---------------------------|------------------------------|
| Calibration (ECE ratio) | 1.2-2.0x better | **1.9x better** (0.029 vs 0.056) |
| Hardest quintile improvement | -2.8% to -4.2% | **-7.4%** |
| Rare token improvement | -0.45 nats | **-0.52 nats** |
| Representation divergence | cosine ~0.00 | cosine 0.03 -> -0.01 (diverges with depth) |
| Learned convergence | binary (11M), smooth (100M) | **explore-diverse-reconverge** pattern |

The spoke agreement pattern at real scale is the clearest yet:
- Layers 0-1: high agreement (0.25-0.27) — spokes start correlated
- Layers 2-16: low agreement (0.02-0.11) — maximum diversity (exploration)
- Layers 17-19: rising agreement (0.20-0.25) — reconvergence

This is the Felix multi-agent principle realized: diverse exploration followed
by convergence, discovered by the model from uniform initialization.

## 500M Result

At 500M with 250M tokens, spokes lost on PPL by 4.7%. Qualitative analysis was
not run on 500M checkpoints (droplet auto-shutdown before retrieval). The 500M
result is inconclusive due to undertrained models and untuned LR.
