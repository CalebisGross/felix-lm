# Felix-LM Ablation Study

## 1. Overview

This document records a systematic ablation study of Felix-LM, a Multi-Stream Progressive Merging
(MSPM) architecture for causal language modeling. The study isolates the contribution of each
architectural component — attention type, deep supervision, stream count, layer distribution, merge
mechanism design, and regularization strategy — relative to a parameter-matched single-stream
transformer baseline (M0).

The central research question: **Does progressive multi-stream merging improve language modeling,
and if so, which design choices are load-bearing?**

## 2. Experimental Protocol

### 2.1 Common Training Setup

All experiments use a fixed training protocol to ensure fair comparison:

| Parameter | Value |
|-----------|-------|
| Dataset | WikiText-103 (train / validation / test) |
| Tokenizer | GPT-2 BPE (V = 50,257) |
| Sequence length | 512 tokens |
| Effective batch size | 32 (micro-batch 8, gradient accumulation 4) |
| Optimizer | AdamW (beta1=0.9, beta2=0.999, eps=1e-8) |
| Learning rate | 3e-4, cosine decay to 0 |
| Warmup | 1000 steps (linear) |
| Weight decay | 0.1 |
| Gradient clipping | max norm 1.0 |
| Precision | FP32 |
| Hardware | AMD Radeon RX 7800 XT (16GB VRAM), ROCm 7.1.1 |

### 2.2 Evaluation Metric

All results report **test-set perplexity** (PPL = exp(cross-entropy loss)). Lower is better.
Per-stage perplexities are reported where deep supervision is enabled, computed from each exit
head's logits against the same targets.

### 2.3 Directional Tests

Full training runs (3 epochs, ~21,500 steps) are expensive. For exploratory experiments that test a
single variable against the established best config (M2-nosup), we use **2-epoch directional tests**
(~14,000 steps). These are sufficient to determine if a change is clearly beneficial or harmful,
though final PPL numbers are higher than a full 3-epoch run due to incomplete training.

To enable comparison: M2-nosup at 2 epochs achieves approximately 136 PPL (estimated from its
training curve). Two-epoch experiments should be compared against this reference, not the final
3-epoch M2-nosup result of 118.41.

### 2.4 Parameter Budget

All configs target ~11M parameters. The embedding matrix (50,257 x 128 = 6.4M params) dominates
the budget, leaving ~4.6M for transformer layers, merges, and exit heads. This is important context:
at this scale, the "multi-stream tax" (duplicate stream processing, merge projections) consumes a
proportionally large share of the non-embedding budget.

---

## 3. Baseline

### Experiment 1: M0 — Single-Stream Transformer

**Config:** `m0` | **Params:** 11,172,736 | **Date:** 2026-03-02 | **Epochs:** 3

A standard pre-norm transformer with 18 layers, dim=128, 4 heads, full causal attention, and
standard RoPE. This is the parameter-matched control against which all MSPM variants are measured.

| Split | Loss | PPL |
|-------|------|-----|
| Test | 4.7506 | **115.66** |

M0 allocates its entire non-embedding budget to a single deep stack of transformer layers. All MSPM
variants must justify their structural overhead against this simple, strong baseline.

---

## 4. Primary Architecture

### Experiment 2: M2 — MSPM-HETERO (Original Design)

**Config:** `m2` | **Params:** 11,092,608 | **Date:** 2026-03-01 | **Epochs:** 3

The original Felix-LM architecture as specified in the design document:
- **Stage 0:** 4 streams, dim=64, 4 layers, linear attention
- **Stage 1:** 2 streams, dim=128, 4 layers, sliding-window attention (w=64)
- **Stage 2:** 1 stream, dim=128, 5 layers, full causal attention
- Deep supervision ON (weights: 0.15 / 0.25 / 0.60)
- Gated merge with bias init = 1.0

| Split | Loss | PPL | Stage 0 PPL | Stage 1 PPL | Stage 2 PPL |
|-------|------|-----|-------------|-------------|-------------|
| Validation | 5.0325 | 153.31 | 317.01 | 138.88 | 124.37 |
| Test | 5.0116 | 150.14 | 309.89 | 136.25 | 121.81 |

**Result:** 150.14 PPL (+29.8% vs M0). The progressive stage improvement (309 -> 136 -> 122)
confirms that the merge mechanism successfully integrates stream information. However, the overall
weighted loss is substantially worse than the baseline due to the early-stage exit heads dragging
down training. Stage 2 alone (121.81) is much closer to M0.

---

## 5. Ablation Studies

### 5.1 Attention Type

#### Experiment 3: M2-FULLCAUSAL

**Config:** `m2_fullcausal` | **Params:** 11,092,608 | **Date:** 2026-03-02 | **Epochs:** 3
**Control:** M2 (Exp 2) | **Variable:** Replace heterogeneous attention with full causal in all stages

| Split | Loss | PPL | Stage 0 PPL | Stage 1 PPL | Stage 2 PPL |
|-------|------|-----|-------------|-------------|-------------|
| Test | 4.8605 | 129.09 | 244.51 | 118.07 | 107.49 |

**Analysis:** Switching to full causal attention closes ~60% of the M2-to-M0 gap (150 -> 129),
confirming that linear attention in Stage 0 was a significant bottleneck under deep supervision.
Notably, the Stage 2 exit head (107.49) **outperforms M0 (115.66)** — evidence that multi-stream
merging can produce better representations than a single stream, but the weighted supervision loss
obscures this.

**Post-training diagnostics:**
- Gate bias learned from init=1.0 to ~0.65 (~35% selectivity acquired)
- Both stream halves gated symmetrically (no stream dominance)
- Merge projection matrices near full rank (97/128 and 106/128)
- Pairwise cosine similarity between stream Q-weights ~0.00 (fully orthogonal specialization)
- Embedding projections also near-orthogonal across streams

### 5.2 Deep Supervision

#### Experiment 4: M2-NOSUP

**Config:** `m2_nosup` | **Params:** 11,092,608 | **Date:** 2026-03-03 | **Epochs:** 3
**Control:** M2 (Exp 2) | **Variable:** Remove deep supervision loss (train only on final output)

| Split | Loss | PPL |
|-------|------|-----|
| Test | 4.7742 | **118.41** |

**Analysis:** Removing deep supervision is the single largest improvement in the entire ablation
study: **-31.73 PPL** vs M2 with supervision. This narrows the gap to M0 to just 2.75 PPL (+2.4%).

Deep supervision was actively harmful at this scale. Forcing early stages to produce token
predictions creates a conflicting objective: the exit heads need representations that are
linearly decodable to vocabulary logits, while the merge mechanism needs representations optimized
for information fusion. Without supervision, early stages are free to learn whatever intermediate
representations best serve the downstream merge, resulting in dramatically better final output.

This is the **best multi-stream configuration** and serves as the control for all subsequent
directional experiments.

#### Experiment 5: M2-BEST (Fullcausal + No Supervision)

**Config:** `m2_best` | **Params:** 11,092,608 | **Date:** 2026-03-03 | **Epochs:** 3
**Control:** M2-nosup (Exp 4) | **Variable:** Combine both improvements (fullcausal + nosup)

| Split | Loss | PPL |
|-------|------|-----|
| Test | 4.7936 | 120.74 |

**Analysis:** The two improvements are **not additive** — they are mildly antagonistic. Full causal
attention actually hurts by +2.33 PPL when deep supervision is removed. This reveals an interaction
effect: without supervision forcing token prediction at early stages, linear attention in Stage 0
acts as a beneficial regularizer. It constrains Stage 0 to capture broad, low-frequency patterns
(which is all linear attention can express), leaving fine-grained token prediction to the later
full-causal stages. This is exactly the hierarchical specialization the architecture was designed for.

**Conclusion:** The heterogeneous attention schedule (linear -> sliding-window -> full-causal) is
architecturally motivated, not a compromise. It naturally enforces a coarse-to-fine processing
hierarchy across stages.

### 5.3 Stream Count

#### Experiment 6: M2-2STREAM (Fewer, Wider Streams)

**Config:** `m2_2stream` | **Params:** 11,566,848 | **Date:** 2026-03-03 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** 2 streams (dim=128) instead of 4 (dim=64)

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.0897 | 162.34 |

#### Experiment 7: M2-8STREAM (More, Narrower Streams)

**Config:** `m2_8stream` | **Params:** 10,454,144 | **Date:** 2026-03-04 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** 8 streams (dim=32) with 4 stages of merging

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.1655 | 175.13 |

**Analysis (combined):** Stream count has a clear optimum at 4. Both directions — fewer wider
streams (2x128) and more narrower streams (8x32) — are substantially worse. The 2-stream variant
suffers from insufficient diversity: two streams cannot span enough of the representation space
to benefit from merging. The 8-stream variant suffers from insufficient per-stream capacity:
dim=32 is too narrow for meaningful representation at any attention type. Additionally, 8-stream
training is ~27% slower (3.5 vs 4.8 it/s) due to the additional merge stages.

The 4-stream design at dim=64 hits the sweet spot between diversity (enough streams for orthogonal
specialization) and capacity (enough dimensions per stream for expressive representations).

### 5.4 Layer Distribution

#### Experiment 8: M2-NOSUP-BACKLOADED (Deeper Final Stage)

**Config:** `m2_nosup_backloaded` | **Params:** 11,092,096 | **Date:** 2026-03-03 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Layer split 2/4/7 instead of 4/4/5

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.1030 | 164.52 |

**Analysis:** Backloading layers into Stage 2 hurts substantially. With only 2 layers in Stage 0,
streams cannot develop the distinct representations needed for effective merging. The merge
mechanism receives nearly-identical inputs and reduces to an expensive identity operation.
The balanced 4/4/5 split ensures each stage has enough depth for its role: Stage 0 for stream
specialization, Stage 1 for intermediate fusion, Stage 2 for final refinement.

### 5.5 Merge Mechanism

#### Experiment 9: M2-NOSUP-GATE0 (Gate Initialization)

**Config:** `m2_nosup_gate0` | **Params:** 11,092,608 | **Date:** 2026-03-05 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Gate bias init = 0.0 (sigmoid(0) = 0.5) vs default 1.0 (sigmoid(1) = 0.73)

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.1203 | 167.38 |

**Analysis:** Starting with gates half-open (0.5) instead of mostly-open (0.73) causes a large
degradation. This is consistent with the residual learning principle: the merge should start close
to a pass-through operation and learn selective filtering. With gates at 0.5, half the input signal
is attenuated from step 0, creating an information bottleneck that the model struggles to recover
from. The default init of 1.0, which allows ~73% of each dimension through, provides enough
information flow for early training while leaving room for the gates to learn selectivity.

#### Experiment 10: M2-TCG (Token-Conditional Gating)

**Config:** `m2_tcg` | **Params:** 11,092,800 | **Date:** 2026-03-05 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace static linear gate with bottleneck MLP (2d -> d -> 2d, ReLU)

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.1202 | 167.37 |

**Analysis:** Content-dependent gating provides no benefit over the static gate. The bottleneck MLP
introduces a nonlinearity (ReLU) in the gating path that may impede gradient flow, and the
additional parameters (~200) are negligible at this scale. The static sigmoid gate — which applies
a uniform, learned per-dimension mask — is sufficient. This suggests that the merge's role is
primarily dimensional selection (which features to keep), not content-dependent routing (which
features to keep *for this token*).

#### Experiment 12: M2-RESIDUAL (Residual Skip Connection Across Merge)

**Config:** `m2_residual` | **Params:** 11,100,800 | **Date:** 2026-03-05 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Add residual path: output = mean(streams) + gated_projection(streams). For Stage 0->1 merge (64->128), the skip path uses a learned linear projection; for Stage 1->2 (128->128), the skip is a parameter-free mean.

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.1131 | 166.19 |

**Analysis:** The residual merge path does not improve performance, achieving 166.19 PPL — roughly
+30 PPL above the 2-epoch M2-nosup reference (~136). The hypothesis was that the merge is an
information bottleneck lacking a residual path, but the result suggests the bottleneck is not in
gradient flow. The gated projection with bias=1.0 already starts mostly-open (Exp 9 confirmed
this matters), which functions as a soft residual: most information passes through at initialization.
Adding an explicit skip connection on top of this creates redundancy rather than complementarity —
the model must now learn to use *both* the skip and gated paths, which may actually make
optimization harder by doubling the number of ways information can flow through the merge. The
simple gate-only design, initialized to be mostly transparent, is the right inductive bias.

**Combined merge mechanism analysis (Exps 9, 10, 12):** Three different attempts to improve the
merge — better initialization (gate0), richer gating (TCG), and residual paths (residual) — all
degraded performance. The original design (static sigmoid gate, bias=1.0, no skip connection) is
a robust optimum. The merge's simplicity appears to be a feature, not a limitation: it forces all
representational work into the transformer layers before and after the merge, where the model has
the most capacity.

### 5.6 Regularization

#### Experiment 11: M2-STREAMDROP (Stream Dropout)

**Config:** `m2_streamdrop` | **Params:** 11,092,608 | **Date:** 2026-03-05 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** 15% probability of zeroing one stream per merge pair (surviving stream scaled 2x)

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.1608 | 174.30 |

**Analysis:** Stream dropout is counterproductive. The streams already specialize into orthogonal
representations naturally (pairwise cosine ~0.00 observed in Exp 3), so forcing independence
via dropout is redundant. Worse, it destabilizes the merge layer: the gated projection learns
to expect consistent input patterns from both streams, and randomly zeroing one breaks this
assumption. The 2x scaling of the survivor only partially compensates — the gate values learned
for two-stream input are inappropriate for single-stream input. This is analogous to why dropout
inside residual connections can hurt: the downstream computation depends on the full signal path.

### 5.7 Cross-Stream Communication

#### Experiment 13: M2-CROSSATTN (Cross-Stream Attention at Merge)

**Config:** `m2_crossattn` | **Params:** ~11.4M | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Add bidirectional cross-attention between stream pairs before gated merge

**Hypothesis:** Allowing streams to attend to each other before merging enables information exchange
that improves merge quality. Currently streams are completely independent until the gated projection —
cross-attention adds a "soft communication" channel. This tests whether the merge benefits from
streams coordinating what they contribute, rather than blindly fusing independent representations.

#### Experiment 14: M2-SHARED-DEEP (Shared Weights + Deeper Final Stage)

**Config:** `m2_shared_deep` | **Params:** ~11M | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** All Stage 0 streams share transformer weights; freed params go to Stage 2 (5->10 layers)

**Hypothesis:** Stream diversity may not require independent weights — initialization diversity
alone may suffice. Weight sharing frees ~1.5M params for a deeper Stage 2. This tests whether
the architecture benefits more from stream specialization (independent weights) or post-merge
depth (shared weights + deeper final stage).

### 5.8 Novel Merge Algorithms

These experiments challenge the core assumption that concat->gate->project is the optimal merge.

#### Experiment 15: M2-GEOMETRIC (Multiplicative Feature Conjunction)

**Config:** `m2_geometric` | **Params:** 11,010,560 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace gated merge with geometric mean: sign(a*b)*sqrt(|a*b|)

**Hypothesis:** The standard merge aggregates information (addition-like). Geometric merge creates
feature conjunction — a feature must be present in BOTH streams to survive. This is fundamentally
different: it rewards agreement rather than accumulation. If streams specialize into complementary
features, conjunction may produce more discriminative merged representations.

#### Experiment 16: M2-HADAMARD (Fixed Orthogonal Rotation + Learned Scaling)

**Config:** `m2_hadamard` | **Params:** 10,961,408 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace learned projection with fixed Hadamard matrix + learned diagonal

**Hypothesis:** The standard merge has O(d^2) params in the projection. Hadamard merge separates
"how to mix" (fixed Hadamard — maximally mixes all dimensions equally) from "how much to keep"
(learned diagonal — O(d) params). If the learned projection mostly converges to something close
to an orthogonal mixing matrix (as polar decomposition analysis in Section 3.3 suggests), then
the fixed Hadamard may be sufficient, with ~99% fewer merge params.

#### Experiment 17: M2-ANTIMERGE (Competitive Stream Selection)

**Config:** `m2_antimerge` | **Params:** 11,010,690 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace merge with per-token stream competition (winner routes forward via STE)

**Hypothesis:** Instead of combining streams, make them compete. A learned scorer picks the
better stream per-token, and the winner routes forward via straight-through estimator. This
creates evolutionary selection pressure: streams that produce better representations survive.
The competitive dynamic may drive stronger specialization than cooperative merging.

#### Experiment 18: M2-NOISE-MERGE (Variational Merge Bottleneck)

**Config:** `m2_noise_merge` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Inject N(0, 0.1) noise at merge output during training

**Hypothesis:** Gaussian noise at the merge bottleneck forces streams to encode robust, redundant
information — features that survive noise corruption. This is analogous to a variational bottleneck
(VAE-like). At inference (no noise), the clean merged signal may be higher quality because the
streams were trained to be noise-robust. The key question is whether the noise level (std=0.1)
is calibrated correctly: too low and it's invisible, too high and it destroys the signal.

### 5.9 Training Objectives

#### Experiment 19: M2-DIVERGE-LOW (Contrastive Stream Divergence, lambda=0.01)

**Config:** `m2_diverge_low` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Auxiliary loss maximizing cosine distance between stream pairs (weight=0.01)

#### Experiment 20: M2-DIVERGE-MID (Contrastive Stream Divergence, lambda=0.1)

**Config:** `m2_diverge_mid` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Same as Exp 19 but weight=0.1

#### Experiment 21: M2-DIVERGE-HIGH (Contrastive Stream Divergence, lambda=0.5)

**Config:** `m2_diverge_high` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Same as Exp 19 but weight=0.5

**Hypothesis (Exps 19-21):** Streams already specialize naturally (pairwise cosine ~0.00 in Exp 3),
but this is emergent, not optimized. An explicit divergence loss pushes streams apart in
representation space, which may produce MORE diverse specialization than what emerges from
random initialization alone. The three weight levels test whether the optimal push is gentle
(0.01, nudge), medium (0.1, clear pressure), or aggressive (0.5, dominant objective). If the
natural specialization is already optimal, all three should hurt; if not, there should be an
optimal weight that improves PPL.

#### Experiment 22: M2-FROZEN-INIT (Frozen Stream Projections)

**Config:** `m2_frozen_init` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Freeze stream initialization projections (W_s^init) for first 7000 steps (~1 epoch)

**Hypothesis:** Stream projections are initialized orthogonally, creating maximally diverse initial
representations. During training, gradient updates may erode this diversity as all projections
drift toward the same loss minimum. Freezing them for the first epoch forces the transformer
layers to "lock in" to the orthogonal initialization structure before the projections can drift.
Inspired by biological cell differentiation: cells commit to a lineage early, then specialize
within that lineage.

### 5.10 Structural Stream Diversity

#### Experiment 23: M2-ASYMMETRIC (Structurally Heterogeneous Streams)

**Config:** `m2_asymmetric` | **Params:** 10,829,952 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Each Stage 0 stream has a different block: attn-only, FFN-only, wide-window (w=256), narrow-window (w=8)

**Hypothesis:** Current streams differ only in initialization; their architecture is identical.
By giving each stream a fundamentally different computation (attention-only, FFN-only, etc.),
streams CANNOT learn the same function even if gradients push them that way. The merge must
synthesize genuinely complementary views: relational structure (attn-only), local features
(FFN-only), broad context (wide window), and n-gram patterns (narrow window). This is the
most radical stream diversity possible within the transformer framework.

#### Experiment 24: M2-BOTTLENECK (Information Compression at Merge)

**Config:** `m2_bottleneck` | **Params:** 11,063,936 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Merge projection goes through 4x compression bottleneck (2*d -> d/4 -> d)

**Hypothesis:** The standard merge preserves too much information, preventing the model from
learning to prioritize what matters. A radical bottleneck (128 -> 32 -> 128) forces the merge
to distill only the essence of what both streams agree on. This is information-theoretic: the
bottleneck acts as an implicit rate constraint, keeping only the most mutually informative features.

### 5.11 Combination Experiments

These test whether individually neutral/negative changes become positive when combined.

#### Experiment 25: M2-BOTTLENECK-HALF (Moderate Bottleneck)

**Config:** `m2_bottleneck_half` | **Params:** 11,084,416 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) and M2-bottleneck (Exp 24) | **Variable:** Bottleneck ratio 0.5 instead of 0.25

**Hypothesis:** If the 4x bottleneck (Exp 24) is too aggressive, a 2x bottleneck may hit the
sweet spot — enough compression to force distillation, but enough capacity to preserve essential
information.

#### Experiment 26: M2-ASYMMETRIC-BOTTLENECK

**Config:** `m2_asymmetric_bottleneck` | **Params:** 10,801,280 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Asymmetric streams (Exp 23) + bottleneck merge (Exp 24)

**Hypothesis:** Structurally different streams produce genuinely different representations,
and the bottleneck forces the merge to find the common signal. The combination may be
synergistic: diversity creates richer inputs, and compression forces useful fusion.

#### Experiment 27: M2-ASYMMETRIC-DIVERGE

**Config:** `m2_asymmetric_diverge` | **Params:** 10,829,952 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Asymmetric streams + divergence loss (0.1)

**Hypothesis:** Asymmetric architecture already enforces structural diversity. Adding explicit
divergence loss on top may be redundant (pushing streams apart when they're already constrained
to be different) or synergistic (ensuring the learnable parameters within each stream also
diverge, not just the architecture).

#### Experiment 28: M2-GEOMETRIC-DIVERGE

**Config:** `m2_geometric_diverge` | **Params:** 11,010,560 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Geometric merge + divergence loss (0.1)

**Hypothesis:** Geometric merge rewards feature conjunction (both streams must agree). Divergence
loss pushes streams apart. The combination tests whether it's useful to have maximally different
streams whose points of agreement are maximally informative.

### 5.12 Wild Card Experiments

#### Experiment 29: M2-PROGRESSIVE-UNFREEZE (Back-to-Front Training)

**Config:** `m2_progressive_unfreeze` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Only Stage 2 trainable initially; Stage 1 unfreezes at 1/3 training, Stage 0 at 2/3

**Hypothesis:** Standard training optimizes all stages simultaneously, which may cause early
stages to adapt to the (changing) expectations of later stages. Progressive unfreezing first
establishes a strong final stage, then tunes intermediate processing to feed it better input.
This curriculum approach (simple→complex) has worked well in transfer learning (ULMFiT).

#### Experiment 30: M2-STREAM-PERMUTE (Random Merge Pairings)

**Config:** `m2_stream_permute` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Randomly shuffle which streams pair at merge time each batch

**Hypothesis:** Fixed merge pairings (0↔1, 2↔3) allow co-adaptation: streams learn to produce
outputs that complement their specific partner. Random permutation forces each stream to produce
representations that are useful when merged with ANY other stream. This is a stronger form of
stream independence than dropout — the merge still receives full signal, but from unpredictable
sources.

#### Experiment 31: M2-ASYMMETRIC-V2 (Alternative Stream Types)

**Config:** `m2_asymmetric_v2` | **Params:** 10,829,952 | **Date:** pending | **Epochs:** 2
**Control:** M2-asymmetric (Exp 23) | **Variable:** Different stream combination: linear + full-causal + FFN-only + attn-only

**Hypothesis:** Tests whether the specific choice of stream architectures matters, or only that
they differ. If M2-asymmetric (Exp 23) and this variant produce similar results, the key insight
is "structural diversity" generically; if they differ substantially, the specific stream types
matter and should be tuned.

---

## 6. Future Experiments

### Scale to 100M Parameters

**Dataset:** FineWeb-Edu | **Hardware:** MI300X
**Blocked on:** Finalizing best small-scale configuration from the MI300X batch run.

At 100M parameters, the embedding matrix becomes a smaller fraction of the total budget, and the
"multi-stream tax" is proportionally reduced. The architectural advantages of MSPM may become more
apparent at this scale.

---

## 7. Summary

### 7.1 Results Table

| # | Model | Key Variable | Epochs | Test PPL | vs M0 | vs M2-nosup |
|---|-------|-------------|--------|----------|-------|-------------|
| 1 | M0 (baseline) | — | 3 | **115.66** | — | — |
| 2 | M2 | Original MSPM | 3 | 150.14 | +29.8% | — |
| 3 | M2-fullcausal | All full causal attn | 3 | 129.09 | +11.6% | — |
| 4 | **M2-nosup** | **No deep supervision** | **3** | **118.41** | **+2.4%** | **ref** |
| 5 | M2-best | Fullcausal + nosup | 3 | 120.74 | +4.4% | +2.0% |
| 6 | M2-2stream | 2 streams (dim=128) | 2 | 162.34 | +40.4% | — |
| 7 | M2-8stream | 8 streams (dim=32) | 2 | 175.13 | +51.4% | — |
| 8 | M2-nosup-backloaded | Layers 2/4/7 | 2 | 164.52 | +42.2% | — |
| 9 | M2-nosup-gate0 | Gate bias = 0.0 | 2 | 167.38 | +44.7% | — |
| 10 | M2-tcg | MLP gate | 2 | 167.37 | +44.7% | — |
| 11 | M2-streamdrop | Stream dropout 15% | 2 | 174.30 | +50.7% | — |
| 12 | M2-residual | Residual merge path | 2 | 166.19 | +43.7% | — |

### 7.2 Key Findings

1. **Deep supervision is the dominant failure mode.** Removing it yields a 32-point PPL improvement
   (Exp 4), the largest single-variable effect in the study. The supervision objective conflicts
   with learning merge-optimal representations.

2. **The original M2-nosup design is a robust local optimum.** Every structural variation tested —
   attention type, stream count, layer distribution, gate mechanism, regularization — made
   performance worse. The design document's choices (4 streams, 4/4/5 layers, hetero attention,
   simple gate with bias=1.0) are well-calibrated.

3. **Heterogeneous attention enforces beneficial specialization.** Linear attention in Stage 0
   acts as an inductive bias for coarse pattern capture when supervision is off (Exp 5). The
   coarse-to-fine hierarchy (linear -> sliding-window -> full-causal) is the right design.

4. **4 streams is optimal at 11M scale.** Both 2 and 8 streams are substantially worse (Exps 6-7).
   The sweet spot balances per-stream capacity against representational diversity.

5. **The merge is an information bottleneck.** It is the only component without a residual path.
   Gate initialization matters (Exp 9): gates must start mostly-open to preserve information flow.
   Attempts to make the gate more expressive (TCG, Exp 10) or regularize streams (dropout, Exp 11)
   both hurt, suggesting the merge is already operating at its optimum given its simple design.

6. **Multi-stream merging works, but the "tax" is high at small scale.** The embedding matrix
   consumes 6.4M of the 11M budget. M0 gets 18 full-width layers from the remaining 4.6M; M2
   gets roughly 9 effective full-width layers after accounting for 4 narrow streams and merge
   projections. The 2.4% PPL gap may narrow at larger scales where the embedding fraction shrinks.

7. **Stage 2 output can beat M0.** Under full causal attention + deep supervision (Exp 3), the
   Stage 2 exit head achieves 107.49 PPL vs M0's 115.66. The merged representation is genuinely
   richer — the challenge is making the overall training objective (not just the final stage)
   exploit this.

### 7.3 Open Questions

- Does the M2-nosup advantage grow at 100M+ parameters where the multi-stream tax is amortized?
- Can any of the novel merge algorithms (geometric, Hadamard, competitive) beat the simple gated merge?
- Is forced stream diversity (asymmetric, divergence loss) better than emergent diversity?
- Does the merge benefit from information compression (bottleneck) or is full-rank projection needed?
- Can training curriculum (frozen init, progressive unfreeze) improve optimization without new params?
- Are combination effects (asymmetric + bottleneck, geometric + divergence) synergistic or redundant?
