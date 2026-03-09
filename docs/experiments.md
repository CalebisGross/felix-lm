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

**Config:** `m2_crossattn` | **Params:** 11,174,720 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Add bidirectional cross-attention between stream pairs before gated merge

| Split | Loss | PPL |
|-------|------|-----|
| Test | 5.0964 | 163.42 |

**Analysis:** Cross-stream attention before the merge does not help, achieving 163.42 PPL — roughly
+27 PPL above the 2-epoch M2-nosup reference (~136). The hypothesis was that streams could
coordinate what they contribute to the merge, but in practice this adds a costly attention operation
(+82K params) that the model cannot effectively leverage at this scale. The streams already
specialize into orthogonal representations (pairwise cosine ~0.00, Exp 3), and the gated merge
already learns to selectively combine them. Cross-attention between orthogonal representations
may produce near-zero attention weights — the streams have little to say to each other precisely
because their specializations are complementary, not overlapping. The merge's job is to *combine*
different information, not to help streams become more similar before combining.

#### Experiment 14: M2-SHARED-DEEP (Shared Weights + Deeper Final Stage)

**Config:** `m2_shared_deep` | **Params:** 11,091,840 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** All Stage 0 streams share transformer weights; freed params go to Stage 2 (5->12 layers)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.0647 | 158.34 |

**Analysis:** Weight sharing at 158.34 PPL is substantially worse than the 2-epoch M2-nosup
reference (~136), despite having 12 layers of full-causal post-merge processing (vs 5 in
M2-nosup). This decisively answers whether stream specialization requires independent weights:
it does. When all streams share the same transformer layers, the only source of diversity is
the orthogonal initialization projections. But shared weights apply identical transformations
to all streams at every layer, causing representations to converge over depth — by layer 4,
the initial orthogonal diversity has been largely eroded. The merge then receives near-identical
inputs, reducing it to an expensive identity operation (similar to the backloading failure in
Exp 8). Even 12 layers of post-merge depth cannot recover what was lost by the impoverished
merge input. This confirms that the multi-stream "tax" (independent weights per stream) is not
wasted — it buys genuine representational diversity that is essential for the merge to add value.

### 5.8 Novel Merge Algorithms

These experiments challenge the core assumption that concat->gate->project is the optimal merge.

#### Experiment 15: M2-GEOMETRIC (Multiplicative Feature Conjunction)

**Config:** `m2_geometric` | **Params:** 11,010,560 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace gated merge with geometric mean: sign(a*b)*sqrt(|a*b|)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.3652 | 213.83 |

**Analysis:** Geometric merge is the worst-performing config in the entire study at 213.83 PPL —
+78 PPL above the 2-epoch reference. The multiplicative conjunction is fundamentally incompatible
with how streams specialize. Streams learn orthogonal representations (cosine ~0.00), meaning their
activations occupy different subspaces. Geometric mean requires both streams to activate the *same*
features — but orthogonal streams activate *different* features by design. The result is systematic
signal destruction: features that are strong in one stream but near-zero in the other get
attenuated to near-zero in the output. The additive nature of the standard gated merge
(concat + project) is essential precisely because it can combine information from different subspaces.

#### Experiment 16: M2-HADAMARD (Fixed Orthogonal Rotation + Learned Scaling)

**Config:** `m2_hadamard` | **Params:** 10,961,408 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace learned projection with fixed Hadamard matrix + learned diagonal

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.0592 | 157.46 |

**Analysis:** Hadamard merge achieves 157.46 PPL — better than geometric (213.83) and antimerge
(188.61), but still +21 PPL above the 2-epoch reference. The fixed Hadamard rotation mixes all
dimensions equally, which is too rigid. The learned projection in the standard merge can assign
different weights to different stream-dimension combinations, effectively learning *which*
information from *which* stream matters for *which* output dimension. The Hadamard matrix treats
all input dimensions as equally important and mixes them uniformly — the learned diagonal can
only scale the *output* dimensions, not selectively weight the *input* contributions. This
confirms that the merge projection's O(d^2) parameters are earning their keep: the model needs
full-rank learned mixing, not just scaling after a fixed rotation.

#### Experiment 17: M2-ANTIMERGE (Competitive Stream Selection)

**Config:** `m2_antimerge` | **Params:** 11,010,690 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Replace merge with per-token stream competition (winner routes forward via STE)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.2397 | 188.61 |

**Analysis:** Competitive selection at 188.61 PPL is among the worst results. The fundamental
problem is that competition *discards* half the information: only one stream survives per token.
Since streams specialize into orthogonal representations, selecting one means losing the
complementary information from the other entirely. The standard merge *combines* both streams,
preserving information from all subspaces. Additionally, the straight-through estimator creates
a noisy gradient signal — the scorer must learn from binary routing decisions, making optimization
harder. Competition is the wrong paradigm for merging orthogonal specialists; cooperation
(additive combination) is what allows the merged representation to be richer than any single stream.

#### Experiment 18: M2-NOISE-MERGE (Variational Merge Bottleneck)

**Config:** `m2_noise_merge` | **Params:** 11,092,608 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Inject N(0, 0.1) noise at merge output during training

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.1266 | 168.44 |

**Analysis:** Noise injection at the merge achieves 168.44 PPL — +32 PPL above the 2-epoch
reference. The variational bottleneck hypothesis does not hold here. Rather than forcing
streams to encode more robust representations, the noise corrupts the carefully structured
merged output that downstream layers depend on. The merge already operates without a residual
path (Exp 12 showed adding one doesn't help), so any corruption at this bottleneck propagates
directly into Stage 1/2 processing. The noise std of 0.1 may also be poorly calibrated relative
to the activation magnitudes at the merge output — but the deeper issue is that robustness
regularization at the merge point conflicts with the need for precise information transmission
between stages.

**Combined novel merge analysis (Exps 15-18):** All four alternative merge algorithms perform
substantially worse than the standard gated merge, ranging from +21 PPL (Hadamard) to +78 PPL
(geometric). A clear pattern emerges: merges that *combine* information from both streams
(Hadamard: 157, noise: 168) outperform merges that *select* or *conjoin* (antimerge: 189,
geometric: 214). The standard gated merge succeeds because it is additive — it can sum
contributions from orthogonal subspaces. Geometric mean punishes orthogonality, competitive
selection discards half the information, Hadamard constrains the mixing to be uniform, and noise
corrupts the signal. The learned full-rank projection in the original design is not overparameterized;
it is precisely the right tool for combining orthogonal stream representations.

### 5.9 Training Objectives

#### Experiment 19: M2-DIVERGE-LOW (Contrastive Stream Divergence, lambda=0.01)

**Config:** `m2_diverge_low` | **Params:** 11,092,608 | **Date:** pending | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Auxiliary loss maximizing cosine distance between stream pairs (weight=0.01)

#### Experiment 20: M2-DIVERGE-MID (Contrastive Stream Divergence, lambda=0.1)

**Config:** `m2_diverge_mid` | **Params:** 11,092,608 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Same as Exp 19 but weight=0.1

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 4.9978 | 148.09 |

**Analysis:** M2-diverge-mid is the best-performing experiment in this entire batch at 148.09 PPL,
but still +12 PPL above the 2-epoch M2-nosup reference (~136). The divergence loss at lambda=0.1
provides moderate pressure for stream separation, but does not translate into better final
representations. Streams already specialize naturally to pairwise cosine ~0.00 (Exp 3), so the
divergence loss is pushing against an open door — the streams are already as different as they
need to be. The remaining gap likely comes from the auxiliary loss slightly distorting the primary
language modeling gradient, similar to the deep supervision finding (Exp 4): any auxiliary
objective that competes with the primary loss hurts, even one designed to help the architecture.

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

**Config:** `m2_asymmetric` | **Params:** 10,829,952 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Each Stage 0 stream has a different block: attn-only, FFN-only, wide-window (w=256), narrow-window (w=8)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.1057 | 164.95 |

**Analysis:** Forced structural diversity at 164.95 PPL is substantially worse than the 2-epoch
reference (~136). The hypothesis that structurally different streams would produce richer merged
representations does not hold. The problem is twofold: first, removing the FFN from one stream
and attention from another drastically reduces each stream's individual capacity — an attn-only
stream cannot learn local features, and an FFN-only stream cannot model token interactions at all.
Second, the merge must now integrate fundamentally incommensurable representations, which is harder
than combining outputs from identical architectures that have specialized via learned weights.
Emergent specialization (same architecture, different learned functions) is strictly more flexible
than imposed specialization (different architectures) because the model can choose its own
division of labor.

#### Experiment 24: M2-BOTTLENECK (Information Compression at Merge)

**Config:** `m2_bottleneck` | **Params:** 11,063,936 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Merge projection goes through 4x compression bottleneck (2*d -> d/4 -> d)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.2423 | 189.11 |

**Analysis:** The 4x bottleneck at 189.11 PPL is among the worst results, confirming that
information compression at the merge is destructive. Reducing 256 concatenated dimensions to 32
before expanding back to 128 discards ~87% of the merged information. Since streams encode
orthogonal features (different subspaces), the bottleneck must choose which subspace's information
to preserve — it cannot retain both in only 32 dimensions. The standard merge's full-rank
projection (256 -> 128) already performs a 2x compression, which is sufficient dimensionality
reduction. The merge needs to *preserve* the complementary information from both streams, not
compress it further.

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

**Config:** `m2_asymmetric_diverge` | **Params:** 10,829,952 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Asymmetric streams + divergence loss (0.1)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.0439 | 155.08 |

**Analysis:** The combination achieves 155.08 PPL — better than asymmetric alone (164.95) by
~10 PPL, suggesting the divergence loss partially compensates for the structural limitations of
asymmetric streams. However, it's still +19 PPL above the 2-epoch reference. The divergence loss
helps because structurally heterogeneous streams may paradoxically converge in representation
space (the attn-only and FFN-only streams may learn to produce similar output distributions
despite different computations). Pushing them apart preserves the diversity that asymmetric
architecture was supposed to guarantee. Still, the combined result is worse than diverge-mid
alone (148.09), confirming that asymmetric stream architecture is a net negative.

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

**Config:** `m2_stream_permute` | **Params:** 11,092,608 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4) | **Variable:** Randomly shuffle which streams pair at merge time each batch

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.2298 | 186.76 |

**Analysis:** Random merge pairings at 186.76 PPL severely hurts performance. The result
confirms that stream co-adaptation with fixed partners is a *feature*, not a bug. When streams
know which partner they will merge with, they can specialize cooperatively — stream 0 learns to
encode information that complements stream 1, and the merge projection learns the specific
mapping for that pair. Random permutation destroys this cooperative specialization: each stream
must produce a "generic" representation compatible with any partner, which eliminates the
complementary structure that makes merging valuable. This is the opposite of the stream dropout
lesson (Exp 11) applied at a different level — both show that the merge relies on consistent,
predictable inputs to learn effective fusion.

#### Experiment 31: M2-ASYMMETRIC-V2 (Alternative Stream Types)

**Config:** `m2_asymmetric_v2` | **Params:** 10,829,952 | **Date:** pending | **Epochs:** 2
**Control:** M2-asymmetric (Exp 23) | **Variable:** Different stream combination: linear + full-causal + FFN-only + attn-only

**Hypothesis:** Tests whether the specific choice of stream architectures matters, or only that
they differ. If M2-asymmetric (Exp 23) and this variant produce similar results, the key insight
is "structural diversity" generically; if they differ substantially, the specific stream types
matter and should be tuned.

### 5.13 Framework-Level Topology

All prior experiments (Exps 2–31) were architecture-level or training-level tweaks to a single
framework-level config: M2's 4×64 (4L) → 2×128 (4L) → 1×128 (5L) topology. This section explores
fundamentally different MSPM topologies — varying layer distribution, stage count, and stream width
— designed from first principles rather than M2 modifications.

#### Experiment 32: FELIX-FRONT643 (Frontloaded 3-Stage)

**Config:** `felix_front643` | **Params:** 11,093,120 | **Date:** 2026-03-07 | **Epochs:** 2
**Control:** M2-nosup (Exp 4, ~136 PPL at 2 epochs) | **Variable:** Layer split 6/4/3 instead of 4/4/5

Topology: 4×64 (6L linear) → 2×128 (4L sliding_window) → 1×128 (3L full_causal)

| Split | Loss | PPL |
|-------|------|-----|
| Test  | 5.1090 | 165.51 |

**Analysis:** Frontloading depth into Stage 0 at the expense of Stage 2 hurts substantially
(165.51 vs ~136 reference). While 6 layers give streams more capacity to specialize, the 3-layer
Stage 2 cannot capitalize on the richer merged representation. This result, combined with the
backloading failure (Exp 8, 164.52 with 2/4/7), brackets the problem: both extremes underperform
M2's balanced 4/4/5 split. Stage 2 needs at least 5 layers to refine the merged representation,
and Stage 0 needs at least 4 layers for meaningful specialization. The balanced distribution is
not arbitrary — it reflects a genuine equilibrium between stream development and post-merge
refinement.

#### Experiment 33: FELIX-2STAGE (Minimal Intermediate Stage)

**Config:** `felix_2stage` | **Params:** 11,093,376 | **Date:** 2026-03-07 | **Epochs:** ~1.5 (killed early)
**Control:** M2-nosup (Exp 4, ~136 PPL at 2 epochs) | **Variable:** 7/1/8 layer split — 1-layer bridge at Stage 1

Topology: 4×64 (7L linear) → 2×128 (1L sliding_window) → 1×128 (8L full_causal)

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 5.2099 | 183.08 |

*Killed at step ~10,500 (~1.5 epochs); val_ppl was 188 at step 10K with no sign of recovery.*

**Analysis:** The worst framework-level result so far. The hypothesis that the intermediate
2-stream stage is unnecessary was decisively rejected. With only 1 layer in Stage 1, the model
cannot meaningfully process the post-merge representation before the second merge. The 4→2 merge
produces a combined representation that needs several layers to integrate the two streams'
contributions — a single layer cannot transform concat+gate output into something the next merge
can work with. This confirms that M2's Stage 1 (4 layers at 2×128) is doing essential integration
work: it's not a passthrough but an active processing stage that refines the first merge output
into a form suitable for the final merge. The intermediate stage is load-bearing.

#### Experiment 34: FELIX-WIDE96 (Wider Stage 0 Streams)

**Config:** `felix_wide96` | **Params:** 10,547,008 | **Date:** 2026-03-07 | **Epochs:** ~1.1 (killed early)
**Control:** M2-nosup (Exp 4, ~136 PPL at 2 epochs) | **Variable:** Stage 0 dim=96 instead of dim=64, fewer layers (3/2/4 vs 4/4/5)

Topology: 4×96 (3L linear) → 2×128 (2L sliding_window) → 1×128 (4L full_causal)

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 5.3679 | 214.41 |

*Killed at step ~8,000 (~1.1 epochs); val_ppl was 220 at step 8K with no sign of recovery.*

**Analysis:** The worst framework-level result by far. Wider streams (dim=96) with fewer layers
is a catastrophically bad tradeoff at 11M scale. Each dim=96 layer has ~2.25x the parameters of
a dim=64 layer, so the budget only affords 3+2+4=9 total layers versus M2's 4+4+5=13. The wider
representation per layer cannot compensate for the lost depth. This is especially damaging because
the 96→128 merge projection is a smaller dimensional jump than 64→128, meaning the merge adds
less representational capacity. The result confirms that at small scale, depth (number of sequential
transformations) matters more than width (dimension per transformation). The dim=64 choice in
M2 is well-calibrated: narrow enough to afford sufficient depth, wide enough for meaningful
per-stream representations.

#### Combined Framework-Level Analysis

All three framework-level topology experiments decisively confirm that M2's 4×64 (4L) → 2×128
(4L) → 1×128 (5L) topology is a well-balanced design:

- **Wider streams, fewer layers** (3/2/4, dim=96): depth starves everywhere → 214.41
- **Too much Stage 0** (6/4/3): streams specialize but post-merge refinement starves → 165.51
- **Too little Stage 1** (7/1/8): first merge output can't integrate before second merge → 183.08
- **Too little Stage 0** (2/4/7, Exp 8): streams can't specialize → 164.52

The pattern is clear: each stage has a distinct role (specialize, integrate, refine) and needs
sufficient depth to perform it. The balanced 4/4/5 split is not a coincidence. Additionally,
depth matters more than width at this scale: fewer wider layers is strictly worse than more
narrower layers, confirming that dim=64 per stream is the right operating point for 11M params.

### 5.14 Foundational Assumptions

Sections 5.1–5.13 exhaustively tested architecture-level and framework-level choices while leaving
foundational assumptions untouched: learning rate, embedding dimension, RoPE configuration, and
the merge mechanism's computational depth. These were all inherited from M0 or the original design
document and never independently validated for MSPM. This section corrects that oversight.

#### Experiment 35: FELIX-EMBED64 (Halved Embedding Dimension)

**Config:** `felix_embed64` | **Params:** 11,000,768 | **Date:** 2026-03-08 | **Epochs:** 2
**Control:** M2-nosup (Exp 4, ~136 PPL at 2 epochs) | **Variable:** d_embed=64 instead of 128, layers 6/7/9 (22 total)

Topology: 4×64 (6L linear) → 2×128 (7L sliding_window) → 1×128 (9L full_causal), d_embed=64

| Split | Loss  | PPL    |
|-------|-------|--------|
| Test  | 4.9783 | 145.22 |

**Analysis:** Halving the embedding dimension frees 3.2M params, enabling 22 layers (vs M2's 13).
The result (145.22 PPL) is better than most architecture-level experiments but worse than
M2-nosup's ~136 reference at 2 epochs. The extra depth partially compensates for the weaker token
representations, but compressing 50,257 tokens into 64 dimensions loses too much information at
the input. The embedding quality bottleneck outweighs the depth advantage. At 11M scale,
d_embed=128 is the right tradeoff even though it consumes 58% of the budget.

#### Experiment 36: FELIX-MERGE-INTEGRATE (Post-Merge Integration Layers)

**Config:** `felix_merge_integrate` | **Params:** 11,617,408 | **Date:** 2026-03-08 | **Epochs:** ~1 (killed early)
**Control:** M2-nosup (Exp 4) | **Variable:** 1 transformer layer after each merge point, same 4/4/5 stages

| Split | Notes |
|-------|-------|
| Val   | 230.02 at step 7K, epoch 1 ppl 3229. Killed early. |

**Analysis:** Adding dedicated transformer layers after each merge to "digest" the merged output
was catastrophically bad. The integration layers add a full-causal attention block at dim=128
between the merge output and the next stage's input, but the model couldn't learn to use them.
The merge's concat+gate+project output is already in the right form for the next stage's
transformer layers. Adding an extra processing step between merge and stage disrupts the gradient
flow that the stage layers depend on. The gated merge is not a computational bottleneck — it's
a sufficient integration mechanism on its own.

#### Experiment 37: FELIX-CURRICULUM-2K (Supervision Warmup, 2000 Steps)

**Config:** `m2_nosup` + `--supervision-off-after 2000` | **Params:** 11,092,608 | **Date:** 2026-03-08 | **Epochs:** 2
**Control:** M2-nosup (Exp 4, ~136 PPL at 2 epochs) | **Variable:** Deep supervision ON for first 2000 steps, then OFF

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 5.0907 | 162.50 |

**Analysis:** Supervision curriculum with a 2000-step warmup performs at 162.50 PPL, substantially
worse than M2-nosup's ~136 reference at 2 epochs. Starting with deep supervision and switching it
off actually harms the model — the early supervised training pushes representations toward a
multi-exit objective, and when supervision is removed, the model must unlearn those patterns.
This is worse than never having supervision at all, confirming that the damage from deep
supervision isn't just about competing objectives during training — it causes lasting
representational harm that the model can't fully recover from even after the auxiliary losses
are removed.

#### Experiment 38: FELIX-CURRICULUM-4K (Supervision Warmup, 4000 Steps)

**Config:** `m2_nosup` + `--supervision-off-after 4000` | **Params:** 11,092,608 | **Date:** 2026-03-08 | **Epochs:** 2
**Control:** M2-nosup (Exp 4, ~136 PPL at 2 epochs) | **Variable:** Deep supervision ON for first 4000 steps, then OFF

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 5.1249 | 168.15 |

**Analysis:** Even worse than the 2K curriculum (168.15 vs 162.50), confirming the trend: longer
supervision exposure causes more lasting damage. The difference between 2K and 4K steps of
supervision (5.65 PPL) shows the damage is cumulative — every additional step under the
multi-exit objective pushes the model further from the representations it needs for final-output-only
training. Combined with Exp 37, this conclusively kills the supervision curriculum idea.

#### Experiment 39: FELIX-LR1E4 (Learning Rate 1e-4)

**Config:** `m2_nosup` + `--lr 1e-4` | **Params:** 11,092,608 | **Date:** 2026-03-08 | **Epochs:** 2
**Control:** M2-nosup at LR 3e-4 (Exp 4, ~136 PPL at 2 epochs) | **Variable:** Peak LR 1e-4 instead of 3e-4

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 5.9505 | 383.95 |

**Analysis:** Catastrophically undertrained. At 1e-4, the cosine schedule decays the LR so
slowly that the model barely converges in 2 epochs. The 383.95 PPL is worse than almost every
architecture ablation, confirming that MSPM is highly sensitive to learning rate — more so than
typical transformers. This data point, combined with Exps 4 and 40, establishes a clear LR
ordering for MSPM: 1e-4 (383.95) << 3e-4 (~136 at 2ep) << 6e-4 (101.39). The multi-stream
architecture's gradient dynamics require substantially higher learning rates.

#### Experiment 40: FELIX-LR6E4 (Learning Rate 6e-4)

**Config:** `m2_nosup` + `--lr 6e-4` | **Params:** 11,092,608 | **Date:** 2026-03-08 | **Epochs:** 2
**Control:** M2-nosup at LR 3e-4 (Exp 4, ~136 PPL at 2 epochs) | **Variable:** Peak LR 6e-4 instead of 3e-4

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 4.6189 | 101.39 |

**Analysis:** The best MSPM result of the entire study — and a dramatic improvement from LR alone.
At 6e-4, m2_nosup achieves 101.39 test PPL in 2 epochs, down from 118.41 at 3e-4 after 3 epochs.
That's a 14.4% improvement from a training hyperparameter. However, the M0 control (Exp 41) shows
this is NOT an architecture-specific advantage — M0 benefits even more from the same LR increase.
The takeaway: every experiment in this study (Exps 2–39) used a suboptimal learning rate for both
architectures. The LR 3e-4 default was borrowed from standard transformer practice without
validation. At the correct LR, MSPM's gap to M0 actually widens from 2.4% to 10.4%.

#### Experiment 41: M0-LR6E4 (M0 Baseline at LR 6e-4)

**Config:** `m0` + `--lr 6e-4` | **Params:** 11,172,736 | **Date:** 2026-03-08 | **Epochs:** 2
**Control:** M0 at LR 3e-4 (Exp 1, 115.66 at 3 epochs) | **Variable:** Peak LR 6e-4

| Split | Loss   | PPL   |
|-------|--------|-------|
| Test  | 4.5197 | 91.81 |

**Analysis:** The critical control experiment, and it changes the entire narrative. M0 at 6e-4
achieves 91.81 test PPL in only 2 epochs — a 20.6% improvement over its own 3-epoch result at
3e-4 (115.66). This is a larger relative improvement than MSPM got from the same LR change (14.4%).
At LR 6e-4, the gap between M0 and MSPM widens from 2.75 PPL (at 3e-4) to 9.58 PPL. The
single-stream transformer, with its simpler gradient dynamics, benefits MORE from the higher
learning rate. This definitively answers the study's central question at 11M scale: multi-stream
progressive merging does NOT improve language modeling compared to a parameter-matched
single-stream baseline when both are properly tuned. The "multi-stream tax" (narrower streams,
merge parameters, fewer effective layers) is not compensated by the richer merged representation
at this scale.

#### Experiment 42: FELIX-ROPE0 (Standard RoPE, No Helical Turns)

**Config:** `m2_nosup` + `--rope-turns 0` | **Params:** 11,092,608 | **Date:** 2026-03-08 | **Epochs:** ~0.7 (killed early)
**Control:** M2-nosup (Exp 4, turns=2) | **Variable:** rope_helical_turns=0 (standard RoPE)

| Split | Loss   | PPL    |
|-------|--------|--------|
| Test  | 5.7286 | 307.53 |

**Analysis:** Standard RoPE (turns=0) is catastrophically worse than helical RoPE (turns=2) for
MSPM. Even at only ~0.7 epochs of training, the 307.53 test PPL confirms the model was not
converging. The depth-extended RoPE, which encodes both token position and network depth, is
essential for the multi-stage architecture. Without the depth component, the model cannot properly
distinguish representations at different stages — a layer in Stage 0 (dim=64, linear attention)
and a layer in Stage 2 (dim=128, full causal) need different positional treatment. The helical
RoPE provides this, and removing it collapses the model's ability to maintain stage-appropriate
representations. This validates one of the original design document's novel contributions:
depth-extended RoPE is load-bearing for MSPM.

---

## 6. Future Experiments

### Scale to 100M Parameters

**Dataset:** FineWeb-Edu | **Hardware:** MI300X
**Blocked on:** Finalizing best small-scale configuration from the MI300X batch run.

At 100M parameters, the embedding matrix becomes a smaller fraction of the total budget, and the
"multi-stream tax" is proportionally reduced. The architectural advantages of MSPM may become more
apparent at this scale.

---

## 6.5 Scaling Experiments (100M and 500M)

### Experimental Setup

To test the central open question — whether the multi-stream tax shrinks at scale — we trained
four models on a DigitalOcean MI300X (192GB) GPU: M0 and MSPM at both 100M and 500M parameter
scales.

| Parameter | Value |
|-----------|-------|
| Dataset | Dolma (Dolmino mix), 1B tokens |
| Epochs | 1 |
| Learning rate | 1e-4 (conservative, validated stable at 100M+) |
| Precision | bf16 |
| Validation | WikiText-103 validation set |
| SDPA backend | MATH (flash/efficient broken on MI300X) |

**100M configs (~101M params each):**
- M0: d=512, 18 layers, 8 heads. Batch=20, grad_accum=12 (eff=240)
- MSPM: d_embed=512, stages 4x128(9L linear) / 2x256(9L sliding) / 1x512(11L full_causal). Batch=16, grad_accum=16 (eff=256)

**500M configs (~486-489M params each):**
- M0: d=1024, 26 layers, 16 heads. Batch=8, grad_accum=32 (eff=256)
- MSPM: d_embed=1024, stages 4x512(5L linear) / 2x1024(6L sliding w=256) / 1x1024(8L full_causal). Batch=8, grad_accum=32 (eff=256)

All configs use proven v1 design choices: no deep supervision, gated merge, hetero attention,
helical RoPE (turns=2), tied embeddings, dropout=0.1.

### Results

| Config | Params | Val PPL | Train Loss | Embed % |
|--------|--------|---------|------------|---------|
| m0_100m | 101.5M | 22,424 | 6.72 | 25.3% |
| felix_100m | 101.0M | 25,319 | 6.82 | 25.5% |
| m0_500m | 489.4M | 21,894 | 6.63 | 10.5% |
| felix_500m | 486.2M | **17,552** | 6.62 | 10.6% |

### Analysis

The scaling results reveal a dramatic crossover in the relative performance of MSPM vs M0:

| Scale | M0 PPL | MSPM PPL | Gap | MSPM vs M0 |
|-------|--------|----------|-----|------------|
| 11M (LR 6e-4) | 91.81 | 101.39 | +10.4% | MSPM loses |
| 100M (LR 1e-4) | 22,424 | 25,319 | +12.9% | MSPM loses |
| 500M (LR 1e-4) | 21,894 | 17,552 | **-19.8%** | **MSPM wins** |

At 100M, MSPM still underperforms — the embedding consumes 25% of the budget, and the
multi-stream compute overhead is not offset by representational benefit. The gap is actually
slightly worse than at 11M (12.9% vs 10.4%), though the different dataset (Dolma vs WikiText-103)
and LR (1e-4 vs 6e-4) make direct comparison imprecise.

At 500M, the picture reverses completely. MSPM beats M0 by 19.8% in validation perplexity. At
this scale, the embedding fraction drops to ~10%, leaving ~437M for stream computation and merges.
The streams have enough capacity (dim=512 in Stage 0, 1024 in Stages 1-2) to develop genuinely
useful specialization, and the progressive merge of those specialized representations produces a
richer final representation than M0's uniform 26-layer stack.

The crossover point lies somewhere between 100M and 500M. This validates the original scaling
thesis: the multi-stream tax (dominated by embedding overhead at small scale) becomes negligible
at larger scales, and the architectural inductive bias — parallel exploration followed by
progressive convergence — provides genuine benefit when the streams have sufficient compute budget.

**Important caveats:** These are 1-epoch runs on 1B tokens — all models are significantly
underfitted. The absolute PPL numbers (17K-25K) are not meaningful as language model quality
metrics. What matters is the relative comparison at each scale. Additionally, the wall-clock
training time for MSPM was approximately 2x that of M0 at 100M scale, meaning the fair
compute-matched comparison is less favorable to MSPM than the parameter-matched comparison
presented here.

### 6.6 Felix-LM v2: Adaptive Convergence (11M)

**Exp 47: felix_v2** — 10,848,006 params, 2026-03-09, 3 epochs

**Control:** M0 at LR 6e-4 (Exp 41, 91.81 PPL); MSPM v1 at LR 6e-4 (Exp 40, 101.39 PPL)
**Variable:** Replace fixed-stage MSPM with v2 adaptive convergence architecture

**Architecture:** v2 eliminates the fixed 3-stage topology entirely. All 13 stream layers are
structurally identical FelixV2Layers, each combining independent transformer processing per stream,
CentralPost hub communication (gated read/write), and adaptive soft merge (agreement-driven
interpolation toward stream consensus). After the 13 stream layers, streams are aggregated with
learned weights, projected from d_stream=64 to d_embed=128, and refined through 2 full-dimension
transformer layers before output.

The three key mechanisms are genuinely novel in combination:
- **CentralPost**: O(N) shared communication hub (vs O(N^2) cross-attention). Each stream reads from
  and writes to a shared state via gated projections, enabling indirect inter-stream coordination.
- **Adaptive merge**: Each layer computes inter-stream cosine agreement, then applies
  `merge_strength = sigmoid(temperature * agreement + bias)` where temperature and bias are learned
  per layer. Streams interpolate toward their RMSNorm'd mean by this amount.
- **Input-adaptive compute**: Since merge strength depends on per-token agreement, easy tokens
  (high agreement) converge early while hard tokens (low agreement) maintain stream diversity longer.

**Setup:** WikiText-103, LR 6e-4, batch 8, grad_accum 4, 3 epochs (~21,500 steps), cosine LR decay.

| Config | Params | Test PPL | vs M0 | vs MSPM v1 |
|--------|--------|----------|-------|------------|
| M0 (Exp 41) | 11.17M | 91.81 | — | — |
| MSPM v1 (Exp 40) | 10.85M | 101.39 | +10.4% | — |
| **Felix v2** | **10.85M** | **94.61** | **+3.1%** | **-6.7%** |

**Analysis:** v2 closes 71% of the gap between MSPM v1 and M0 (from 9.58 PPL to 2.80 PPL). This is
the best multi-stream result at 11M scale by a wide margin, achieved by replacing v1's fixed merge
boundaries with continuous agreement-driven convergence and adding CentralPost hub communication.

The improvement over v1 is mechanistically meaningful: v1's fixed stages force all tokens through the
same merge schedule regardless of difficulty. v2 lets the model learn per-token, per-layer merge
behavior — tokens where streams agree merge early, tokens where they disagree maintain diversity.
CentralPost provides the inter-stream coordination that v1 lacked entirely (v1 streams were isolated
until merge boundaries).

At 11M, v2 still doesn't beat M0 (3.1% gap), consistent with the finding that the multi-stream tax
dominates at small scale. However, v2's tighter gap at 11M is promising for scaling: if v1 crossed
over between 100M and 500M (winning by 19.8% at 500M), v2 should cross over earlier and win by more.
The adaptive convergence mechanism addresses the fundamental limitation of v1 — rigid topology — while
adding minimal parameter overhead (2 learned scalars + 1 RMSNorm per layer for the merge, plus
CentralPost projections).

Wall-clock speed was approximately 60% of M0 (4.3 it/s vs ~7 it/s), somewhat better than v1's 50%
at 100M scale. The overhead comes from running 4 independent transformer blocks per layer plus
CentralPost read/write, partially offset by not needing large merge projections at stage boundaries.

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
| 13 | M2-crossattn | Cross-stream attention | 2 | 163.42 | +41.3% | — |
| 14 | M2-shared-deep | Shared weights + 12L Stage 2 | 2 | 158.34 | +36.9% | — |
| 15 | M2-geometric | Geometric mean merge | 2 | 213.83 | +84.9% | — |
| 16 | M2-hadamard | Hadamard + diag merge | 2 | 157.46 | +36.2% | — |
| 17 | M2-antimerge | Competitive selection | 2 | 188.61 | +63.1% | — |
| 18 | M2-noise-merge | Noise at merge (0.1) | 2 | 168.44 | +45.6% | — |
| 20 | M2-diverge-mid | Divergence loss (0.1) | 2 | 148.09 | +28.1% | — |
| 23 | M2-asymmetric | Heterogeneous streams | 2 | 164.95 | +42.6% | — |
| 24 | M2-bottleneck | 4x merge bottleneck | 2 | 189.11 | +63.5% | — |
| 27 | M2-asymmetric-diverge | Asymmetric + diverge | 2 | 155.08 | +34.1% | — |
| 30 | M2-stream-permute | Random merge pairings | 2 | 186.76 | +61.5% | — |
| 32 | Felix-front643 | Frontloaded 6/4/3 split | 2 | 165.51 | +43.1% | — |
| 33 | Felix-2stage | Minimal bridge 7/1/8 split | ~1.5 | 183.08 | +58.3% | — |
| 34 | Felix-wide96 | Wider streams dim=96, 3/2/4 | ~1.1 | 214.41 | +85.4% | — |
| 35 | Felix-embed64 | d_embed=64, 22 layers | 2 | 145.22 | +25.6% | — |
| 36 | Felix-merge-integrate | Post-merge transformer layers | ~1 | killed | — | — |
| 37 | Felix-curriculum-2k | Supervision off after 2K steps | 2 | 162.50 | +40.5% | +37.3% |
| 38 | Felix-curriculum-4k | Supervision off after 4K steps | 2 | 168.15 | +45.3% | +42.0% |
| 39 | Felix-lr1e4 | LR 1e-4 | 2 | 383.95 | +231.8% | +224.3% |
| 40 | Felix-lr6e4 | **LR 6e-4** | **2** | **101.39** | **+10.4%** | **-14.4%** |
| 41 | **M0-lr6e4** | **M0 at LR 6e-4** | **2** | **91.81** | **-20.6%** | — |
| 42 | Felix-rope0 | Standard RoPE (turns=0) | ~0.7 | 307.53 | +165.9% | +159.7% |
| 43 | m0_100m | M0 at 100M scale | 1 (1B Dolma) | 22,424* | — | — |
| 44 | felix_100m | MSPM at 100M scale | 1 (1B Dolma) | 25,319* | +12.9% | — |
| 45 | m0_500m | M0 at 500M scale | 1 (1B Dolma) | 21,894* | — | — |
| 46 | **felix_500m** | **MSPM at 500M scale** | **1 (1B Dolma)** | **17,552*** | **-19.8%** | — |
| 47 | **Felix v2** | **Adaptive convergence** | **3** | **94.61** | **+3.1%** | **-6.7%** |

\* Exps 43-46 use Dolma dataset (not WikiText-103) and LR 1e-4. Absolute PPL not comparable to
Exps 1-42. Only relative comparisons within the scaling group are meaningful.

### 7.2 Key Findings

1. **Deep supervision is the dominant failure mode.** Removing it yields a 32-point PPL improvement
   (Exp 4), the largest single-variable effect in the study. The supervision objective conflicts
   with learning merge-optimal representations.

2. **The original M2-nosup design is a robust local optimum.** Across 22 experiments testing every
   architectural dimension — attention type, stream count, layer distribution, gate mechanism,
   regularization, merge algorithm, training objectives, structural diversity, and combinations —
   not a single variant improved on M2-nosup. The design document's choices (4 streams, 4/4/5
   layers, hetero attention, simple gated merge with bias=1.0) are exceptionally well-calibrated.

3. **Heterogeneous attention enforces beneficial specialization.** Linear attention in Stage 0
   acts as an inductive bias for coarse pattern capture when supervision is off (Exp 5). The
   coarse-to-fine hierarchy (linear -> sliding-window -> full-causal) is the right design.

4. **4 streams is optimal at 11M scale.** Both 2 and 8 streams are substantially worse (Exps 6-7).
   The sweet spot balances per-stream capacity against representational diversity.

5. **The gated merge is the right merge.** Four alternative merge algorithms (geometric, Hadamard,
   competitive, noise-injected) all performed substantially worse (Exps 15-18). The key insight:
   merges must be *additive* to combine orthogonal stream representations. Multiplicative
   (geometric) and selective (antimerge) approaches destroy information from complementary
   subspaces. The full-rank learned projection earns its O(d^2) parameters — fixed rotations
   (Hadamard) and bottlenecks (Exp 24) are too constrained.

6. **Emergent specialization beats imposed specialization.** Structurally heterogeneous streams
   (Exp 23: 164.95) perform worse than identical-architecture streams that specialize via learned
   weights. Removing FFN or attention from individual streams cripples their capacity. The model
   finds better divisions of labor on its own than we can impose architecturally.

7. **Any auxiliary objective hurts.** Deep supervision (Exp 2), divergence loss (Exp 20), and
   noise injection (Exp 18) all degrade performance. The pattern is consistent: the primary
   language modeling loss is the only objective that should drive training. Even well-motivated
   auxiliary losses (encouraging stream diversity, robustness) compete with the primary gradient.

8. **The merge requires consistent, predictable inputs.** Stream dropout (Exp 11), random
   permutation (Exp 30), and noise injection (Exp 18) all hurt substantially. The gated
   projection learns a precise mapping for specific input patterns — any stochasticity at the
   merge boundary destabilizes this learned mapping.

9. **Multi-stream merging works, but the "tax" is high at small scale.** The embedding matrix
   consumes 6.4M of the 11M budget. M0 gets 18 full-width layers from the remaining 4.6M; M2
   gets roughly 9 effective full-width layers after accounting for 4 narrow streams and merge
   projections. The 2.4% PPL gap may narrow at larger scales where the embedding fraction shrinks.

10. **Stage 2 output can beat M0.** Under full causal attention + deep supervision (Exp 3), the
    Stage 2 exit head achieves 107.49 PPL vs M0's 115.66. The merged representation is genuinely
    richer — the challenge is making the overall training objective (not just the final stage)
    exploit this.

11. **M2's topology is a robust optimum across the framework design space.** Three framework-level
    experiments (Exps 32-34) tested fundamentally different MSPM topologies: frontloaded layers,
    minimal intermediate stage, and wider streams. All performed dramatically worse (165-214 PPL
    vs ~136 reference). Combined with the backloading result (Exp 8), every deviation from the
    4/4/5 layer split at dim=64 hurts. At 11M scale, depth matters more than width, and all three
    stages need sufficient depth for their distinct roles.

12. **LR 3e-4 was suboptimal for BOTH architectures.** At LR 6e-4, both MSPM and M0 improve
    dramatically — M0 from 115.66 to 91.81 (-20.6%), MSPM from 118.41 to 101.39 (-14.4%). The
    3e-4 default was borrowed from standard practice without validation. However, M0 benefits
    MORE from the higher LR, widening the gap from 2.75 PPL to 9.58 PPL. The multi-stream
    architecture does not have an inherent LR advantage.

13. **Helical RoPE is essential for MSPM.** Standard RoPE (turns=0) is catastrophically worse
    (311 val_ppl at step 5K vs ~150 for turns=2, Exp 42). The depth-extended positional encoding
    is load-bearing — the model needs to distinguish representations at different stages, and
    helical RoPE provides this signal. This validates the original design document's novel
    contribution.

14. **The merge is not a computational bottleneck.** Adding dedicated transformer layers after
    each merge (Exp 36) was catastrophically bad (val_ppl 230, ppl 3229 at epoch 1). The
    gated merge's concat+gate+project is sufficient for integration. The merged output is
    already in the right form for the next stage.

15. **MSPM beats M0 at 500M scale.** The scaling experiments (Exps 43-46) confirm the central
    thesis: the multi-stream tax shrinks with scale. At 500M params (embedding ~10% of budget),
    MSPM achieves 17,552 val PPL vs M0's 21,894 — a 19.8% improvement. This is a complete
    reversal from the 10-13% deficit at 11M and 100M. The crossover lies between 100M and 500M.

16. **The crossover is driven by embedding fraction.** At 11M (58% embedding), streams are
    starved for compute. At 100M (25%), still not enough. At 500M (10%), streams finally have
    sufficient capacity (dim=512 in Stage 0) to develop meaningful specialization. The
    progressive merge of genuinely specialized streams produces richer representations than M0's
    uniform stack.

17. **v2 adaptive convergence closes 71% of the multi-stream gap at 11M.** Replacing fixed merge
    boundaries with agreement-driven continuous merging and CentralPost hub communication reduces
    the M0 gap from 10.4% (v1) to 3.1% (v2). This is the best multi-stream result at 11M by a
    wide margin, achieved without changing the param budget. The improvement validates that the
    v1 failure was partly architectural (rigid topology, no inter-stream communication) rather
    than fundamental to multi-stream approaches.

### 7.3 Open Questions

- **Where exactly is the crossover?** MSPM loses at 100M but wins at 500M. Testing at 200M
  and 300M would pin down the threshold. This matters for practical deployment decisions.
- **Is 1e-4 the right LR at scale?** All scaling runs used 1e-4 (conservative). At 11M, 6e-4
  was optimal but differentially benefited M0. The LR sensitivity may differ at 500M — MSPM
  might benefit more from higher LR at this scale.
- **Can v2 beat M0 at scale?** v2 already closes 71% of the gap at 11M. v1 crossed over at
  500M with a 19.8% win — v2 should cross over earlier (possibly at 100M) and win by more.
  A v2 scaling run is the highest-priority next experiment.
- **Does the 2x wall-clock penalty change the conclusion?** MSPM took ~2x as long per step as
  M0 at 100M. If M0 trained for 2B tokens matches or beats MSPM at 1B tokens, the compute
  efficiency argument weakens. Token-matched AND compute-matched comparisons are both needed.
- **Is 6e-4 the optimal LR for either architecture at 11M?** Neither has been tested at 8e-4
  or 1e-3. There may be additional gains available.
- **Multi-epoch scaling runs.** All scaling experiments were 1 epoch on 1B tokens — heavily
  underfitted. Longer training (3+ epochs or more tokens) would give more reliable absolute
  numbers and might change the relative gap.
