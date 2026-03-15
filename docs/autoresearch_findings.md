# Felix-LM v2 Autoresearch: Comprehensive Findings

**82 experiments, 11M and 100M scale, March 2026**

## The Question

Can the Felix multi-agent principles — parallel exploration, progressive convergence, CentralPost communication, and adaptive agreement-driven merging — improve a causal language model over a plain transformer at the same parameter budget?

## The Answer (Short Version)

No. At both 11M and 100M parameters, a plain transformer beats every Felix v2 variant we tested when both architectures are given their optimal hyperparameters. The multi-stream mechanism with CentralPost and adaptive merge is a consistent tax of +2 to +6 PPL.

The one apparent exception (Felix winning at 100M, exp 74) was a learning rate artifact: Felix happened to train well at LR 3e-3, but the baseline hadn't been tested at LR 2e-3 yet, where it was significantly better.

## What We Built

### v2 Architecture

Felix-LM v2 ("Adaptive Convergence") processes input through:

1. **Stream initialization** — shared embedding projected into N independent streams at dimension d_stream
2. **Felix layers** — each stream processed by independent transformer blocks, then:
   - **CentralPost** — shared hub where streams read/write via gated projections (O(N) communication)
   - **Adaptive merge** — streams pulled toward their mean proportional to inter-stream agreement (cosine similarity)
3. **Stream aggregation** — learned-weighted sum of streams
4. **Output projection** — d_stream to d_embed
5. **Refinement layers** — standard transformer blocks at d_embed
6. **Output head** — tied embeddings with logit scaling

### The Baseline

When num_layers=0, v2 degenerates into a plain transformer: embed -> project -> 20 refine layers -> output. This turned out to be the best configuration every time.

## Experiment Phases

### Phase 1: Architecture Search (exp 1-19, LR 6e-4)

Started with the default v2 config (13 stream layers + 2 refine layers, 4 streams at d_stream=64). Systematically shifted compute from stream to refine:

| Exp | Config | Val PPL |
|-----|--------|---------|
| baseline | 13 stream + 2 refine | 374.83 |
| exp 3 | 9 stream + 6 refine | 329.34 |
| exp 5 | 4 stream + 11 refine | 308.81 |
| exp 7 | 1 stream + 14 refine | 302.99 |
| exp 8 | 0 stream + 15 refine | 297.92 |
| exp 17 | 0 stream + 20 refine | 288.11 |
| exp 19 | 0 stream + 20 refine, wd=0 | 286.43 |

**Finding: Every stream layer removed improved PPL.** The relationship was monotonic — zero stream layers was always best. Architecture tweaks at fixed depth (d_stream width, FFN ratio, head count, RoPE ablation) were all neutral within noise (exp 9-14).

### Phase 2: Hyperparameter Discovery (exp 20-58)

The first 19 experiments used LR 6e-4. This turned out to be catastrophically suboptimal.

**Learning rate** was the biggest lever by far:

| LR | Val PPL | Improvement |
|----|---------|-------------|
| 6e-4 | 286.43 | (baseline) |
| 1e-3 | 207.31 | -27.6% |
| 2e-3 | 148.70 | -28.3% |
| 4e-3 | 115.40 | -22.4% |
| 8e-3 | 97.19 | -15.8% |

Subsequent sweeps found interacting optima:

| Parameter | Optimal | Sweep Range | Effect Size |
|-----------|---------|-------------|-------------|
| LR | 2e-2 | 6e-4 to 3e-2 | 286 -> 59 PPL |
| Weight decay | 0.1 | 0.0 to 0.2 | 97 -> 82 PPL |
| beta2 | 0.99 | 0.95 to 0.999 | 74 -> 66 PPL |
| Warmup | 500 | 250 to 750 | 59 -> 58 PPL |
| Grad accum | 8 (eff batch 64) | 4 to 8 | 58 -> 46 PPL |
| torch.compile | yes | — | 45 -> 44.8 PPL |
| d_stream | 128 (= d_embed) | 64, 128 | 46 -> 45 PPL |

**Critical insight: weight decay and beta2 interact with LR.** At wd=0, optimal LR was 8e-3 and PPL was 97. Adding wd=0.1 stabilized higher LR (1e-2 -> 74 PPL). Adding beta2=0.99 stabilized even higher LR (2e-2 -> 59 PPL). Each regularization change unlocked a higher optimal LR.

Parameters that didn't matter: dropout (redundant with wd), min_lr_ratio (doesn't matter at 2500 steps), grad_clip variations (1.0 was optimal), beta1 variations, label smoothing (not comparable via PPL), seq_len changes (not comparable).

**Final 11M optimized baseline: 44.81 PPL** (0 stream + 20 refine, d=128, LR 2e-2, wd=0.1, beta2=0.99, warmup=500, grad_accum=8, torch.compile)

### Phase 3: Making Felix Work (exp 59-72)

With hyperparameters locked in, we tried every approach to make streams help:

**Independent streams (replacing refine layers):**

| Config | d_stream | Streams | PPL | Gap |
|--------|----------|---------|-----|-----|
| 0+20 baseline | 128 | 0 | 44.81 | — |
| 2+18 (d64) | 64 | 4 | 49.03 | +4.14 |
| 4+16 (d64) | 64 | 4 | 50.86 | +5.97 |
| 2+18 (d128) | 128 | 2 | 50.94 | +6.13 |

**Shared-weight streams:**

| Config | PPL | Gap |
|--------|-----|-----|
| Shared 2+18 d128 | 59.57 | +14.76 |

Shared weights destroyed diversity — all streams identical before CentralPost, making the 4x forward passes pure waste.

**Additive streams (ON TOP of full 20 refine, not replacing):**

| Config | PPL | Gap |
|--------|-----|-----|
| 1 stream + 20 refine (d64, 2s) | 47.56 | +2.75 |
| 1 stream + 20 refine (d64, 2s, merge_bias=1.0) | 47.38 | +2.57 |
| 2 stream + 20 refine (d64, 2s) | 48.44 | +3.63 |
| 1 stream + 20 refine (d128, 2s) | 49.13 | +4.32 |

**Lightweight exchange (our LightFelixLayer, replacing CentralPost+merge):**

| Config | PPL | Gap |
|--------|-----|-----|
| Light 1+20 (d64, 2s) | 47.09 | +2.28 |

**Training duration comparison (best Felix vs baseline):**

| Steps | Baseline | Felix (add1_open) | Gap |
|-------|----------|-------------------|-----|
| 2,500 | 44.81 | 47.38 | +2.57 |
| 5,000 | 39.47 | 40.23 | +0.76 |
| 10,000 | 37.20 | 38.85 | +1.65 |

Gap narrowed at 5K steps but widened again at 10K. Not converging.

**Best Felix at 11M: 47.09 PPL** (light exchange, 1 stream + 20 refine, gap +2.28)

### Phase 4: 100M Scale (exp 73-82)

Moved to 100M params (d_embed=512, ~110M total) running locally on AMD RX 7800 XT.

**Initial results (LR 3e-3):**

| Config | PPL | vs Baseline |
|--------|-----|-------------|
| Baseline 0+20 | 72.88 | — |
| Felix 2+19 (d256, 2s) | 71.94 | -0.94 (!) |
| Felix additive 1+20 (d128, 2s) | 73.00 | +0.12 |

This looked like a breakthrough — Felix beating baseline at 100M! But then:

**The LR artifact (the most important finding):**

| Config | LR 3e-3 | LR 2e-3 |
|--------|---------|---------|
| Baseline 0+20 | 72.88 | **61.37** |
| Felix 4+18 | 261 (unstable) | **67.39** |
| Felix 2+19 | 71.94 | (not tested) |

The baseline improved by 11.5 PPL just from changing LR from 3e-3 to 2e-3. Felix's apparent win was because we tested Felix at its optimal LR first but hadn't optimized the baseline yet.

**Fair 100M comparison (each at optimal LR):**
- **Baseline: 61.37 PPL** (LR 2e-3)
- **Best Felix: 67.39 PPL** (4+18, LR 2e-3, gap +6.02)

Felix still loses by 6 PPL at 100M.

## Why Felix Loses: Root Cause Analysis

### 1. The Parameter Efficiency Problem

A stream layer contains:
- N independent transformer blocks at d_stream (the useful work)
- CentralPost: N read projections + N read gates + N write projections + N write gates + norm
- Adaptive merge: agreement computation + temperature/bias params + norm + interpolation

For 2 streams at d_stream=256 (100M config):
- Useful compute: 2 x transformer_block(d=256) = ~1.6M params
- CentralPost overhead: ~67K params
- One refine layer at d=512: ~4.2M params

Two stream layers "cost" about 3.2M params but only provide the compute equivalent of ~0.75 refine layers (since each stream block is at half width). The CentralPost and merge add overhead without proportionate benefit.

### 2. CentralPost Doesn't Help

CentralPost was designed as an O(N) communication mechanism (vs O(N^2) cross-attention). But empirically:
- Shared-weight streams with CentralPost (exp 62): 59.57 PPL — terrible
- Light exchange (just gated mean, no CentralPost): 47.09 PPL — best Felix result
- Full CentralPost (exp 64): 47.56 PPL — slightly worse than light exchange

CentralPost's gated read/write through a separate d_post dimension adds parameters and computation without meaningful information exchange. The simpler "look at the mean" approach works slightly better.

### 3. Adaptive Merge Doesn't Help

The agreement-driven merge computes cosine similarity between streams, maps it through a learned sigmoid, and interpolates toward the mean. In practice:
- merge_bias=-1.0 (start mostly independent): slightly worse
- merge_bias=1.0 (start mostly merged): slightly better
- The mechanism adds compute for the agreement calculation but the learned temperature and bias end up producing a roughly constant gate value

The adaptive part of "adaptive merge" doesn't appear to learn meaningfully different behavior from a fixed gate.

### 4. The Fundamental Tension

Felix-LM's thesis is that **parallel exploration followed by progressive convergence** provides a useful inductive bias. But at both scales tested:

- **Exploration is expensive.** Running N independent transformer blocks costs N times as many parameters per layer.
- **Convergence destroys information.** Merging N streams into one (or interpolating toward the mean) loses the diversity that exploration created.
- **Communication is weak.** CentralPost provides only indirect, low-bandwidth communication between streams. Each stream learns in near-isolation, then must agree with others at merge time.

The plain transformer doesn't have this exploration/convergence overhead. Every parameter goes directly into a single, deep representation. At the scales we tested, depth beats breadth.

### 5. The Embedding Tax (11M Only)

At 11M params, the GPT-2 vocabulary (50,257 tokens) at d_embed=128 consumes 6.4M params (58% of budget). Only 5.3M params remain for compute. Multi-stream overhead on top of this leaves almost nothing for actual processing.

At 100M (d=512), embedding is 25.7M (23% of budget), leaving 84M for compute. This is why we expected streams to help at 100M — but the parameter efficiency problem (cause #1) persists regardless of scale.

## What We Learned About Optimization

1. **LR is always the biggest lever.** At both 11M and 100M, finding the right LR produced larger improvements than any architectural change. The first 19 architecture experiments were essentially invalid because they used the wrong LR.

2. **Hyperparameters interact.** Weight decay enables higher LR. Higher beta2 enables higher LR on top of that. You can't optimize one at a time — each change shifts the optimal value of the others.

3. **Always compare at optimal LR for each config.** The exp 74 false positive (Felix "winning" at 100M) happened because we compared Felix at its optimal LR against baseline at a suboptimal LR. Different architectures have different optimal LRs.

4. **Larger batch helps.** Going from effective batch 32 to 64 improved PPL by ~1.5 points, even after accounting for seeing more data. Larger batches give better gradient estimates, especially at high LR.

5. **torch.compile is free performance.** 2.2x throughput with slightly better results, no downsides.

## What Remains Interesting

Despite the negative result on PPL, some Felix ideas have merit:

1. **Cross-stream agreement as confidence.** The cosine similarity between independent streams processing the same input is a genuine uncertainty signal with no analog in single-stream models. This could be valuable for hallucination detection even if it doesn't improve training loss.

2. **Depth-extended RoPE.** Helical RoPE (encoding layer index in addition to position) consistently helps by ~0.9 PPL at 11M. This is a novel contribution independent of the multi-stream architecture.

3. **The scaling question isn't fully answered.** We tested 11M and 100M. The v1 results showed crossover at 500M, but that was before proper LR tuning. A properly controlled experiment at 500M would be definitive.

4. **The design paper's core claim may need revision.** The paper claims the helical funnel trajectory provides a useful inductive bias. The evidence suggests the "funnel" part (stream merging) actively hurts, while the "helical" part (depth-extended RoPE) modestly helps. These are separable contributions.

## Recommendations for v3

If pursuing a v3 architecture, the evidence suggests:

1. **Don't duplicate transformer blocks across streams.** This is the primary source of parameter waste. Instead, consider approaches where streams share compute but diverge in some lightweight way (different attention masks, different positional encodings, different gating).

2. **Make communication the architecture, not an add-on.** CentralPost bolted onto independent streams doesn't work. Instead, design a single transformer backbone with built-in multi-view computation — closer to multi-head attention (which already processes N parallel views) than to N separate networks.

3. **The agreement signal needs a gradient pathway.** Currently, agreement drives the merge strength but doesn't directly affect the training loss. If agreement-based confidence is valuable, it needs to be part of the objective, not just a monitoring metric.

4. **Consider the MoE analogy seriously.** Mixture-of-Experts uses N expert networks but routes different tokens to different experts — so total compute stays at ~1x, not Nx. Felix's "all tokens through all streams" approach is fundamentally more expensive. A routing-based variant would be more parameter-efficient.

5. **Test at 500M+ with proper controls.** If pursuing the current architecture, the only path to a positive result is scale. But the LR artifact at 100M is a warning: always include a properly optimized baseline.

## Raw Numbers

Total experiments: 82
- Architecture variants tested: ~20 distinct configs
- Hyperparameters swept: LR, wd, beta1, beta2, warmup, grad_clip, grad_accum, dropout, ffn_mult, num_heads, d_stream, seq_len, label_smoothing, min_lr_ratio, torch.compile
- Scales tested: 11M (local, ~10-30 min/run), 100M (local, ~30 min/run)
- Best 11M PPL: 44.81 (plain transformer) vs 47.09 (best Felix), gap +2.28
- Best 100M PPL: 61.37 (plain transformer) vs 67.39 (best Felix), gap +6.02
- Total training runs: ~150+ hours of GPU time
- Hardware: AMD RX 7800 XT (16GB VRAM), ROCm
