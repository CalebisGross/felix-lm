# Felix-LM Research Directions

Ideas and hypotheses for future experimentation, documented so nothing gets lost.

---

## 1. Adaptive Stream Merging (from Fresnel Fibonacci insight)

**Current state:** Fixed 4→2→1 merge schedule hardcoded in config.
**Problem:** We don't know if this is optimal. Maybe some stream pairs should merge early (redundant) while others should stay separate longer (specializing).

**Idea:** Learn *when* to merge based on cross-stream agreement. High agreement = merge early (streams are redundant, save compute). Low agreement = keep separate (streams are capturing different things).

**Implementation sketch:**
- After each layer, compute pairwise agreement between streams
- If agreement > threshold, merge that pair early
- If agreement stays low, delay merge to a later layer
- The threshold could be learned or annealed during training

**Inspiration:** Fresnel Experiment 013 (Fibonacci Gaussian Splatting) — structured sparse placement beats uniform dense. Same principle: don't merge on a rigid schedule, merge where it helps.

**Priority:** After ablations complete and if fixed architecture doesn't beat M0 at scale.

---

## 2. Untested Structural Parameters

**Critical concern:** The current 4→2→1 structure with dim=64→128→128 and layer distribution 4/4/5 was chosen by reasonable heuristics, NOT by search or optimization. These parameters could be far from optimal.

**Parameters that need systematic exploration:**

| Parameter | Current Value | Alternatives to Test |
|-----------|--------------|---------------------|
| Stream count (S_0) | 4 | 2, 8, 16 |
| Stage count (K) | 3 | 2, 4, 5 |
| Dim per stream (d_0) | 64 | 32, 128 |
| Dim growth pattern | 64→128→128 | 64→128→256, 64→64→128 |
| Layer distribution | 4/4/5 | 2/4/7, 3/3/7, 6/4/3 |
| Merge gate bias init | 1.0 | 0.0, 0.5, 2.0 |
| FFN multiplier | 4 | 2, 3, 8 |

**Key question:** Is the underperformance vs M0 caused by the *architecture* or by a *bad default configuration*? We won't know until we search this space.

**Approach:** Start with layer distribution (cheapest to test — same params, just redistributed). Then stream count. Dim growth last (changes param count).

---

## 3. Cross-Stream Agreement as Compute Allocator

**Current state:** Agreement is logged as a diagnostic but not used during inference.
**Thesis reference:** Section 4.3 (early exit), Section 6.3 (diagnostics).

**Idea:** Use agreement signal to allocate variable compute per token:
- High agreement at Stage 0 merge → all streams agree → easy token → exit early
- Low agreement → streams disagree → hard token → continue to full depth

**Why this matters:** This is the efficiency story. Even if Felix-LM matches (not beats) M0 on perplexity, if it can skip 30-50% of compute on easy tokens via early exit, that's a win for inference cost.

**Connection to Mixture of Depths (Raposo et al., 2024):** They route tokens to skip layers. Our version is architecturally motivated — agreement is a *natural* difficulty signal, not a learned router.

**Priority:** High — this could be the main contribution even if raw PPL doesn't beat baseline.

---

## 4. Progressive Resolution Training (from Fresnel HFTS)

**Fresnel approach:** Train at 64x64, validate at 256x256 (16x speedup).

**Felix-LM equivalent:** Curriculum over sequence length AND stream count:
- Phase 1: seq_len=128, 2 streams (fast iteration, find good hyperparams)
- Phase 2: seq_len=256, 4 streams (introduce full architecture)
- Phase 3: seq_len=512+, 4 streams (full training)

**Why:** Faster experimentation cycle. Also, progressive stream introduction might help the model learn to use streams more effectively — start simple, add complexity.

**Priority:** Medium — useful for speeding up the hyperparameter search in #2.

---

## 5. Test-Time Stream Optimization (from Fresnel Exp 018)

**Fresnel result:** SSIM 0.93 without training by optimizing directly on the input.

**Felix-LM equivalent:** At inference time, for a difficult prompt:
- Run forward pass, check agreement
- If low agreement (uncertain), run additional forward passes with different stream initializations or dropout masks
- Ensemble the predictions

**This is speculative.** May not be practical for autoregressive generation. But for tasks like classification or scoring (where you process a full sequence once), it could work.

**Priority:** Low — explore after core architecture is validated.

---

## 6. Complex-Valued Stream Representations (long-term)

**Idea:** Replace real-valued hidden states with complex tensors:
- Magnitude = content strength (what)
- Phase = relational information (how it relates to other tokens/streams)
- Merge via interference (constructive = agreement, destructive = disagreement)

**Why:** Makes the "helical funnel" literally true — a helix is a complex exponential e^{iθ}. Phase synchronization between streams becomes a principled merge signal grounded in wave physics.

**Research to read first:**
- Complex Recurrent Units (CRU) — complex-valued RNNs for long-range dependencies
- Diff Transformer (Ye et al., 2024) — differential attention as noise cancellation
- Titans (Google, 2025) — short-term vs long-term memory modules

**Priority:** Felix-LM v2. Do NOT attempt this before the current real-valued architecture is fully characterized. This is a rewrite, not a tweak.

---

## 7. Scaling Strategy

**Current:** 11M params on WikiText-103, local RX 7800 XT.
**Next:** 50-100M params on FineWeb-Edu, MI300X droplet ($70 credit).

**What to test at scale:**
- Does the M2 vs M0 gap close, hold, or widen?
- Does stream specialization become more pronounced with wider dims?
- Does early exit become more effective (more variance in token difficulty)?

**Critical:** Run M0 baseline at every scale point. No cherry-picking. If M0 wins at 100M too, the architecture needs rethinking, not more scale.

---

## Experiment Priority Order

1. Finish current ablations (M2-nosup, then M2-best = fullcausal + nosup)
2. Test layer distribution variants (cheap, same params)
3. Test 2-stream config (fewer streams, wider dims)
4. Scale to 50-100M with best small-scale config
5. Implement early exit with agreement signal
6. If results are positive: adaptive merging, progressive training
7. Long-term: complex-valued representations
