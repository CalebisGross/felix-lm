# Felix-LM Self-Improvement Experiments

Agreement-based self-improvement for Felix-LM v2. The hypothesis: inter-stream agreement
alone (without answer verification) can serve as a training signal to improve a language model.

## Protocol

1. Load supervised checkpoint (baseline)
2. Generate completions from random prompts
3. Filter by agreement threshold + repetition check
4. Train on high-agreement sequences
5. Mix in real data batches each cycle (prevent forgetting)
6. Evaluate val PPL every N cycles, early-stop on stall or degradation

Safety: revert to best checkpoint if val PPL degrades >5% from baseline.

---

## Experiment S1: Threshold 0.57, LR 1e-5

**Date:** 2026-03-10 | **Checkpoint:** felix_v2/best.pt (step 21500, 94.50 val PPL)

**Hypothesis:** Agreement threshold at ~p50 of the distribution (0.57, ~50% keep rate) will
select high-quality self-generated sequences that improve the model.

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | felix_v2/best.pt (94.50 val PPL) |
| Agreement threshold | 0.57 |
| Self-improvement LR | 1e-5 (10x lower than supervised) |
| Sequences per cycle | 200 |
| Real data batches/cycle | 5 |
| Temperature | 0.8, top-k 50 |
| Prompt length | 64 tokens |
| Generated length | 64 tokens |

### Results

| Metric | Value |
|--------|-------|
| Baseline val PPL | 94.50 |
| Final val PPL | 94.50 (reverted) |
| PPL at cycle 5 | 102.03 (+8.0%) |
| Delta | +0.00 (safety revert triggered) |
| Cycles completed | 5 / 40 |
| Total kept | 771 |
| Time | 1083s (~18 min) |

**Per-cycle keep rate:**

| Cycle | Kept/Total | Keep Rate | Self Loss | Real Loss |
|-------|------------|-----------|-----------|-----------|
| 1 | 135/200 | 67.5% | 3.7842 | 4.8208 |
| 2 | 151/200 | 75.5% | 3.6579 | 4.6779 |
| 3 | 161/200 | 80.5% | 3.5182 | 4.6636 |
| 4 | 164/200 | 82.0% | 3.5281 | 4.7478 |
| 5 | 160/200 | 80.0% | 3.3575 | 4.6207 |

**Agreement distribution:**

| Cycle | Mean | Std | Min | Max | p25 | p50 | p75 |
|-------|------|-----|-----|-----|-----|-----|-----|
| 1 | 0.574 | 0.011 | 0.545 | 0.603 | 0.568 | 0.574 | 0.582 |
| 5 | 0.586 | 0.013 | 0.539 | 0.619 | 0.579 | 0.586 | 0.594 |

### Analysis

The run failed immediately. Val PPL jumped from 94.50 to 102.03 (+8.0%) after just 5 cycles,
triggering the >5% safety revert. Three problems are visible in the data:

**1. Keep rate was far too high.** The threshold of 0.57 was supposed to be selective (~50%),
but kept 68-82% of sequences. The agreement distribution shifted upward during training (mean
0.574 to 0.586 in 5 cycles), meaning the model quickly learned to produce higher-agreement
outputs without actually improving. The threshold needs to track the distribution, not be fixed.

**2. Self-loss dropped while val PPL rose.** Self-loss decreased monotonically (3.78 to 3.36)
while validation PPL degraded. This is textbook overfitting to self-generated data. The model
is memorizing patterns in its own outputs rather than learning generalizable improvements. The
generated sequences (64 tokens from 64-token prompts) may be too short and too easy for
meaningful self-training.

**3. Agreement is not a sufficient quality signal at this scale.** At 11M params, the model's
agreement distribution is extremely narrow (std=0.011). There isn't enough variance in agreement
to meaningfully separate good from bad generations. The difference between a "high agreement"
and "low agreement" sequence is noise, not signal. This contrasts with the arithmetic setting
(felix-auto) where agreement clearly correlated with correctness.

### Next Steps

Several directions to try, one at a time:

- **Higher threshold (0.59+):** Force much more selective filtering. Risk: may keep too few
  sequences per cycle to train on.
- **Adaptive threshold:** Use per-cycle percentile (e.g., top 20%) instead of fixed value.
  This tracks the distribution shift.
- **Lower LR (1e-6):** The 1e-5 LR may be too aggressive for self-training. Slower updates
  might prevent the rapid divergence.
- **Longer sequences:** Generate 128 or 256 tokens instead of 64. Longer generations are
  harder and may produce more meaningful agreement signal.
- **More real data mixing:** 5 real batches per cycle may be insufficient to anchor the model.
  Try 10-20.

---

## Experiment S2: Adaptive Top-20% Threshold, LR 1e-5

**Date:** 2026-03-10 | **Checkpoint:** felix_v2/best.pt (step 21500, 94.50 val PPL)

**Hypothesis:** An adaptive percentile threshold (top 20% per cycle) will track the agreement
distribution shift and maintain consistent selectivity, preventing the keep-rate inflation
that caused S1's rapid degradation.

**Variable changed from S1:** Fixed threshold 0.57 replaced with adaptive top-20% percentile.

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | felix_v2/best.pt (94.50 val PPL) |
| Threshold | Top 20% per cycle (adaptive) |
| Self-improvement LR | 1e-5 |
| Sequences per cycle | 200 |
| Real data batches/cycle | 5 |
| Temperature | 0.8, top-k 50 |
| Prompt / gen length | 64 / 64 tokens |

### Results

| Metric | Value |
|--------|-------|
| Baseline val PPL | 94.50 |
| Final val PPL | 94.50 (patience stall) |
| Best val PPL seen | 96.79 (cycle 15) |
| Delta | +0.00 (reverted to baseline) |
| Cycles completed | 20 / 40 (stalled at patience=4) |
| Total kept | 567 |
| Time | 4109s (~68 min) |

**Per-cycle keep rate:**

| Cycle | Kept/Total | Keep Rate | Self Loss | Real Loss |
|-------|------------|-----------|-----------|-----------|
| 1 | 39/200 | 19.5% | 3.8673 | 4.5239 |
| 2 | 38/200 | 19.0% | 3.7637 | 4.6315 |
| 3 | 40/200 | 20.0% | 3.7692 | 4.6663 |
| 4 | 38/200 | 19.0% | 3.7767 | 4.5450 |
| 5 | 39/200 | 19.5% | 3.7225 | 4.7813 |
| 10 | 24/200 | 12.0% | 3.4424 | 4.5511 |
| 15 | 24/200 | 12.0% | 3.3986 | 4.5767 |
| 20 | 31/200 | 15.5% | 3.1435 | 4.8800 |

Note: keep rate dropped below 20% because the repetition filter removed some sequences that
passed the agreement threshold.

**Agreement distribution drift:**

| Cycle | Mean | Std | p50 | p75 | Adaptive Cutoff |
|-------|------|-----|-----|-----|-----------------|
| 1 | 0.575 | 0.010 | 0.577 | 0.582 | 0.583 |
| 5 | 0.581 | 0.012 | 0.581 | 0.589 | 0.591 |
| 10 | 0.593 | 0.017 | 0.592 | 0.606 | 0.607 |
| 15 | 0.598 | 0.020 | 0.600 | 0.612 | 0.616 |
| 20 | 0.603 | 0.021 | 0.607 | 0.619 | 0.622 |

### Analysis

S2 is a clear improvement over S1 in terms of stability: it ran 20 cycles without triggering
the 5% safety revert (best val PPL was 96.79 vs S1's 102.03). The adaptive threshold worked
as designed, maintaining ~20% selection rate and tracking the distribution upward (cutoff rose
from 0.583 to 0.622 as agreement mean drifted from 0.575 to 0.603).

However, the model still never improved over baseline. Every eval checkpoint was worse than
94.50. The key observations:

**1. Agreement drift is the fundamental problem, not the threshold.** The agreement mean
shifted from 0.575 to 0.603 over 20 cycles (+0.028), and std widened from 0.010 to 0.021.
The model is learning to produce higher-agreement outputs, but this doesn't correlate with
better language modeling. Agreement is a measure of stream consensus, not output quality.
Training on high-agreement sequences teaches the model to agree more, creating a self-
reinforcing loop that drifts away from the data distribution.

**2. Self-loss keeps dropping while val PPL stays flat/worsens.** Self-loss went from 3.87
to 3.14 (-19%) while val PPL stayed at ~97. The model fits its own outputs better without
generalizing. The 5 real-data batches per cycle slow the drift but don't prevent it.

**3. The signal-to-noise ratio in agreement may be too low for self-improvement.** At 11M
params, agreement std is only 0.010-0.021. The entire range of agreement values falls within
a narrow band (~0.52-0.64). There may not be enough meaningful variance to separate "the model
is confident and correct" from "the model is confident and wrong."

### Diagnosis

Two runs, same conclusion: agreement-filtered self-training on a language model does not
improve val PPL. The mechanism that worked for arithmetic (where agreement correlates with
correctness) does not transfer to open-ended language modeling (where agreement correlates
with stream consensus, which is orthogonal to quality).

Possible pivots:
- **Loss-filtered instead of agreement-filtered:** Keep sequences where the model's own loss
  is lowest (most "natural" generations). Agreement as tiebreaker, not primary filter.
- **Much lower LR (1e-6):** Maybe the updates are still too aggressive. But this risks
  doing nothing at all.
- **Rethink the approach:** Agreement may only be useful as a training signal when there's
  a verifiable correctness criterion. For open-ended LM, we might need a different signal
  entirely (e.g., perplexity under the original checkpoint as a quality filter).

---

## Experiment S3: Reference-Model Loss Filter, LR 1e-5

**Date:** 2026-03-10 | **Checkpoint:** felix_v2/best.pt (step 21500, 94.50 val PPL)

**Hypothesis:** Filtering by frozen reference model's perplexity (instead of agreement) will
select generations that stay close to the learned data distribution. Because the reference
model is frozen, the quality criterion can't drift like agreement did in S1-S2.

**Variable changed from S2:** Filter criterion switched from adaptive agreement percentile
to frozen reference model's per-sequence loss (top 20% lowest loss = most natural).

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | felix_v2/best.pt (94.50 val PPL) |
| Filter | Ref model loss, top 20% lowest |
| Self-improvement LR | 1e-5 |
| Sequences per cycle | 200 |
| Real data batches/cycle | 5 |
| Temperature | 0.8, top-k 50 |
| Prompt / gen length | 64 / 64 tokens |

### Results

| Metric | Value |
|--------|-------|
| Baseline val PPL | 94.50 |
| Final val PPL | 94.50 (patience stall) |
| Best val PPL seen | 95.94 (cycle 5) |
| Delta | +0.00 (reverted to baseline) |
| Cycles completed | 20 / 40 (stalled at patience=4) |
| Total kept | 745 |
| Time | 4532s (~76 min) |

**Per-cycle keep rate:**

| Cycle | Kept/Total | Self Loss | Real Loss |
|-------|------------|-----------|-----------|
| 1 | 36/200 | 3.0198 | 4.8204 |
| 5 | 36/200 | 2.8930 | 4.7373 |
| 10 | 35/200 | 2.8397 | 4.6462 |
| 15 | 39/200 | 2.7667 | 4.7665 |
| 20 | 38/200 | 2.8294 | 4.8064 |

**Reference loss distribution (stable, no drift):**

| Cycle | Ref Loss Mean | Ref Loss Std | p25 | p50 | Cutoff |
|-------|---------------|--------------|-----|-----|--------|
| 1 | 3.574 | 0.487 | 3.277 | 3.600 | 3.197 |
| 5 | 3.550 | 0.502 | 3.257 | 3.608 | 3.171 |
| 10 | 3.483 | 0.465 | 3.197 | 3.504 | 3.096 |
| 15 | 3.506 | 0.517 | 3.158 | 3.554 | 3.050 |
| 20 | 3.528 | 0.451 | 3.245 | 3.570 | 3.149 |

**Agreement distribution (no drift, confirming loss filter doesn't push agreement):**

| Cycle | Mean | Std |
|-------|------|-----|
| 1 | 0.574 | 0.011 |
| 5 | 0.574 | 0.011 |
| 10 | 0.576 | 0.012 |
| 15 | 0.575 | 0.011 |
| 20 | 0.577 | 0.010 |

### Analysis

S3 validates the frozen-reference approach: no drift in either agreement or ref loss
distributions. The filter is stable and well-calibrated. But the model still doesn't improve.

The key insight comparing across all three experiments:

| Exp | Filter | Drift? | Worst Val PPL | Stability |
|-----|--------|--------|---------------|-----------|
| S1 | Fixed agreement 0.57 | Yes (severe) | 102.03 | Safety revert at cycle 5 |
| S2 | Adaptive top-20% agreement | Yes (moderate) | 97.41 | Patience stall at cycle 20 |
| S3 | Ref model loss top-20% | No | 96.54 | Patience stall at cycle 20 |

S3 is the most stable (smallest degradation: +2.04 vs S2's +2.91 vs S1's +7.53), but
stability without improvement is still failure. The progression from S1 to S3 solved the
drift problem but revealed the deeper issue: **self-generated data at 11M scale is not
informative enough to improve the model, regardless of how it's filtered.**

The self-loss is consistently lower than real-loss (2.8-3.0 vs 4.5-4.8), meaning the model
finds its own generations much easier than real data. Training on easy self-generated sequences
doesn't push the model toward harder, more informative patterns. It's the equivalent of a
student studying only problems they already know how to solve.

---

## Conclusion: Self-Improvement at 11M Scale

Three experiments with three different filtering strategies all reached the same result:
+0.00 PPL (safety revert or patience stall). The approach does not work at this scale.

**Why it fails:**

1. **No verifiable signal.** In arithmetic (felix-auto), agreement correlated with
   correctness because there was a ground truth. In open-ended LM, agreement correlates
   with stream consensus, which is orthogonal to generation quality. The frozen ref-model
   loss is more stable but equally uninformative — it just measures how well the generation
   matches the model's existing distribution, which is circular.

2. **Self-generated data is too easy.** The model's own outputs have loss ~3.0 while real
   data has loss ~4.7. Training on filtered self-generations is training on the easy part
   of the distribution. There's no mechanism to push the model toward the hard, informative
   examples it hasn't mastered yet.

3. **11M scale may be too small.** At 11M params, the model has limited capacity. It may
   already be near its capacity ceiling on WikiText-103, leaving no room for self-improvement
   to find gains. The narrow agreement distribution (std=0.01) supports this — the model
   is similarly confident/uncertain everywhere, with no meaningful variance to exploit.

**What would need to change:**
- A verifiable quality signal (e.g., factual consistency, task-specific correctness)
- Adversarial or contrastive generation (generate hard examples, not easy ones)
- Much larger model where capacity ceiling is higher
- External reward signal (RLHF-style) rather than self-supervision
