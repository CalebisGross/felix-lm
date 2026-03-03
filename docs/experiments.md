# Felix-LM Experiment Log

## Training Setup (all 11M-param experiments)
- Dataset: WikiText-103 (train/validation/test)
- Sequence length: 512
- Effective batch size: 32 (batch_size=8, grad_accum=4)
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1, grad_clip=1.0)
- Schedule: Cosine LR with 1000-step linear warmup
- Epochs: 3 (~21,500 steps)
- Hardware: AMD Radeon RX 7800 XT (16GB VRAM), ROCm 7.1.1

---

## Experiment 1: M2 MSPM-HETERO (primary architecture)
**Config:** `m2` — 4→2→1 streams, linear→sliding_window→full_causal, deep supervision ON
**Params:** 11,092,608
**Date:** 2026-03-01

| Split | Loss | PPL | Stage 0 PPL | Stage 1 PPL | Stage 2 PPL |
|-------|------|-----|-------------|-------------|-------------|
| Validation | 5.0325 | 153.31 | 317.01 | 138.88 | 124.37 |
| Test | 5.0116 | 150.14 | 309.89 | 136.25 | 121.81 |

**Notes:** Progressive stage improvement confirms merge mechanism working.
Best checkpoint at step 21,500.

---

## Experiment 2: M0 UNIFORM (baseline)
**Config:** `m0` — 1 stream, 18 layers, full causal, standard RoPE
**Params:** 11,172,736
**Date:** 2026-03-02

| Split | Loss | PPL |
|-------|------|-----|
| Test | 4.7506 | 115.66 |

**Notes:** Standard transformer baseline. Parameter-matched to M2.

---

## Experiment 3: M2-FULLCAUSAL (attention ablation)
**Config:** `m2_fullcausal` — 4→2→1 streams, ALL full causal, deep supervision ON
**Params:** 11,092,608
**Date:** 2026-03-02
**Hypothesis:** Linear attention in Stage 0 is the bottleneck, not multi-stream merging.

| Split | Loss | PPL | Stage 0 PPL | Stage 1 PPL | Stage 2 PPL |
|-------|------|-----|-------------|-------------|-------------|
| Test | 4.8605 | 129.09 | 244.51 | 118.07 | 107.49 |

**Key finding:** Stage 2 final output (107.49) BEATS M0 baseline (115.66).
Switching to full causal closed ~60% of the M2→M0 gap (150→129).
Multi-stream merging adds value; linear attention was the bottleneck.

### Gate analysis (post-training):
- Gate bias drifted from init=1.0 to ~0.65 (learned ~35% selectivity)
- Stream A/B halves gated equally (no stream dominance)
- Merge projections use near-full rank (97/128 and 106/128)

### Stream specialization:
- Pairwise Q-weight cosine similarity ~0.00 across all stream pairs
- Streams learned fully orthogonal representations (genuine specialization)
- Embedding projections also near-orthogonal

---

## Experiment 4: M2-NOSUP (deep supervision ablation)
**Config:** `m2_nosup` — 4→2→1 streams, hetero attention, deep supervision OFF
**Params:** 11,092,608
**Date:** 2026-03-03
**Hypothesis:** Deep supervision forces early stages to predict tokens, hurting final output.

| Split | Loss | PPL |
|-------|------|-----|
| Test | 4.7742 | 118.41 |

**Key finding:** Removing deep supervision is the single biggest improvement (+32 PPL over M2).
Closes to within 2.75 PPL of M0 baseline — even with hetero attention (linear in Stage 0).
Deep supervision was actively harmful at this scale, likely because forcing early stages to
predict tokens conflicts with learning good intermediate representations for merging.

---

## Planned Experiments

### Experiment 5: M2-BEST (fullcausal + no supervision)
**Hypothesis:** Combining both improvements yields best small-scale result.
**Target:** Beat M0's 115.66 on overall metric.
**Hardware:** Mac Mini M4 (MPS backend)

### Experiment 6: Scale to 100M params
**Dataset:** FineWeb-Edu
**Hardware:** MI300X droplet
**Blocked on:** Finalizing best small-scale config.

---

## Summary Table

| Model | Attention | Deep Sup | Test PPL | Stage 2 PPL | vs M0 |
|-------|-----------|----------|----------|-------------|-------|
| M0 (baseline) | full causal | N/A | **115.66** | — | — |
| M2 | hetero | ON | 150.14 | 121.81 | +29.8% |
| M2-fullcausal | full causal | ON | 129.09 | **107.49** | +11.6% |
| M2-nosup | hetero | OFF | 118.41 | — | +2.4% |
