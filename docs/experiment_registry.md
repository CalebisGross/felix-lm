# Experiment Registry

Pre-registration log for all Felix-LM experiments. Every experiment gets an entry
here BEFORE training starts, with hypothesis, variable, control, and prediction.
Results are filled in after completion.

## v3 Hub-and-Spoke Experiments

### EXP-1: v3_none
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Establish v3 transformer baseline (no spokes) for comparison
- **Variable:** N/A (baseline)
- **Control:** N/A (this IS the control)
- **Prediction:** Should be close to v2_baseline (~44.81) but may differ due to plain nn.Embedding vs StreamInit
- **Params:** 11.68M | **Config:** v3_none
- **Result:** 47.90 PPL
- **Verdict:** CONFIRMED — baseline established, 3.09 PPL worse than v2 due to embedding difference
- **Analysis:** The gap vs v2_baseline (44.81) comes from v3 using raw nn.Embedding while v2 uses StreamInitialization with learned projections. This sets the bar for spokes to clear.

### EXP-2: v3_base (progressive gates)
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Progressive gate schedule (small early, large late) helps spokes converge gradually
- **Variable:** gate_schedule=progressive (vs none in EXP-1)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Expect -0.5 to -1.0 PPL improvement
- **Params:** 12.01M | **Config:** v3_base
- **Result:** 48.61 PPL (+0.71 vs control)
- **Verdict:** REFUTED — progressive gates HURT
- **Analysis:** Starving early-layer spokes with small gates prevents them from learning useful projections. The progressive schedule imposed an assumption that was wrong.

### EXP-3: v3_uniform
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Uniform gates (all sigmoid(0)=0.5) let spokes self-organize
- **Variable:** gate_schedule=uniform (vs progressive in EXP-2)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Should beat progressive. Uncertain vs baseline — exploratory.
- **Params:** 12.01M | **Config:** v3_uniform
- **Result:** 47.64 PPL (-0.26 vs control)
- **Verdict:** CONFIRMED — uniform beats both progressive and baseline
- **Analysis:** Letting all spokes start at equal influence and learn their own schedule works better than imposing structure. First evidence that spokes help.

### EXP-4: v3_r8
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Rank 8 is enough capacity per spoke at 11M scale
- **Variable:** spoke_rank=8 (vs 16 in EXP-3)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Expect less improvement than r16 but still positive
- **Params:** 11.85M | **Config:** v3_r8
- **Result:** 48.82 PPL (+0.92 vs control)
- **Verdict:** REFUTED — rank 8 is too small, actively hurts
- **Analysis:** At rank 8 each spoke has too little capacity to learn a useful projection. The overhead of the gating mechanism isn't justified by the information the spokes can extract.

### EXP-5: v3_r32
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Rank 32 gives spokes enough capacity to extract more diverse signal
- **Variable:** spoke_rank=32 (vs 16 in EXP-3)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Expect -0.5 to -1.0 PPL, more than r16's -0.26
- **Params:** 12.34M | **Config:** v3_r32
- **Result:** 47.18 PPL (-0.72 vs control)
- **Verdict:** CONFIRMED — rank 32 is substantially better than r16
- **Analysis:** Efficiency sweet spot. The -0.72 PPL at 5.6% param overhead is the best spoke improvement per added parameter.

### EXP-6: v3_r64
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Rank 64 continues the monotonic rank scaling
- **Variable:** spoke_rank=64 (vs 32 in EXP-5)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Expect marginal gain over r32, diminishing returns
- **Params:** 12.99M | **Config:** v3_r64
- **Result:** 47.06 PPL (-0.84 vs control)
- **Verdict:** CONFIRMED — diminishing returns. Only -0.12 more than r32 for 5.6% more params
- **Analysis:** r32 to r64 gains 0.12 PPL for doubling spoke params. r16 to r32 gained 0.46 for the same doubling. r32 is the efficiency knee.

### EXP-7: v3_2spoke
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** 2 spokes provide enough diversity for useful feedback
- **Variable:** num_spokes=2 (vs 4 in EXP-3)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Expect less improvement than 4 spokes
- **Params:** 11.85M | **Config:** v3_2spoke
- **Result:** 48.62 PPL (+0.72 vs control)
- **Verdict:** REFUTED — 2 spokes actively hurt
- **Analysis:** Two probes don't provide enough diverse perspectives. The overhead of the gating and aggregation isn't justified without sufficient diversity.

### EXP-8: v3_8spoke
- **Date:** 2026-03-15
- **Status:** FAILED
- **Hypothesis:** 8 spokes provide more diversity than 4
- **Variable:** num_spokes=8 (vs 4 in EXP-3)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Uncertain — may help or may be too many
- **Params:** 12.34M | **Config:** v3_8spoke
- **Result:** Killed at step 500 (val_ppl 156.23, 2.5x slower)
- **Verdict:** INCONCLUSIVE — impractical due to compile overhead
- **Analysis:** torch.compile overhead with 8 spokes in a loop made training 2.5x slower. Not worth pursuing at this implementation.

### EXP-9: v3_uniform (random W_up init)
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Random W_up init (std=0.01) helps spokes bootstrap faster than zeros
- **Variable:** W_up init = random (vs zeros in EXP-3)
- **Control:** v3_uniform (zeros) = 47.64 PPL
- **Prediction:** Uncertain — could go either way
- **Params:** 12.01M | **Config:** v3_uniform (modified init)
- **Result:** 47.94 PPL (+0.30 vs zeros init)
- **Verdict:** REFUTED — random init hurts
- **Analysis:** Random init introduces noise that the model must overcome. Zeros lets spokes start silent and gradually bootstrap, which is less disruptive to the backbone.

### EXP-10: v3_deep22
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Extra depth is a better use of params than spokes
- **Variable:** 22 layers no spokes (vs 20 layers + spokes in EXP-5)
- **Control:** v3_r32 (20L + spokes) = 47.18 PPL, 12.34M params
- **Prediction:** Could go either way — depth vs diversity
- **Params:** 12.21M | **Config:** v3_deep22
- **Result:** 47.69 PPL (+0.51 vs v3_r32)
- **Verdict:** Spokes beat extra depth at matched params
- **Analysis:** 4 spokes at rank 32 are worth more than 2 additional transformer layers. Diverse cheap feedback > monolithic depth.

### EXP-11: v3_r32 at 5000 steps
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Spoke advantage holds with extended training
- **Variable:** Training steps = 5000 (vs 2500 in EXP-5)
- **Control:** v2_baseline at 5000 steps
- **Prediction:** v3_r32 should improve but may not catch v2_baseline due to embedding gap
- **Params:** 12.34M | **Config:** v3_r32
- **Result:** 41.89 PPL (v2_baseline at 5000 = 39.63)
- **Verdict:** CONFIRMED — spoke advantage persists but embedding gap remains
- **Analysis:** v3_r32 improves from 47.18 to 41.89 with more training (-5.29), similar trajectory to v2_baseline (44.81 to 39.63 = -5.18). The embedding gap (~2.3 PPL) is stable.

### EXP-12: v3_100m_none
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Establish 100M baseline for scale comparison
- **Variable:** Scale (100M vs 11M)
- **Control:** N/A (100M baseline)
- **Prediction:** Should be substantially better than 11M baseline due to more compute capacity
- **Params:** 109.6M | **Config:** v3_100m_none
- **Result:** 43.88 PPL
- **Verdict:** CONFIRMED — 100M baseline established
- **Analysis:** Embedding fraction drops from 55% to 23% at 100M, leaving much more budget for compute layers.

### EXP-13: v3_100m_r32
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Spoke advantage scales — should be larger at 100M than 11M
- **Variable:** 4 spokes r=32 at 100M scale (vs v3_100m_none)
- **Control:** v3_100m_none = 43.88 PPL
- **Prediction:** Expect -0.7 to -1.5 PPL (at least matching 11M's -0.72)
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 42.69 PPL (-1.19 vs control)
- **Verdict:** CONFIRMED — advantage grows with scale (1.5% at 11M -> 2.7% at 100M)
- **Analysis:** At 100M more of the param budget is in compute layers where spokes can contribute. The nearly doubled relative improvement suggests continued scaling gains.

### EXP-14: v3_proj
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Embedding projection closes the gap between v3 and v2 baselines
- **Variable:** embed_proj=True (vs False in EXP-1)
- **Control:** v3_none = 47.90 PPL
- **Prediction:** Expect -1.0 to -2.0 PPL based on v2's StreamInit advantage (~3 PPL)
- **Params:** 11.70M | **Config:** v3_proj
- **Result:** 46.57 PPL (-1.33 vs control)
- **Verdict:** CONFIRMED — projection alone worth -1.33 PPL
- **Analysis:** A simple linear projection after embedding captures much of v2's StreamInit advantage. Closes the gap from 3.09 PPL to 1.76 PPL.

### EXP-15: v3_proj_r32
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Embed projection and spokes target different bottlenecks and compound
- **Variable:** embed_proj=True + spokes (vs proj-only in EXP-14 and spokes-only in EXP-5)
- **Control:** v3_none = 47.90 PPL | proj alone = -1.33 | spokes alone = -0.72 | additive = -2.05
- **Prediction:** Expect -1.5 to -2.0 PPL (near-additive compounding)
- **Params:** 12.36M | **Config:** v3_proj_r32
- **Result:** 45.98 PPL (-1.92 vs control, 94% additive)
- **Verdict:** CONFIRMED — improvements compound, different bottlenecks
- **Analysis:** Projection improves embedding quality, spokes improve per-layer processing. Nearly perfect additivity (94%) confirms these are independent improvements.

### EXP-16: v3_100m_proj
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Embedding projection helps at 100M scale (where embedding is smaller fraction)
- **Variable:** embed_proj=True at 100M (vs v3_100m_none)
- **Control:** v3_100m_none = 43.88 PPL
- **Prediction:** Expect smaller relative gain than 11M (-0.5 to -1.0 PPL) since embedding is only 23% of params at 100M
- **Params:** 109.9M | **Config:** v3_100m_proj
- **Result:** 42.86 PPL (-1.02 vs control)
- **Verdict:** CONFIRMED — projection helps at scale, slightly less than predicted range but close
- **Analysis:** Projection gives -1.02 PPL at 100M vs -1.33 at 11M. The smaller absolute gain is expected since embedding is a smaller fraction of compute at 100M (23% vs 55%). Relative improvement: 2.3% at 100M vs 2.8% at 11M — fairly stable across scales.

### EXP-17: v3_100m_proj_r32
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Projection + spokes compound at 100M as they do at 11M
- **Variable:** embed_proj=True + spokes at 100M (vs v3_100m_r32 and v3_100m_proj)
- **Control:** v3_100m_none = 43.88 | spokes alone = -1.19 | proj alone = -1.02 | additive = -2.21
- **Prediction:** If 94% additive (like 11M), expect ~41.8 PPL. Best v3 result yet.
- **Params:** 112.5M | **Config:** v3_100m_proj_r32
- **Result:** 42.58 PPL (-1.30 vs baseline, 59% additive)
- **Verdict:** PARTIALLY CONFIRMED — still best 100M result, but compounding is weaker than at 11M
- **Analysis:** At 100M, proj+spokes give -1.30 total vs the expected -2.21 if fully additive (59% additivity vs 94% at 11M). The combined config still beats spokes-only (42.69) by 0.11 and proj-only (42.86) by 0.28, but the improvements partially overlap at this scale. This suggests that at 100M, projection and spokes may be competing for some of the same representational capacity. The spokes-only config (42.69, -1.19) remains the more efficient choice at 100M since projection adds minimal benefit on top of spokes.

### EXP-18: v3_100m_r64
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Higher rank spokes capture more at 100M where overhead is cheap (r64 = 4.8% vs r32's 2.4%)
- **Variable:** spoke_rank=64 (vs 32 in EXP-13)
- **Control:** v3_100m_r32 = 42.69 PPL | v3_100m_none = 43.88
- **Prediction:** At 11M, r64 gave -0.12 more than r32 (diminishing). At 100M expect larger gain since more compute to leverage: -0.3 to -0.5 more than r32, so ~42.2-42.4 PPL
- **Params:** 114.9M | **Config:** v3_100m_r64
- **Result:** 43.22 PPL (-0.66 vs none, +0.53 vs r32)
- **Verdict:** REFUTED — higher rank HURTS at 100M. r32 is optimal at both scales.
- **Analysis:** At 11M, r64 beat r32 by 0.12 PPL. At 100M, r64 is 0.53 PPL worse than r32. The extra spoke capacity may be overfitting or interfering with the backbone at scale. At 100M the backbone (d=512) has much richer representations and the spokes' job is to provide diverse *lightweight* feedback — making spokes more powerful (r=64) may cause them to overfit to specific patterns rather than providing diverse signal. This confirms r32 as the universal sweet spot. Skipping EXP-19 (r128) as it would likely be even worse.

### EXP-19: v3_100m_r128
- **Date:** 2026-03-15
- **Status:** SKIPPED
- **Hypothesis:** At 100M, rank 128 may be in the sweet spot since the backbone is much larger
- **Variable:** spoke_rank=128 (vs 64 in EXP-18)
- **Control:** v3_100m_r32 = 42.69 PPL | v3_100m_none = 43.88
- **Prediction:** Skipped — EXP-18 showed r64 is WORSE than r32 at 100M. r128 would likely be even worse.
- **Params:** ~120M | **Config:** v3_100m_r128
- **Result:** SKIPPED
- **Verdict:** SKIPPED based on EXP-18 results
- **Analysis:** r64 at 100M was 0.53 PPL worse than r32. Higher rank hurts at scale — spokes need to stay lightweight.

### EXP-20: v3_100m_r32_lr3e3
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Spokes may stabilize higher LR at 100M (like weight decay did in v2 HP sweep)
- **Variable:** LR 3e-3 (vs 2e-3 in EXP-13)
- **Control:** v3_100m_r32 at LR 2e-3 = 42.69 PPL
- **Prediction:** If spokes stabilize: ~41.5-42.0. If overshoot: >43.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 41.66 PPL / 1.1583 BPB (-1.03 vs LR 2e-3, -2.22 vs baseline)
- **Verdict:** CONFIRMED — LR 3e-3 is better than 2e-3 at 100M with spokes. NEW BEST 100M.
- **Analysis:** Higher LR gives a larger gain (-1.03) than any architecture change we tried at 100M. This mirrors the v2 autoresearch finding that LR was the biggest lever. Important caveat: we need to also test the baseline (no spokes) at LR 3e-3 to confirm the spoke advantage holds. The spoke advantage could be a LR artifact (as it was in EXP-81/82 with v2).

### EXP-21a: v3_100m_none_lr3e3 (skepticism check)
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** The baseline also improves at LR 3e-3, and the spoke advantage shrinks or disappears (as happened with v2 in EXP-81)
- **Variable:** LR 3e-3 on baseline (no spokes), to check if EXP-20's improvement is spokes or LR
- **Control:** v3_100m_none at LR 2e-3 = 43.88 PPL | v3_100m_r32 at LR 3e-3 = 41.66
- **Prediction:** If LR artifact: baseline at 3e-3 will be ~41.5-42.0 (matching spokes). If spoke advantage is real: baseline at 3e-3 > 42.5.
- **Params:** 109.6M | **Config:** v3_100m_none
- **Result:** 41.30 PPL / 1.1564 BPB — baseline BEATS spokes by 0.36 PPL at matched LR
- **Verdict:** CONFIRMED — the spoke advantage at 100M was a LR artifact. Spokes were masking suboptimal LR.
- **Analysis:** At LR 2e-3, spokes appeared to help (-1.19 PPL). At LR 3e-3, the baseline jumps by -2.58 PPL while spokes only jump by -1.03 — the baseline benefits MORE from higher LR. This is the same pattern as v2 EXP-81: Felix looked good at a suboptimal LR because the extra parameters provided implicit regularization that partially compensated for the too-low LR. Once LR is right, the plain transformer wins. The scientific method (organized skepticism, always check the baseline at the same settings) caught this. Without EXP-21a we would have claimed spokes scale to 100M — and been wrong.

### EXP-22: v3_100m_none_muon
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Muon's orthogonalized gradients converge faster than AdamW per step, giving better results in matched steps
- **Variable:** Optimizer: MuonAdamW (vs AdamW in EXP-21a at LR 3e-3)
- **Control:** v3_100m_none with AdamW at LR 3e-3 = 41.30 PPL / 1.1564 BPB
- **Prediction:** Muon typically converges 1.5-2x faster per step on transformer weights. Expect ~40.0-41.0 PPL if Muon's advantage transfers.
- **Params:** 109.6M | **Config:** v3_100m_none
- **Result:** 111.10 PPL / 1.4634 BPB — nearly 3x worse than AdamW
- **Verdict:** REFUTED — Muon with default LRs fails badly. Needs dedicated HP sweep.
- **Analysis:** Muon requires fundamentally different LR scales than AdamW. Our setup used muon_lr=0.0245 (muP-scaled) for 2D weights and embed_lr=0.03 (10x base) for embeddings. These defaults came from Karpathy's much smaller models and don't transfer to our 100M config. The embedding LR of 0.03 in particular may be far too aggressive. Muon is not a drop-in replacement — it needs its own HP sweep, which is an expensive multi-experiment commitment. The NS iteration count (10 vs Karpathy's 5) may also be too many, wasting compute on over-precise orthogonalization. Parking Muon for now; it's a research project in itself.

### EXP-23: v3_100m_r32_lr3e3 (for qualitative comparison)
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Even if spokes lose on PPL at matched LR, the qualitative differences found at 11M (calibration, rare tokens, long-range) may persist at 100M
- **Variable:** Spokes at LR 3e-3 (to match baseline's optimal LR for fair qualitative comparison)
- **Control:** v3_100m_none at LR 3e-3 = 41.30 PPL / 1.1564 BPB
- **Prediction:** PPL will be ~41.5-42.0 (slightly worse than baseline). But calibration, rare token CE, and positional patterns from 11M may survive.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 40.98 PPL / 1.1539 BPB — BEATS baseline by 0.32 PPL at matched LR!
- **Verdict:** CONFIRMED (partially) — spokes win on PPL AND we can now test qualitative properties
- **Analysis:** EXP-21a's conclusion ("spoke advantage was LR artifact") was WRONG — the baseline checkpoint got overwritten by the Muon run. Retrained baseline at LR 3e-3 = 41.48 PPL. Spokes at 40.98 = -0.50 PPL advantage (1.2%). Spoke advantage is real at 100M, just smaller than at LR 2e-3. Deep qualitative analysis confirmed: better calibration (ECE 0.0067 vs 0.0082), hard-token specialization, learned convergence schedule, completely different representations.

### EXP-24: v3_100m_r32_lr4e3
- **Date:** 2026-03-15
- **Status:** COMPLETED
- **Hypothesis:** Spokes may benefit from higher LR than baseline
- **Variable:** LR 4e-3 (vs 3e-3 in EXP-23)
- **Control:** v3_100m_r32 at LR 3e-3 = 40.98 PPL
- **Prediction:** If spokes want higher LR: ~40.5. If overshoot: >41.5.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 43.55 PPL / 1.1726 BPB — overshoot, +2.57 vs LR 3e-3
- **Verdict:** REFUTED — LR 3e-3 is optimal, 4e-3 overshoots
- **Analysis:** LR 4e-3 destabilizes training at 100M with spokes. Skipping EXP-25 (baseline at LR 4e-3) since spokes already lost. Moving to spoke-specific LR (EXP-26) which is a more targeted intervention.

### EXP-25: v3_100m_none_lr4e3
- **Date:** 2026-03-15
- **Status:** SKIPPED (EXP-24 showed LR 4e-3 overshoots for spokes, no need to test baseline)
- **Hypothesis:** Baseline may also improve at LR 4e-3 (skepticism check for EXP-24)
- **Variable:** LR 4e-3 on baseline (matching EXP-24 for fair comparison)
- **Control:** v3_100m_none at LR 3e-3 = 41.48 PPL
- **Prediction:** If 3e-3 was near-optimal: ~41.5-42.0 (slight overshoot). If not: <41.
- **Params:** 109.6M | **Config:** v3_100m_none
- **Result:**
- **Verdict:**
- **Analysis:**

### EXP-26: v3_100m_r32_spokeLR
- **Date:** 2026-03-15
- **Status:** RUNNING
- **Hypothesis:** Spoke parameters benefit from higher LR than backbone — they're small, low-rank, and need to learn fast
- **Variable:** Separate LR for spoke params (3x backbone LR) via param groups
- **Control:** v3_100m_r32 at LR 3e-3 (uniform) = 40.98 PPL
- **Prediction:** If spoke LR matters: ~40.0-40.5. Small params often benefit from higher LR.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 40.71 PPL / 1.1516 BPB — NEW BEST 100M! -0.27 vs uniform LR, -0.77 vs baseline
- **Verdict:** CONFIRMED — spoke-specific LR helps. Spoke advantage grows from -0.50 to -0.77 PPL.
- **Analysis:** Spoke params (2.6M out of 112.3M) are low-rank projections that need to learn fast to be useful. At uniform LR, they're undertrained relative to the 86.5M backbone params. Giving them 3x LR lets them keep up. This is a muP-like insight: different parameter groups have different optimal LRs.

### EXP-27: v3_100m_r32_swiglu (with spoke-LR 3x)
- **Date:** 2026-03-15
- **Status:** RUNNING
- **Hypothesis:** SwiGLU-style spokes (gated down projection) are more expressive per rank than standard SiLU spokes
- **Variable:** spoke_swiglu=True + spoke-LR 3x (building on EXP-26's finding)
- **Control:** v3_100m_r32 spoke-LR 3x = 40.71 PPL (EXP-26)
- **Prediction:** SwiGLU is strictly more expressive. Expect -0.1 to -0.3 PPL on top of spoke-LR gain.
- **Params:** 113.6M | **Config:** v3_100m_r32_swiglu
- **Result:** 41.71 PPL / 1.1591 BPB — worse than standard spokes by 1.0 PPL
- **Verdict:** REFUTED — SwiGLU hurts. Extra gate projection adds noise, not expressiveness.
- **Analysis:** SwiGLU works well in FFN layers where the gate and value projections are operating on rich representations at full width. In spokes, the projections are low-rank (r=32) — there may not be enough capacity in the rank-32 bottleneck for the gating to be useful. The extra d*r parameters (1.3M) are wasted. Simple SiLU spokes remain optimal.

### EXP-28: v3_100m_r32_spokeLR5x
- **Date:** 2026-03-16
- **Status:** COMPLETED
- **Hypothesis:** Spoke LR 5x may be even better than 3x
- **Variable:** spoke-lr-mult=5.0 (vs 3.0 in EXP-26)
- **Control:** v3_100m_r32 spoke-LR 3x = 40.71 PPL
- **Prediction:** If 3x was undertrained: ~40.3-40.5. If overshoot: >41.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 41.86 PPL / 1.1602 BPB — overshoot, +1.15 vs 3x
- **Verdict:** REFUTED — 3x is the sweet spot. Skipping EXP-29 (10x would be worse).
- **Analysis:** Spoke LR sweep: 1x=40.98, 3x=40.71, 5x=41.86. Clear peak at 3x. The 2.6M spoke params need ~3x the backbone LR but not more.

### EXP-29: SKIPPED (10x spoke LR would overshoot worse than 5x)

### EXP-30: v3_r32_spokeLR3x (11M)
- **Date:** 2026-03-16
- **Status:** COMPLETED
- **Hypothesis:** Spoke-specific 3x LR also helps at 11M
- **Variable:** spoke-lr-mult=3.0 at 11M (vs uniform LR in v3_r32 = 47.18)
- **Control:** v3_r32 at LR 2e-2 (uniform) = 47.18 PPL
- **Prediction:** If 3x helps at 11M like 100M: ~46.5-46.8
- **Params:** 12.34M | **Config:** v3_r32
- **Result:** 47.54 PPL / 1.1997 BPB — worse by 0.36 PPL
- **Verdict:** REFUTED — spoke-LR 3x hurts at 11M. The optimal multiplier is scale-dependent.
- **Analysis:** At 11M, backbone LR is 2e-2 (already high). Spoke-LR 3x = 6e-2 overshoots. At 100M, backbone LR is 3e-3, so spoke-LR 3x = 9e-3 — a more reasonable absolute value. The optimal spoke LR may be ~9e-3 in absolute terms, not a fixed multiplier of backbone LR.

### EXP-32: v3_r32_spokeLR_abs9e3 (11M)
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** Optimal spoke LR is ~9e-3 absolute (not a fixed multiplier). At 11M with backbone LR 2e-2, this means spoke-lr-mult=0.45.
- **Variable:** spoke-lr-mult=0.45 (so spoke LR = 2e-2 * 0.45 = 9e-3, matching 100M's optimal)
- **Control:** v3_r32 uniform LR = 47.18 PPL | v3_r32 spoke-LR 3x = 47.54
- **Prediction:** If absolute LR theory is right: ~46.5-47.0 (better than both uniform and 3x)
- **Params:** 12.34M | **Config:** v3_r32
- **Result:** 47.77 PPL / 1.2011 BPB — worse than both uniform (47.18) and 3x (47.54)
- **Verdict:** REFUTED — absolute LR theory is wrong. At 11M, spokes prefer uniform LR.
- **Analysis:** Spoke LR is scale-dependent. At 11M (LR 2e-2, spokes=5.6% params): uniform is optimal. At 100M (LR 3e-3, spokes=2.4% params): 3x is optimal. The likely driver is param fraction — when spokes are a larger fraction, uniform LR is fine. When they're a tiny fraction of a large model, they need a boost to keep up.

### EXP-33: v3_100m_r32_every2 (selective spokes)
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** Spokes on every other layer (10 of 20) capture most of the benefit at half the cost
- **Variable:** spoke_every_n=2 + spoke-LR 2x (10 spoke layers vs 20)
- **Control:** v3_100m_r32 spoke-LR 2x (20 spoke layers) = 40.62 PPL
- **Prediction:** If benefit is distributed: ~40.9-41.1 (small loss). If concentrated in specific layers: could match or beat.
- **Params:** ~111M | **Config:** v3_100m_r32_every2
- **Result:** 41.27 PPL / 1.1556 BPB — still beats baseline (-0.21) but loses 75% of every-layer benefit
- **Verdict:** Benefit is distributed across layers, not concentrated. Every-layer spokes are worth the cost.
- **Analysis:** Halving spokes (10 of 20 layers) loses 0.65 of the 0.86 PPL advantage. The 0.1 steps/s speedup doesn't compensate. Spokes contribute cumulatively through depth — consistent with the learned gate schedule where all layers participate (gates 0.15-0.74).

### EXP-34: v3_100m_proj_r32_spokeLR2x
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** Embed proj + spokes with optimized spoke-LR 2x compound better than the earlier 59% (which used uniform LR)
- **Variable:** embed_proj=True + spoke-LR 2x (combining best of EXP-16 and EXP-31)
- **Control:** v3_100m_r32 spoke-LR 2x = 40.62 | baseline = 41.48 | proj alone ~42.86 (LR 2e-3, not retested at 3e-3)
- **Prediction:** If they compound well: ~40.0-40.3. Proj gave -1.02 alone at LR 2e-3, but at 3e-3 it may be less.
- **Params:** 112.5M | **Config:** v3_100m_proj_r32
- **Result:** 39.76 PPL / 1.1447 BPB — NEW BEST 100M! -1.72 vs baseline, -0.86 vs spokes-only
- **Verdict:** CONFIRMED — proj + spokes + spoke-LR 2x compound strongly. Best result in entire project.
- **Analysis:** Spokes alone: -0.86. Proj + spokes + spoke-LR: -1.72. The projection doubles the spoke benefit — better embeddings give spokes richer signal to work with. At 2.6% total overhead (2.6M spokes + 0.3M proj on 109.6M baseline), this is excellent efficiency. Deep analysis confirms: ECE 0.0062 (best yet), sharp learned convergence (layers 0-8 explore, 9-17 converge, 18-19 pullback), hard-token specialization persists (-3.4% on hardest quintile), representations completely different (cosine ~0.00).

### EXP-35: v3_proj_r32_spokeLR2x (11M — does spoke-LR 2x help with proj at 11M?)
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** At 11M, spoke-LR 3x hurt but spoke-LR 2x might be the sweet spot, especially combined with proj
- **Variable:** embed_proj + spoke-LR 2x at 11M (vs v3_proj_r32 uniform LR = 45.98)
- **Control:** v3_proj_r32 (uniform LR) = 45.98 PPL
- **Prediction:** Uncertain — 3x hurt at 11M, but 2x is gentler. Could go either way. ~45.5-46.5.
- **Params:** 12.36M | **Config:** v3_proj_r32
- **Result:** 46.12 PPL / 1.1902 BPB — slightly worse than uniform (45.98) by 0.14
- **Verdict:** REFUTED — spoke-specific LR doesn't help at 11M regardless of multiplier.
- **Analysis:** Consistent with EXP-30/32: at 11M, backbone LR 2e-2 is already optimal for spokes. Spoke-LR separation is a 100M-scale technique only.

### MI300X-SCALING-1: v3_baseline_100m (1B tokens Dolma)
- **Date:** 2026-03-16
- **Status:** COMPLETED (after fixing grad_accum)
- **Hypothesis:** 100M baseline converges on 1B tokens of Dolma
- **Variable:** Real-scale training (1B tokens Dolma vs 2500 steps WikiText)
- **Control:** N/A (first real-scale v3 run)
- **Prediction:** PPL should reach 20-30 range
- **Params:** 109.9M | **Config:** v3_baseline_100m
- **Result:** best val_ppl=542.97, BPB=1.957 (after fixing grad_accum from 17 to 4)
- **Verdict:** COMPLETED — high absolute PPL due to domain mismatch (Dolma train, WikiText eval) and seq_len=2048
- **Analysis:** First attempt diverged (grad_accum=17 = only 120 opt steps). Fixed with grad_accum=4 (2179 opt steps). Lesson: always verify optimizer step count.

### MI300X-SCALING-2: v3_100m_proj_r64 (1B tokens Dolma)
- **Date:** 2026-03-16
- **Status:** COMPLETED
- **Hypothesis:** Spokes beat baseline at 100M with real-scale training
- **Variable:** proj + r64 spokes + spoke-LR 2x
- **Control:** v3_baseline_100m = 542.97 PPL
- **Prediction:** Based on local experiments (-5.6%), expect ~510-520
- **Params:** 115.2M | **Config:** v3_100m_proj_r64
- **Result:** best val_ppl=503.30, BPB=1.946 — SPOKES WIN by 7.3%
- **Verdict:** CONFIRMED — spoke advantage holds and GROWS at real scale
- **Analysis:** Advantage increased from 5.6% (local) to 7.3% (1B tokens). Deep analysis confirmed: 1.9x better calibration (ECE 0.029 vs 0.056), -7.4% on hardest tokens, learned convergence, divergent representations.

### MI300X-SCALING-3: v3_baseline_500m (250M tokens Dolma)
- **Date:** 2026-03-17
- **Status:** COMPLETED
- **Hypothesis:** Establish 500M baseline
- **Variable:** Scale to 500M params (directional, 250M tokens due to budget)
- **Control:** N/A
- **Prediction:** Lower PPL than 100M due to more capacity
- **Params:** 455.2M | **Config:** v3_baseline_500m
- **Result:** best val_ppl=795.55, BPB=2.073
- **Verdict:** COMPLETED — severely undertrained (0.55 tokens/param vs 100M's 9.1 tokens/param)
- **Analysis:** 795 PPL worse than 100M's 543 because 250M tokens is not enough for a 500M model.

### MI300X-SCALING-4: v3_500m_proj_r64 (250M tokens Dolma)
- **Date:** 2026-03-17
- **Status:** COMPLETED
- **Hypothesis:** Spokes beat baseline at 500M
- **Variable:** proj + r64 spokes + spoke-LR 2x at 500M
- **Control:** v3_baseline_500m = 795.55 PPL
- **Prediction:** If 100M advantage holds: ~740 PPL
- **Params:** 467.8M | **Config:** v3_500m_proj_r64
- **Result:** best val_ppl=832.56, BPB=2.094 — spokes LOSE by 4.7%
- **Verdict:** INCONCLUSIVE — spokes hurt at 500M with these settings, but LR was untuned and training was severely undertrained. Cannot distinguish HP issue from genuine scaling failure.
- **Analysis:** Spokes trailed at every checkpoint. Three confounds: (1) LR 3e-4 guessed, not tuned (spoke-LR 2x calibrated for 100M only). (2) 250M tokens = 0.55 tokens/param, backbone too undertrained for spokes to add value. (3) Possible genuine scaling limit. Would need 1B+ tokens and LR sweep to resolve.

### EXP-36: v3_100m_proj_none_lr3e3 (skepticism: does proj help baseline too?)
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** Baseline + proj may also improve, shrinking the spoke advantage. Must check before claiming -1.72 PPL.
- **Variable:** embed_proj=True on baseline (no spokes) at LR 3e-3
- **Control:** v3_100m_none at LR 3e-3 = 41.48 PPL
- **Prediction:** Proj gave -1.02 at LR 2e-3. At LR 3e-3, expect smaller: ~40.8-41.2. The real spoke advantage is EXP-34 minus this.
- **Params:** 109.9M | **Config:** v3_100m_proj (gate_schedule=none)
- **Result:** 40.15 PPL / 1.1469 BPB — proj alone gives -1.33 at LR 3e-3 (even better than at 2e-3!)
- **Verdict:** CONFIRMED — projection is a bigger lever than spokes at 100M. Fair spoke advantage = 39.76 - 40.15 = -0.39.
- **Analysis:** The 2x2 at 100M with LR 3e-3: no-proj/no-spokes=41.48, proj-only=40.15 (-1.33), spokes-only=40.62 (-0.86), proj+spokes=39.76 (-1.72). Additivity: -1.33 + -0.86 = -2.19 expected, got -1.72 (79% additive). Better than the 59% at LR 2e-3, but projection still accounts for most of the gain. Spokes add -0.39 on top of projection — a real but modest improvement. The qualitative benefits (calibration, hard tokens) may be the stronger argument for spokes than raw PPL.

### EXP-37: v3_100m_proj_r64_spokeLR2x (higher rank with optimized LR)
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** r64 failed at uniform LR (EXP-18: 43.22), but with spoke-LR 2x + proj it may work since spokes learn faster
- **Variable:** spoke_rank=64 (vs 32 in EXP-34), with proj and spoke-LR 2x
- **Control:** v3_100m_proj_r32 spoke-LR 2x = 39.76 PPL
- **Prediction:** If the earlier r64 failure was a LR issue: ~39.3-39.6. If rank is genuinely worse at 100M: >40.
- **Params:** ~118M | **Config:** v3_100m_proj_r64
- **Result:** 39.15 PPL / 1.1399 BPB — NEW BEST! r64 works with spoke-LR. -1.00 vs proj-only (40.15).
- **Verdict:** CONFIRMED — EXP-18's r64 failure was a LR artifact. With spoke-LR 2x, higher rank helps.
- **Analysis:** r64 + proj + spoke-LR 2x: 39.15 vs r32 + proj + spoke-LR 2x: 39.76. The extra rank gives -0.61 PPL for ~5.3M more params (4.8% overhead). Fair spoke advantage (vs proj-only baseline): 39.15 - 40.15 = -1.00 PPL. Spokes now account for a meaningful 2.5% improvement on top of projection. The spoke-LR discovery unlocked rank scaling that was previously hidden.

### EXP-38: v3_100m_proj_r128_spokeLR2x
- **Date:** 2026-03-16
- **Status:** RUNNING
- **Hypothesis:** Rank scaling continues with spoke-LR 2x. r128 may push further.
- **Variable:** spoke_rank=128 (vs 64 in EXP-37)
- **Control:** v3_100m_proj_r64 spoke-LR 2x = 39.15 PPL
- **Prediction:** r32->r64 gave -0.61. Diminishing returns: r64->r128 expect -0.2 to -0.4, so ~38.8-39.0.
- **Params:** ~123M | **Config:** v3_100m_proj_r128
- **Result:**
- **Verdict:**
- **Analysis (EXP-38):**

### EXP-31: v3_100m_r32_spokeLR2x
- **Date:** 2026-03-16
- **Status:** COMPLETED
- **Hypothesis:** 2x may be closer to optimal than 3x
- **Variable:** spoke-lr-mult=2.0 (between 1x=40.98 and 3x=40.71)
- **Control:** v3_100m_r32 spoke-LR 3x = 40.71 PPL
- **Prediction:** Should be between 40.71 and 40.98.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:** 40.62 PPL / 1.1509 BPB — NEW BEST! Beats 3x by 0.09.
- **Verdict:** CONFIRMED — 2x is the peak. Spoke LR curve: 1x=40.98, 2x=40.62, 3x=40.71, 5x=41.86.
- **Analysis:** The spoke LR optimum at 100M is 2x backbone (6e-3 absolute). The curve is fairly flat between 2x and 3x (0.09 PPL difference) but drops off sharply at 5x. Total spoke advantage at 100M with optimized LR: -0.86 PPL (2.1% improvement).

### EXP-29: v3_100m_r32_spokeLR10x
- **Date:** 2026-03-16
- **Status:** REGISTERED
- **Hypothesis:** Find the upper bound — where does spoke LR start hurting?
- **Variable:** spoke-lr-mult=10.0 (vs 3.0 in EXP-26)
- **Control:** v3_100m_r32 spoke-LR 3x = 40.71 PPL
- **Prediction:** Likely overshoot at 10x. Expect >41.
- **Params:** 112.3M | **Config:** v3_100m_r32
- **Result:**
- **Verdict:**
- **Analysis (EXP-29):**
