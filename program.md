# Felix-LM Autoresearch

Autonomous research loop for improving Felix-LM v2 architecture.

## Goal

Make the Felix v2 architecture — with its multi-stream processing, CentralPost
communication hub, and adaptive agreement-driven merging — competitive with or
better than a plain transformer at the same parameter budget.

| Model | Val PPL | Config |
|-------|---------|--------|
| Plain transformer (v2_baseline) | 44.81 | 0 stream layers, 20 refine, d128, optimal HPs |
| Felix v2 (v2_stream2) | 49.03 | 2 stream + 18 refine, d64, optimal HPs |
| Felix v2 (v2_stream4) | 50.86 | 4 stream + 16 refine, d64, optimal HPs |

Primary goal: close the gap between Felix (with streams) and the plain transformer.
The architecture must retain at minimum: multi-stream processing, CentralPost, and
adaptive merge. A "Felix" that strips out all Felix components is not a solution.

## Architecture constraints (MANDATORY)

The model MUST include these Felix-defining features:

1. **Multi-stream processing** — at least 2 streams with independent transformer blocks
2. **CentralPost** — the shared communication hub between streams
3. **Adaptive merge** — agreement-driven convergence between streams

Removing these turns v2 into a plain transformer. That's not what we're optimizing.
The research question is: how do we make these features HELP rather than hurt?

## What you modify

- `felix_lm/v2/config.py` — config dataclass, hyperparameters
- `felix_lm/v2/model.py` — model architecture
- `felix_lm/v2/` — any v2 module files (central_post.py, adaptive_merge.py, felix_layer.py)
- `scripts/train.py` — training loop, optimizer, LR schedule (only the v2 path)

## What you do NOT modify

- `felix_lm/config.py` or `felix_lm/model.py` (v1, frozen)
- `scripts/evaluate.py` (eval harness is ground truth)
- Test files (update tests only if your changes break the API)
- Data loading, tokenizer, evaluation metric

## Training command

```bash
python scripts/train.py --config <name> --batch-size 8 --grad-accum 8 \
  --device cuda --lr 2e-2 --max-steps 2500 --weight-decay 0.1 \
  --beta2 0.99 --warmup-steps 500 --compile --no-wandb
```

`--max-steps 2500` with `--grad-accum 8` gives ~30 minutes on our GPU.
Compare all experiments against `v2_baseline` (44.81 PPL) at these same settings.

## Evaluation

The training script evaluates during training. The metric is **val PPL** (lower is better).

## The experiment loop

LOOP FOREVER:

1. Look at current state: read the latest code, check recent results
2. Form a hypothesis — what change might improve val PPL and why
3. Modify the code (config, model, training — whatever is needed)
4. Run tests: `pytest tests/ -q` (must pass before training)
5. git commit with descriptive message
6. Train: run the training command
7. Log results to `autoresearch_results.tsv`
8. If val PPL improved: KEEP the commit, advance
9. If val PPL is same or worse: revert the code change (keep the TSV log)
10. Repeat from step 1

## Results tracking

Log to `autoresearch_results.tsv` (tab-separated):

```
commit	val_ppl	status	description
```

- commit: short hash (7 chars), or `-` if not committed
- val_ppl: validation perplexity (0.0 for crashes)
- status: keep, discard, or crash
- description: what was tried

## Constraints

- Parameter budget: ~10-12M total (comparable to M0's 11.17M)
- Must pass `pytest tests/ -q` before running
- Don't blow up VRAM beyond 16GB
- Architecture MUST include multi-stream + CentralPost + adaptive merge

## Locked-in hyperparameters (from 58 experiments)

These have been thoroughly swept and should NOT be re-tested:

| Parameter | Value | Alternatives tested |
|-----------|-------|-------------------|
| LR | 2e-2 | 1e-3 through 3e-2 |
| Weight decay | 0.1 | 0 through 0.2 |
| beta1 | 0.9 | 0.85 |
| beta2 | 0.99 | 0.95, 0.98, 0.999 |
| Grad clip | 1.0 | 0.5, 2.0 |
| Warmup | 500 | 250, 750 |
| Batch size | 8 x 8 (eff 64) | 8 x 4 |
| Dropout | 0.0 | 0.05 |
| ffn_mult | 4 | 3 |
| Refine heads | 4 | 2, 8 |
| Helical RoPE turns | 2 | 0 |
| torch.compile | yes | — |

## Why streams hurt (the problem to solve)

At 11M params with d_embed=128:
- Embedding: ~6.4M (58% of budget)
- Compute budget: ~5.3M params
- Each stream layer at d_stream=64: ~4 x 53K = 212K (4 streams)
- Each refine layer at d_embed=128: ~262K
- CentralPost per layer: ~33K overhead
- Net cost of 1 stream layer: trades ~262K of refine capacity for 212K+33K of stream capacity

The problem is **param efficiency**: stream layers at d_stream=64 are half-width,
and CentralPost/merge add overhead on top. Trading refine layers for stream layers
is a net compute downgrade.

## Ideas to explore

The challenge is making streams + CentralPost + merge worth their parameter cost:

### Make stream layers param-competitive
- Run streams at d_stream=128 (same width as refine) — eliminates the bottleneck
  but CentralPost now projects 128->d_post->128 which costs more
- Reduce CentralPost overhead (smaller d_post, no gating, simpler projection)
- Share weights across stream blocks (all streams use same weights, differentiate
  via different init or different attention patterns)

### Make CentralPost more useful
- CentralPost currently does gated read+write per stream. Maybe:
  - Use cross-attention instead of gated linear projection
  - Make CentralPost larger (d_post > d_stream) as a richer shared memory
  - Initialize CentralPost from the embedding instead of zeros

### Rethink the merge mechanism
- Current merge interpolates toward the mean based on agreement. Maybe:
  - Use a learned projection instead of mean (like an attention-weighted combine)
  - Merge once at the end instead of every layer
  - Start with merged (all streams identical) and let them diverge, then re-merge

### Hybrid architectures
- Stream layers early (where features diverge), refine layers late (where they converge)
- Shared-weight streams (same block, different CentralPost interactions)
- Mixture-of-experts style: streams as experts, CentralPost as router

### Training innovations
- Different LR for stream vs refine layers
- Curriculum: start without streams, add them partway through training
- Auxiliary loss on stream diversity (encourage specialization)

## Key context from 58 experiments

- v2_baseline (plain transformer): 44.81 PPL — the target to match
- v2_stream2 (2 stream + 18 refine): 49.03 — streams cost +4.14 PPL
- v2_stream4 (4 stream + 16 refine): 50.86 — streams cost +5.97 PPL
- More streams = worse at 11M (monotonic from exp 3-8 at old LR, confirmed at optimal LR)
- The gap is real: even at the correct LR and hyperparams, streams actively hurt
- Streams at d_stream=64 are half the width of refine layers — unfair comparison
- CentralPost + merge overhead makes it worse

## NEVER STOP

Do not pause to ask the human. They may be asleep. Run experiments indefinitely
until manually interrupted. If stuck, think harder, try more radical changes.
