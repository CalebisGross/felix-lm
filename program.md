# Felix-LM Autoresearch

Autonomous research loop for improving Felix-LM v2 architecture.

## Goal

Beat the current best val PPL on WikiText-103. The target to beat:

| Model | Val PPL | Config |
|-------|---------|--------|
| M0 (baseline) | ~92 | Single-stream transformer, LR 6e-4, 2 epochs |
| Felix v2 (current) | 94.50 | 4-stream adaptive convergence, LR 6e-4, 3 epochs |

Primary goal: make v2 beat M0. Secondary: improve v2's absolute PPL.

## What you modify

- `felix_lm/v2/config.py` — config dataclass, hyperparameters
- `felix_lm/v2/model.py` — model architecture
- `felix_lm/v2/` — any v2 module files
- `scripts/train.py` — training loop, optimizer, LR schedule (only the v2 path)

## What you do NOT modify

- `felix_lm/config.py` or `felix_lm/model.py` (v1, frozen)
- `scripts/evaluate.py` (eval harness is ground truth)
- Test files (update tests only if your changes break the API)
- Data loading, tokenizer, evaluation metric

## Training command

```bash
python scripts/train.py --config felix_v2 --batch-size 8 --grad-accum 4 \
  --device cuda --lr 6e-4 --max-steps 2500 --no-wandb
```

`--max-steps 2500` gives ~10 minutes on our GPU. This is the fixed time budget.
The `--no-wandb` flag skips wandb logging for speed.

## Evaluation

After training completes, evaluate:

```bash
python scripts/evaluate.py --checkpoint checkpoints/felix_v2/best.pt \
  --split validation --device cuda --batch-size 8
```

The metric is **val PPL** (lower is better).

## The experiment loop

LOOP FOREVER:

1. Look at current state: read the latest code, check recent results
2. Form a hypothesis — what change might improve val PPL and why
3. Modify the code (config, model, training — whatever is needed)
4. Run tests: `pytest tests/ -q` (must pass before training)
5. git commit with descriptive message
6. Train: run the training command (redirect output to run.log)
7. Evaluate: get val PPL from the checkpoint
8. Log results to `autoresearch_results.tsv`
9. If val PPL improved: KEEP the commit, advance
10. If val PPL is same or worse: `git revert` back to previous best
11. Repeat from step 1

## Results tracking

Log to `autoresearch_results.tsv` (tab-separated):

```
commit	val_ppl	status	description
```

- commit: short hash (7 chars)
- val_ppl: validation perplexity (0.0 for crashes)
- status: keep, discard, or crash
- description: what was tried

## Constraints

- Parameter budget: ~10-12M total (comparable to M0's 11.17M)
- Must pass `pytest tests/ -q` before running
- Don't blow up VRAM beyond 16GB
- Simpler is better — don't add complexity for marginal gains
- Architecture principles from `.claude/rules/architecture-principles.md` are
  PROVEN findings. Don't re-test them unless you have strong reason.

## What to try

The v2 architecture has these knobs:
- num_streams, d_stream, d_embed, d_post
- num_layers, num_refine_layers
- num_heads, num_refine_heads
- ffn_mult
- CentralPost: cp_read_gated, cp_write_gated, cp_num_heads
- Adaptive merge: temperature init, bias init
- attention_type, attention_schedule (hetero attention)
- dropout, gradient_checkpointing

Unexplored v2-specific ideas:
- Hetero attention schedule (already implemented, never trained at 11M)
- Different merge temperature/bias initialization
- CentralPost head count, gating strategy
- Deeper refine stage (more post-merge layers)
- Different d_stream / d_post ratios
- Learning rate warmup / schedule tweaks for v2 specifically

## Key context

- v2 closes 71% of v1-to-M0 gap (94.50 vs 101.39 vs 91.81)
- v2's overhead: CentralPost + per-layer adaptive merge
- At 11M, embedding eats 58% of params. Only ~4.6M for compute.
- Hetero attention (linear early, full causal late) helped v1, untested on v2
- The 2500-step budget is early training — relative ordering matters, not absolute PPL

## NEVER STOP

Do not pause to ask the human. They may be asleep. Run experiments indefinitely
until manually interrupted. If stuck, think harder, try more radical changes.
