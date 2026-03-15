# Felix-LM v3 Autoresearch: Hub-and-Spoke

Autonomous research loop for the v3 hub-and-spoke architecture.

## What changed from v2

82 experiments proved that duplicating transformer blocks across parallel streams
is a parameter tax that never pays for itself. The fundamental mistake: streams
were the expensive primary pathway, CentralPost was a cheap sidecar. v3 inverts this.

**The hub (backbone) IS the CentralPost** — a standard transformer that gets all the
parameter budget, depth, and attention capacity. **Spokes are the agents** — lightweight
low-rank probes that branch off the hub, process through diverse transforms, and gate
their findings back. Each spoke costs almost nothing (~2.8% total overhead for 4 spokes).

## Current standings

| Model | Val PPL | Config |
|-------|---------|--------|
| Plain transformer (v2_baseline) | 44.81 | 20 layers, d128, optimal HPs |
| v3 (hub-and-spoke) | TBD | 20 layers + 4 spokes rank=16, progressive gates |
| Felix v2 (v2_stream2) | 49.03 | 2 stream + 18 refine, d64 |

Primary goal: does the hub-and-spoke overlay (spokes) improve on the plain transformer?

## Architecture

```
For each layer:
  1. h = h + Attention(RMSNorm(h))         # hub (standard transformer)
  2. h = h + FFN(RMSNorm(h))               # hub (standard transformer)
  3. h_norm = RMSNorm(h)                    # spoke preprocessing
     For each spoke s:
       view_s = SiLU(h_norm @ W_down_s)    # project to low-rank [d -> r]
       update_s = view_s @ W_up_s           # project back [r -> d]
     h = h + gate * mean(updates)           # gated residual
```

Spokes provide diversity. Gate schedule provides progressive convergence.
Agreement (cross-spoke cosine similarity) provides confidence.

## What you modify

- `felix_lm/v3/config.py` — v3 config dataclass
- `felix_lm/v3/spokes.py` — spoke layer implementation
- `felix_lm/v3/model.py` — v3 model
- `scripts/train.py` — only the v3 config dispatch and logging

## What you do NOT modify

- `felix_lm/config.py` or `felix_lm/model.py` (v1, frozen)
- `felix_lm/v2/` (v2, frozen)
- `scripts/evaluate.py` (eval harness is ground truth)
- Data loading, tokenizer, evaluation metric

## Training command

```bash
python scripts/train.py --config v3_base --batch-size 8 --grad-accum 8 \
  --device cuda --lr 2e-2 --max-steps 2500 --weight-decay 0.1 \
  --beta2 0.99 --warmup-steps 500 --compile --no-wandb
```

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

## Available v3 configs

| Config | Spokes | Rank | Gate Schedule | Notes |
|--------|--------|------|---------------|-------|
| `v3_base` | 4 | 16 | progressive | Default v3 |
| `v3_none` | 0 | — | none | Plain transformer (should match v2_baseline) |
| `v3_uniform` | 4 | 16 | uniform | All gates start at 0.5 |
| `v3_r8` | 4 | 8 | progressive | Less capacity per spoke |
| `v3_r32` | 4 | 32 | progressive | More capacity (~5% overhead) |
| `v3_2spoke` | 2 | 16 | progressive | Less diversity |
| `v3_8spoke` | 8 | 16 | progressive | More diversity, tinier each |

## Locked-in hyperparameters (from v2 sweep)

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
| Helical RoPE turns | 2 | 0 |
| torch.compile | yes | — |

## Experiment plan

### Phase 1: Does it work at all? (11M)
1. `v3_base` — 4 spokes, rank=16, progressive gates -> compare to v2_baseline (44.81)
2. `v3_none` — 0 spokes (pure transformer) -> should match v2_baseline
3. `v3_uniform` — 4 spokes, rank=16, uniform gates -> ablate the schedule

### Phase 2: Spoke tuning (11M)
4. `v3_r8` — rank=8 (less capacity per spoke)
5. `v3_r32` — rank=32 (more capacity, ~5% overhead)
6. `v3_2spoke` — 2 spokes instead of 4
7. `v3_8spoke` — 8 spokes (very diverse, tiny each)

### Phase 3: Architecture variations (if Phase 1-2 show promise)
- Spoke attention (single-head cross-attention instead of linear)
- Shared W_down across spokes (differentiate only via W_up)
- Spoke dropout (random subset of spokes per layer)
- Deeper spoke bottleneck (2-layer MLP instead of linear)

### Phase 4: Scale test (100M)
- Same design at d=512, compare against v2_100m_base

## Constraints

- Parameter budget: ~10-12M total at 11M scale
- Must pass `pytest tests/ -q` before running
- Don't blow up VRAM beyond 16GB
- Spokes must remain cheap (<5% overhead)

## NEVER STOP

Do not pause to ask the human. They may be asleep. Run experiments indefinitely
until manually interrupted. If stuck, think harder, try more radical changes.
