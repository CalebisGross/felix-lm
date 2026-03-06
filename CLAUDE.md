# Felix-LM Project Rules

## What This Is
Novel causal language model architecture: Multi-Stream Progressive Merging (MSPM).
Design thesis: `docs/felix_lm_design.pdf` (LaTeX source: `felix_lm_design.tex`).
Inspired by and evolved from the original Felix multi-agent framework concepts (helical funnel trajectory),
but this is a standalone project — do not reference or import from older Felix codebases.

## Environment
Two training machines are available. Always use `--device` appropriate to the machine:

**Linux (primary):**
- Python 3.12.3 (NOT 3.14 — PyTorch ROCm incompatible)
- PyTorch with ROCm on AMD GPU (RX 7800 XT, 16GB VRAM)
- Training: `python scripts/train.py --config <name> --batch-size 8 --grad-accum 4 --device cuda`

**Mac Mini M4 (secondary):**
- Python 3.12+, PyTorch with MPS backend (16GB unified memory)
- Training: `python scripts/train.py --config <name> --batch-size 8 --grad-accum 4 --device mps`

**Both machines:**
- Venv: `source .venv/bin/activate`
- Tests: `pytest tests/ -q`
- Always `git pull` before starting a run to ensure latest code
- wandb logs to the same `felix-lm` project from both machines

## Architecture
- 3 stages: Stage 0 (4 streams, dim=64) → Stage 1 (2 streams, dim=128) → Stage 2 (1 stream, dim=128)
- Gated merge between stages with learned sigmoid gates
- Depth-extended RoPE
- Deep supervision loss at merge boundaries (optional)
- Cross-stream agreement for confidence estimation

## Available Configs
| Config | Streams | Attention | Deep Sup | Notes |
|--------|---------|-----------|----------|-------|
| `m0` | 1 (baseline) | full causal | N/A | Standard transformer, 18 layers |
| `m2` | 4→2→1 | hetero | ON | Original MSPM-HETERO |
| `m2_fullcausal` | 4→2→1 | full causal | ON | Isolates attention type |
| `m2_nosup` | 4→2→1 | hetero | OFF | **Best config (118.41 PPL)** |
| `m2_best` | 4→2→1 | full causal | OFF | Fullcausal + nosup combined |
| `m2_nosup_backloaded` | 4→2→1 | hetero | OFF | 2/4/7 layer split |
| `m2_2stream` | 2→1 | linear→full_causal | OFF | Wider streams, single merge |
| `m2_8stream` | 8→4→2→1 | hetero | OFF | Narrower streams, more merges |
| `m2_nosup_gate0` | 4→2→1 | hetero | OFF | Gate bias init=0.0 |

## Code Conventions
- Config dataclasses in `felix_lm/config.py` — every new experiment variant gets a `make_<name>_config()` function
- All experiments logged to wandb project `felix-lm`
- Checkpoints save to `checkpoints/<config_name>/`
- Batch size 8 with gradient accumulation 4 (effective batch size 32) for 16GB VRAM
- Run `pytest tests/ -q` before committing — all tests must pass

## Experiment Workflow
1. Add config function to `felix_lm/config.py`
2. Add config name to the choices in `scripts/train.py`
3. Train: `python scripts/train.py --config <name> --batch-size 8 --grad-accum 4`
4. Evaluate: `python scripts/evaluate.py --checkpoint checkpoints/<name>/best.pt --split test --device <cuda|mps> --batch-size 8`
5. Record results in `docs/experiments.md`
6. Update LaTeX paper (`felix_lm_design.tex`) with findings

## Git Practices
- Run tests before committing
- Use descriptive commit messages
- Don't commit checkpoints, data, or wandb logs (already in .gitignore)
- The .tex source is tracked — update it with experimental results as they come in

## Experiment Logging Rules (MANDATORY)
This is a serious research project. Every experiment must be rigorously documented.

- **Before running:** State the hypothesis and what variable is being tested
- **After every run:** Immediately evaluate on test set and record results in `docs/experiments.md`
- **Every entry must include:** config name, param count, date, hypothesis, test PPL, and a "key finding" interpreting the result
- **Update the summary table** in `docs/experiments.md` after every experiment — no exceptions
- **Negative results are results.** Document what didn't work and why. Failed experiments are as valuable as successes for guiding next steps
- **Never skip evaluation.** Even for 2-epoch directional tests, run the eval and log it
- **Compare apples to apples.** Always note epoch count, and compare against M2-nosup at the same training stage when doing shortened runs
- **Keep `docs/research_directions.md` current** with updated priorities based on what we've learned

## Key Findings So Far

- Deep supervision is the biggest bottleneck — removing it drops PPL by 32 points
- M2-nosup (118.41) is best multi-stream config, within 2.4% of M0 baseline (115.66)
- Hetero attention (linear→sliding_window→full_causal) is better than all-fullcausal when supervision is off
- Linear attention in Stage 0 acts as beneficial regularizer without supervision
- Backloading layers (2/4/7) hurts — early stages need depth for stream specialization
- Fullcausal + nosup improvements are NOT additive (slightly antagonistic)
- Streams genuinely specialize (pairwise cosine ~0.00 between stream weights)
- Gates learn meaningful selectivity (~65% open after training)
- M2-fullcausal Stage 2 output (107.49) beats M0 (115.66), but weighted loss doesn't
