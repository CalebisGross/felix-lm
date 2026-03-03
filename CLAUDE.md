# Felix-LM Project Rules

## What This Is
Novel causal language model architecture: Multi-Stream Progressive Merging (MSPM).
Design thesis: `docs/felix_lm_design.pdf` (LaTeX source: `felix_lm_design.tex`).
Inspired by and evolved from the original Felix multi-agent framework concepts (helical funnel trajectory),
but this is a standalone project — do not reference or import from older Felix codebases.

## Environment
- Python 3.12.3 (NOT 3.14 — PyTorch ROCm incompatible)
- PyTorch with ROCm on AMD GPU (RX 7800 XT, 16GB VRAM)
- Venv: `source .venv/bin/activate`
- Tests: `pytest tests/ -q`
- Training: `python scripts/train.py --config <name> --batch-size 8 --grad-accum 4`

## Architecture
- 3 stages: Stage 0 (4 streams, dim=64) → Stage 1 (2 streams, dim=128) → Stage 2 (1 stream, dim=128)
- Gated merge between stages with learned sigmoid gates
- Depth-extended RoPE
- Deep supervision loss at merge boundaries (optional)
- Cross-stream agreement for confidence estimation

## Available Configs
| Config | Description | Key Flags |
|--------|-------------|-----------|
| `m2` | MSPM-HETERO (primary) | linear→sliding_window→full_causal, deep supervision ON |
| `m2_fullcausal` | All full causal attention | Tests multi-stream value independent of attention type |
| `m2_nosup` | No deep supervision | Tests supervision effect on final output |
| `m0` | Standard transformer baseline | 1 stream, 18 layers, full causal, parameter-matched |

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
4. Evaluate: `python scripts/evaluate.py --checkpoint checkpoints/<name>/best.pt --split test --device cuda --batch-size 8`
5. Record results in `docs/experiments.md`
6. Update LaTeX paper (`felix_lm_design.tex`) with findings

## Git Practices
- Run tests before committing
- Use descriptive commit messages
- Don't commit checkpoints, data, or wandb logs (already in .gitignore)
- The .tex source is tracked — update it with experimental results as they come in

## Key Findings So Far
- M2-fullcausal Stage 2 (final output) achieves 107.49 PPL, beating M0's 115.66
- Streams genuinely specialize (pairwise cosine ~0.00 between stream weights)
- Gates learn meaningful selectivity (~65% open after training)
- Linear attention in Stage 0 is the main bottleneck at small scale
- Multi-stream merging mechanism works; scaling is the next test
