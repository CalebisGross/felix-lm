# Felix-LM

**Multi-Stream Progressive Merging for Causal Language Modeling**

[Paper (PDF)](docs/felix_lm_design.pdf) · [Experiment Log](docs/experiments.md)

---

## What is this?

Felix-LM is a novel transformer architecture where the input is processed by multiple independent streams in parallel, then progressively merged back into a single output. Instead of one big stack of layers, 4 narrow streams each develop their own "take" on the text, and learned gated merges combine them step by step.

Each stage uses the attention type suited to its job: cheap linear attention for broad early exploration, sliding-window for local refinement, and full causal for the final output. A depth-extended RoPE encodes both token position and network depth, which turns out to be essential for the multi-stage design.

## Architecture

The model processes input through 3 stages with decreasing stream counts (4 → 2 → 1):

- **Stage 0** (4 streams, dim=64) — Linear attention, stream specialization
- **Stage 1** (2 streams, dim=128) — Sliding-window attention, post-merge integration
- **Stage 2** (1 stream, dim=128) — Full causal attention, final refinement

Between stages, **gated merge** operations (learned sigmoid gates + linear projections) combine stream pairs. The streams specialize on their own without any explicit encouragement — pairwise cosine similarity between stream weights converges to ~0.00.

See the [design document](docs/felix_lm_design.pdf) for the full mathematical framework.

## Results

42 experiments at ~11M params on WikiText-103 (seq_len=512, effective batch size 32).

| Model | Config | LR | Epochs | Test PPL |
| :---- | :----- | :- | :----- | :------: |
| **M0 at optimal LR** | `m0` | **6e-4** | **2** | **91.81** |
| M0 (baseline) | `m0` | 3e-4 | 3 | 115.66 |
| MSPM at optimal LR | `m2_nosup` | 6e-4 | 2 | 101.39 |
| MSPM no supervision | `m2_nosup` | 3e-4 | 3 | 118.41 |
| MSPM (original) | `m2` | 3e-4 | 3 | 150.14 |

The biggest finding from 42 experiments: both architectures were undertrained at the default LR of 3e-4. At 6e-4, M0 drops from 115.66 to 91.81 and MSPM from 118.41 to 101.39. The single-stream baseline benefits more from the higher LR, widening the gap from 2.4% to 10.4%. At 11M scale, the "multi-stream tax" (embedding dominates the budget, fewer effective layers) outweighs the richer merged representations. The open question is whether this changes at larger scale where the tax shrinks.

### Key findings from 42 experiments

- **No deep supervision.** Removing auxiliary losses at merge boundaries was the single biggest improvement (+32 PPL). The model learns better when it only optimizes the final output.
- **The gated merge is the right merge.** Four alternative algorithms (geometric, Hadamard, competitive, noise-injected) all made things worse. Merges must be additive — the streams learn complementary representations that need to be combined, not selected between.
- **Streams specialize on their own.** Identical architecture per stream, but they learn fully orthogonal representations. Imposed structural diversity (different stream architectures) hurts. Emergent specialization beats designed specialization.
- **Helical RoPE is essential.** Standard RoPE (no depth component) is catastrophically worse for MSPM. The multi-stage architecture needs positional encoding that distinguishes network depth, not just token position.
- **4 streams, balanced 4/4/5 layer split.** Both fewer and more streams hurt. Every layer distribution we tested (frontloaded, backloaded, minimal middle stage) was worse than the balanced split.
- **Nothing stochastic at the merge boundary.** Stream dropout, random merge pairings, and noise injection all hurt. The merge learns a precise mapping that requires consistent inputs.
- **Both architectures want a higher learning rate.** The default 3e-4 was suboptimal for both. At 6e-4, M0 improves by 20.6% and MSPM by 14.4%. M0 benefits more, suggesting simpler gradient dynamics exploit higher LR more efficiently.

Full experiment log with methodology and analysis: [docs/experiments.md](docs/experiments.md)

## Getting Started

### Installation

```bash
git clone https://github.com/CalebisGross/felix-lm.git
cd felix-lm
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch  # or: pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
pip install -e ".[dev]"
```

### Tests

```bash
pytest tests/ -q
```

### Training

```bash
# Best overall config (M0 at optimal LR)
python scripts/train.py --config m0 --batch-size 8 --grad-accum 4 --device cuda --lr 6e-4

# Best MSPM config
python scripts/train.py --config m2_nosup --batch-size 8 --grad-accum 4 --device cuda --lr 6e-4

# Ablation flags
python scripts/train.py --config m2_nosup --rope-turns 0      # test standard RoPE
python scripts/train.py --config m2_nosup --lr 1e-3            # test different LR
python scripts/train.py --config m2_nosup --supervision-off-after 2000  # supervision curriculum
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/m2_nosup/best.pt --split test --device cuda --batch-size 8
```

Training logs are tracked with [Weights & Biases](https://wandb.ai) under the `felix-lm` project.

## Project Structure

```text
felix_lm/
├── model.py              # FelixLM top-level module (Algorithm 1)
├── config.py             # FelixConfig dataclass and experiment configs
├── attention.py          # LinearAttention, SlidingWindowAttention, FullCausalAttention
├── merge.py              # GatedMerge, CrossStreamAttention, MergeLayer
├── rope.py               # Depth-extended RoPE (position + layer depth encoding)
├── stage.py              # Processing stage (N streams x L transformer layers)
├── transformer_block.py  # Pre-norm transformer block with SwiGLU FFN
├── embedding.py          # Token embedding + per-stream learned projections
├── exit_heads.py         # Deep supervision loss and cross-stream agreement
├── diagnostics.py        # Gate statistics, stream similarity, agreement logging
└── utils.py              # Training utilities

scripts/
├── train.py              # Training loop with wandb logging and cosine LR
├── evaluate.py           # Evaluation on test/validation splits
└── count_params.py       # Parameter count verification

docs/
├── felix_lm_design.pdf   # Full design document and theoretical foundation
└── experiments.md         # Detailed experiment log (42 experiments)
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{gross2025felixlm,
  title   = {Felix-LM: Multi-Stream Progressive Merging for Causal Language Modeling},
  author  = {Gross, Caleb},
  year    = {2025},
  url     = {https://github.com/CalebisGross/felix-lm}
}
```

## License

MIT
