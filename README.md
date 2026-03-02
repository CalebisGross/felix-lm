# Felix-LM

**Multi-Stream Progressive Merging for Causal Language Modeling**

A novel causal language model architecture that processes input through multiple independent streams which progressively merge into a single output stream, tracing a helical funnel trajectory in representation space.

## Architecture

Felix-LM introduces Multi-Stream Progressive Merging (MSPM):

- **K stages** with decreasing stream counts: S_0 > S_1 > ... > S_{K-1} = 1
- **Heterogeneous attention**: linear (early) → sliding window (middle) → full causal (final)
- **Gated merge operations** between stages that combine stream pairs
- **Depth-extended RoPE** encoding both token position and network depth
- **Confidence-gated early exit** via cross-stream agreement

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
pip install -e ".[dev]"
```

## Usage

```bash
# Run tests
pytest tests/

# Train
python scripts/train.py
```

See [docs/felix_lm_design.pdf](docs/felix_lm_design.pdf) for the full design document.
