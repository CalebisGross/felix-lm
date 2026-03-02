"""M2 MSPM-HETERO: Primary proof-of-concept configuration.

Multi-stream progressive merging with heterogeneous attention:
- Stage 0 (4 streams, linear attention): broad exploration
- Stage 1 (2 streams, sliding window): local convergence
- Stage 2 (1 stream, full causal): precise synthesis

~11M parameters, trained on WikiText-103 with T=512.
"""

from felix_lm.config import make_m2_config

config = make_m2_config()
