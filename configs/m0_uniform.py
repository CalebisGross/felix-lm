"""M0 UNIFORM: Standard transformer baseline.

Single stream, full causal attention at every layer.
Parameter-matched to M2 for fair comparison.
"""

from felix_lm.config import make_m0_config

config = make_m0_config()
