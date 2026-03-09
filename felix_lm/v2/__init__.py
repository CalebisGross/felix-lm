"""Felix-LM v2: Adaptive Convergence Architecture.

Translates the core Felix multi-agent framework mechanisms into a neural LM:
- CentralPost: shared communication hub between streams
- Adaptive merge: agreement-driven continuous convergence
- Input-adaptive compute: easy tokens converge fast, hard tokens stay divergent
"""

from felix_lm.v2.config import FelixV2Config
from felix_lm.v2.model import FelixLMv2

__all__ = ["FelixV2Config", "FelixLMv2"]
