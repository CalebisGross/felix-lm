"""Felix-LM v3: Hub-and-Spoke Architecture.

The backbone (hub) is a standard transformer — the CentralPost.
Spokes are lightweight low-rank probes that read the hub, process
through diverse learned transforms, and gate their findings back.
"""

from felix_lm.v3.config import FelixV3Config
from felix_lm.v3.model import FelixLMv3

__all__ = ["FelixV3Config", "FelixLMv3"]
