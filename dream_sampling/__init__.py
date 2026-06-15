# dream_sampling/__init__.py
from __future__ import annotations

"""
dream_sampling
--------------
LESS: stability-gated adaptive decoding for masked discrete diffusion LMs.

Supports multiple model families (Dream, LLaDA, etc.) via ModelAdapter.

Public API:
- diffusion_generate_less: LESS sampler (3-signal: conf + persist + drift)
- ModelAdapter, DreamAdapter, LLaDAAdapter: model abstraction layer
- get_model_adapter: factory for creating adapters by name
- SimpleGenerateOutput: minimal container with `.sequences`
- add_gumbel_noise, get_num_transfer_tokens: utilities
"""

from model_adapter import (
    ModelAdapter,
    DreamAdapter,
    LLaDAAdapter,
    get_model_adapter,
)
from .less import diffusion_generate_less
from .utils import (
    SimpleGenerateOutput,
    add_gumbel_noise,
    get_num_transfer_tokens,
)

__all__ = [
    "diffusion_generate_less",
    "ModelAdapter",
    "DreamAdapter",
    "LLaDAAdapter",
    "get_model_adapter",
    "SimpleGenerateOutput",
    "add_gumbel_noise",
    "get_num_transfer_tokens",
]
