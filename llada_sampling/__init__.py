"""
llada_sampling
--------------
LESS for LLaDA's block-wise discrete diffusion.

Public API:
- LLADA_MASK_TOKEN_ID: LLaDA's mask token id (126336)
- stable_less_decode: LESS 3-signal adaptive decoding for LLaDA block diffusion
"""

from .llada_less import LLADA_MASK_TOKEN_ID, stable_less_decode

__all__ = [
    "LLADA_MASK_TOKEN_ID",
    "stable_less_decode",
]
