# dream_sampling/utils.py
from __future__ import annotations
import torch

# The masked-diffusion math helpers live in the shared top-level less_core
# module; re-exported here so dream_sampling's public API is unchanged.
from less_core import add_gumbel_noise, get_num_transfer_tokens

__all__ = ["SimpleGenerateOutput", "add_gumbel_noise", "get_num_transfer_tokens"]


class SimpleGenerateOutput:
    """Minimal HF-like container so callers can use `.sequences`."""
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences
