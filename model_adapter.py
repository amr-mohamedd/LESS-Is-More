# coding=utf-8
# model_adapter.py
#
# Thin adapter layer that abstracts model-specific differences so samplers
# can work with Dream, LLaDA, and future masked diffusion LMs.
#
# Usage:
#   adapter = get_model_adapter("dream", model, mask_token_id=..., eos_token_id=...)
#   logits = adapter.forward(x, attention_mask)  # [B, N, V], already aligned
#
from __future__ import annotations

from typing import Optional

import torch


class ModelAdapter:
    """Base adapter — subclasses override forward() and metadata."""

    def __init__(
        self,
        model,
        *,
        mask_token_id: int,
        eos_token_id: Optional[int] = None,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Run model forward pass and return logits [B, N, V] aligned to positions.

        Args:
            x: [B, N] token ids (prompt + generation, masks at unresolved positions).
            attention_mask: [B, N] 1D attention mask, or None if no padding.
                Each adapter handles model-specific transformations internally.

        Returns:
            logits: [B, N, V] — logits[b, t, :] is the distribution for position t.
        """
        raise NotImplementedError

    @property
    def shift_logits(self) -> bool:
        """Whether this model uses next-token prediction (shift-right-by-1)."""
        return False


class DreamAdapter(ModelAdapter):
    """
    Dream models use next-token prediction with a shift-right-by-1 trick.

    Forward signature: model(x, attention_mask, tok_idx).logits
    - tok_idx: cumulative position indices computed from attention_mask
    - attention_mask: expanded to 4D [B, 1, N, N] if padding exists
    - Logits are shifted right by 1 so logits[t] predicts token at position t
    """

    @property
    def shift_logits(self) -> bool:
        return True

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        tok_idx = None
        attn_arg = "full"

        if attention_mask is not None and torch.any(attention_mask == 0):
            # Compute tok_idx from 1D mask
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # Expand to 4D causal mask [B, 1, N, N]
            attn_arg = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )

        logits = self.model(x, attn_arg, tok_idx).logits  # [B, N, V]
        # Dream shift-right-by-1: align so logits[t] predicts token at position t
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits


class LLaDAAdapter(ModelAdapter):
    """
    LLaDA is a masked LM — logits[t] directly predicts token at position t.

    Forward signature: model(x).logits
    No logit shifting required.
    """

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self.model(x).logits  # [B, N, V]
        return logits


# ── Registry ──────────────────────────────────────────────────────────────────

_ADAPTERS = {
    "dream": DreamAdapter,
    "llada": LLaDAAdapter,
}


def get_model_adapter(
    model_type: str,
    model,
    *,
    mask_token_id: int,
    eos_token_id: Optional[int] = None,
) -> ModelAdapter:
    """
    Create a model adapter by name.

    Args:
        model_type: "dream" or "llada"
        model: the underlying HF model
        mask_token_id: token ID used for masking
        eos_token_id: optional EOS token ID for early stopping
    """
    key = model_type.lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(_ADAPTERS)}")
    return _ADAPTERS[key](model, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
