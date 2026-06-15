# less_core.py
#
# Shared math for the LESS samplers (Dream + LLaDA): the coarse top-K
# Jensen-Shannon drift metric and the masked-diffusion sampling helpers. Kept
# in one place so both model families use exactly the same implementation.
from __future__ import annotations

import torch

# Number of top-K entries used to approximate the per-position marginal when
# measuring step-to-step distributional drift.
DRIFT_TOPK = 8
_JS_EPS = 1e-12


def coarse_topk_js(topk_ids, topk_probs, prev_topk_ids, prev_topk_probs):
    """Jensen-Shannon divergence between two coarse top-K marginals.

    Each distribution is represented by its top-K (id, prob) pairs plus a single
    lumped "other" bucket holding the remaining mass. Returns a [K] tensor of
    per-position JS values in [0, ln 2].
    """
    device = topk_ids.device
    K, Kd = topk_ids.shape
    K2 = 2 * Kd
    ids_cat = torch.cat([topk_ids.clamp(min=0), prev_topk_ids.clamp(min=0)], dim=-1)
    zeros = torch.zeros_like(topk_probs)
    p_cat = torch.cat([topk_probs, zeros], dim=-1).float()
    q_cat = torch.cat([zeros, prev_topk_probs], dim=-1).float()
    sorted_ids, sort_idx = torch.sort(ids_cat, dim=-1)
    p_sorted = torch.gather(p_cat, -1, sort_idx)
    q_sorted = torch.gather(q_cat, -1, sort_idx)
    changes = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    group_idx = torch.zeros((K, K2), device=device, dtype=torch.long)
    group_idx[:, 1:] = changes.long().cumsum(dim=-1)
    p_g = torch.zeros((K, K2), device=device, dtype=torch.float32)
    q_g = torch.zeros((K, K2), device=device, dtype=torch.float32)
    p_g.scatter_add_(1, group_idx, p_sorted)
    q_g.scatter_add_(1, group_idx, q_sorted)
    p_other = (1.0 - topk_probs.float().sum(-1)).clamp_min(0.0)
    q_other = (1.0 - prev_topk_probs.float().sum(-1)).clamp_min(0.0)
    m_g = 0.5 * (p_g + q_g)
    m_o = 0.5 * (p_other + q_other)
    kl_pm = (p_g * (torch.log(p_g + _JS_EPS) - torch.log(m_g + _JS_EPS))).sum(-1)
    kl_qm = (q_g * (torch.log(q_g + _JS_EPS) - torch.log(m_g + _JS_EPS))).sum(-1)
    kl_pm = kl_pm + p_other * (torch.log(p_other + _JS_EPS) - torch.log(m_o + _JS_EPS))
    kl_qm = kl_qm + q_other * (torch.log(q_other + _JS_EPS) - torch.log(m_o + _JS_EPS))
    return (0.5 * (kl_pm + kl_qm)).clamp_min(0.0)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Exponential-race / Gumbel-Top-1 noise injection. temperature=0 -> greedy."""
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Uniformly schedule the number of token commits per step (per sample)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        r = int(remainder[i].item())
        if r > 0:
            num_transfer_tokens[i, :r] += 1
    return num_transfer_tokens
