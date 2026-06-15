"""
dream_sampling/less.py

LESS: stability-gated adaptive decoder for masked discrete diffusion language
models. The acceptance rule is the conjunction of three orthogonal signals:
confidence, multi-step argmax persistence, and coarse top-K Jensen-Shannon
distributional drift. Frontier-first commitment with bounded look-ahead and
asymmetric end-of-sequence tightening complete the rule.

The rule is governed by three thresholds:
  conf_threshold     top-1 probability required to commit. Default=0.75.
  drift_threshold    max coarse top-K JS divergence allowed. Default=0.04.
  persistence_len    n consecutive matching argmax required. Default=2.

collect_step_trace records a per-step audit (fallback bits, accepted counts,
per-position argmax ids) for offline analysis. It forces a host-device sync
each step, so it defaults to False and should stay off for timing/inference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from model_adapter import ModelAdapter, get_model_adapter
from less_core import DRIFT_TOPK, coarse_topk_js


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = int(min(top_k, logits.size(-1)))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def _sample_tokens(
    logits: torch.Tensor, *, temperature: float, top_p: Optional[float], top_k: Optional[int],
) -> torch.Tensor:
    logits = logits / float(temperature)
    if top_p is not None and top_p < 1.0:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    try:
        return torch.distributions.Categorical(probs=probs).sample()
    except Exception:
        return probs.argmax(dim=-1)


_FRONTIER_PATIENCE = 2
_FRONTIER_PATIENCE_WINDOW = 16


@torch.no_grad()
def diffusion_generate_less(
    model,
    input_ids: torch.LongTensor,
    *,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: Optional[int] = None,
    steps: int = 128,
    conf_threshold: float = 0.75,
    drift_threshold: float = 0.04,
    persistence_len: int = 2,
    # ---- standard decoding ----
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    # ---- infrastructure ----
    mask_token_id: Optional[int] = None,
    min_new_tokens_before_eos: int = 0,
    enable_eos_early_stop: bool = True,
    # ---- EOS asymmetric tightening (Sec. 4.3) ----
    eos_conf_threshold: float = 0.9,    # alpha_0
    eos_conf_anneal: float = 0.1,       # beta
    eos_conf_floor: float = 0.55,       # alpha_min
    eos_js_mult: float = 0.5,           # gamma
    return_stats: bool = True,
    collect_step_trace: bool = False,
    model_type: str = "dream",
    adapter: Optional[ModelAdapter] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    if mask_token_id is None:
        raise ValueError("mask_token_id must be provided.")
    if max_new_tokens is None or int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    if int(persistence_len) < 1:
        raise ValueError("persistence_len must be >= 1")

    if adapter is None:
        adapter = get_model_adapter(model_type, model, mask_token_id=int(mask_token_id))

    device = getattr(model, "device", input_ids.device)
    x_in = input_ids.to(device)
    B, Lp = x_in.shape
    max_new_tokens = int(max_new_tokens)
    max_length = Lp + max_new_tokens
    total_steps = int(max(1, steps))

    x = F.pad(x_in, (0, max_length - Lp), value=int(mask_token_id))

    if attention_mask is not None and torch.any(attention_mask == 0):
        attention_mask = attention_mask.to(device)
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1)
    else:
        attention_mask = None

    eos_token_id = adapter.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        eos_token_id = int(eos_token_id) if eos_token_id is not None else None
    if eos_token_id is not None and eos_token_id == int(mask_token_id):
        eos_token_id = None

    # rolling buffer of previous persistence_len argmax ids (oldest at index 0)
    P = int(persistence_len)
    prev_ids_buf = torch.full((B, max_length, P), -1, device=device, dtype=torch.long)
    prev_topk_ids = torch.full((B, max_length, DRIFT_TOPK), -1, device=device, dtype=torch.long)
    prev_topk_probs = torch.zeros((B, max_length, DRIFT_TOPK), device=device, dtype=torch.float32)

    pos_grid = torch.arange(max_length, device=device).unsqueeze(0).expand(B, -1)
    do_sample = float(temperature) > 0.0

    steps_taken_val = total_steps
    ban_id = int(mask_token_id)
    frontier_wait = torch.zeros((B,), device=device, dtype=torch.int32)
    # Per-step audit: fallback fires when the joint rule accepts no position for that batch.
    per_step_fallback: List[List[bool]] = [[] for _ in range(B)]
    per_step_accepted_n: List[List[int]] = [[] for _ in range(B)]
    # Per-step argmax IDs over the generation region (for flip-rate analysis).
    # Committed positions hold their committed token; masked positions use the
    # current step's per-position argmax over the full vocabulary.
    per_step_argmax_ids: List[List[List[int]]] = [[] for _ in range(B)]

    for i in range(total_steps):
        mask_index = (x == ban_id)
        if not mask_index.any():
            steps_taken_val = i
            break

        prog = float((i + 1) / total_steps)
        drift_thr_eff = float(drift_threshold)
        conf_thr_eff = float(conf_threshold)

        valid_mask = mask_index & (pos_grid >= Lp)
        masked_pos = torch.where(valid_mask, pos_grid, torch.full_like(pos_grid, max_length))
        frontier = masked_pos.min(dim=-1).values

        logits = adapter.forward(x, attention_mask)

        mask_positions = mask_index.nonzero(as_tuple=False)
        b_idx = mask_positions[:, 0]
        p_idx = mask_positions[:, 1]
        cand_mask = (p_idx >= Lp)

        mask_logits = logits[mask_index]
        mask_logits[:, ban_id] = torch.finfo(mask_logits.dtype).min
        if eos_token_id is not None and int(min_new_tokens_before_eos) > 0:
            eos_too_early = p_idx < (Lp + int(min_new_tokens_before_eos))
            if eos_too_early.any():
                mask_logits[eos_too_early, eos_token_id] = torch.finfo(mask_logits.dtype).min

        probs = torch.softmax(mask_logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=DRIFT_TOPK, dim=-1)
        topk_probs = topk_probs.float()
        topk_ids = topk_ids.long()
        winners = topk_ids[:, 0]
        conf = topk_probs[:, 0]

        # generalized persistence — all P previous entries must match current winner
        buf = prev_ids_buf[b_idx, p_idx]  # [K, P]
        valid = (buf >= 0).all(dim=-1)
        matches = (buf == winners.unsqueeze(-1)).all(dim=-1)
        persist = valid & matches
        # mode-switch guard is implied by the all-match requirement

        prev_ids_k = prev_topk_ids[b_idx, p_idx]
        prev_probs_k = prev_topk_probs[b_idx, p_idx]
        has_prev = prev_probs_k.sum(-1) > 0.0
        drift = coarse_topk_js(topk_ids, topk_probs, prev_ids_k, prev_probs_k)
        drift = torch.where(has_prev, drift, torch.ones_like(drift))

        is_eos = torch.zeros_like(winners, dtype=torch.bool)
        if eos_token_id is not None:
            is_eos = (winners == int(eos_token_id))
            eos_conf_eff = max(float(eos_conf_floor), float(eos_conf_threshold) - float(eos_conf_anneal) * prog)
            eos_js_thr = float(drift_thr_eff) * float(eos_js_mult)

        accept = (
            cand_mask
            & persist
            & (conf >= conf_thr_eff)
            & (drift <= drift_thr_eff)
        )
        if eos_token_id is not None:
            eos_conf_ok = conf >= eos_conf_eff
            eos_drift_ok = drift <= eos_js_thr
            accept = accept & (~is_eos | (eos_conf_ok & eos_drift_ok))

        # Optional per-step audit trace (off by default — each entry forces a
        # host-device sync, so it is only collected when explicitly requested,
        # e.g. for flip-rate analysis). Normal inference skips this entirely.
        if collect_step_trace:
            # Per-batch fallback bit: True iff joint rule accepted no position.
            for _b in range(B):
                n_acc = int(((b_idx == _b) & cand_mask & accept).sum().item())
                per_step_fallback[_b].append(n_acc == 0)
                per_step_accepted_n[_b].append(n_acc)
            # Per-step argmax over the generation region. Masked positions use
            # the model's current-step argmax; committed positions hold their
            # committed value.
            full_argmax = logits[:, Lp:max_length].argmax(dim=-1)
            committed = ~mask_index[:, Lp:max_length]
            full_argmax = torch.where(committed, x[:, Lp:max_length], full_argmax)
            for _b in range(B):
                per_step_argmax_ids[_b].append(full_argmax[_b].tolist())

        # Update history: shift buffer left, append current winner
        if P > 1:
            prev_ids_buf[b_idx, p_idx] = torch.cat([buf[:, 1:], winners.unsqueeze(-1)], dim=-1)
        else:
            prev_ids_buf[b_idx, p_idx, 0] = winners
        prev_topk_ids[b_idx, p_idx] = topk_ids
        prev_topk_probs[b_idx, p_idx] = topk_probs

        commit_b: List[int] = []
        commit_p: List[int] = []
        commit_tok: List[int] = []

        sampled_ids: Optional[torch.Tensor] = None
        if do_sample:
            sampled_ids = _sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)

        def _default_tok(ki: int) -> int:
            if do_sample and sampled_ids is not None:
                return int(sampled_ids[ki].item())
            return int(winners[ki].item())

        # Frontier-first prefix-run commit
        for b in range(B):
            fpos = int(frontier[b].item())
            if fpos >= max_length:
                continue
            idx_b = torch.where((b_idx == b) & cand_mask)[0]
            if idx_b.numel() == 0:
                continue
            at_frontier = (b_idx == b) & (p_idx == fpos) & cand_mask
            idx_f_arr = torch.where(at_frontier)[0]
            if idx_f_arr.numel() == 0:
                best = idx_b[conf[idx_b].argmax()]
                commit_b.append(b)
                commit_p.append(int(p_idx[best].item()))
                commit_tok.append(int(winners[best].item()))
                frontier_wait[b] = 0
                continue
            idx_f_g = int(idx_f_arr[0].item())
            if not bool(accept[idx_f_g].item()):
                if int(frontier_wait[b].item()) < _FRONTIER_PATIENCE:
                    alt = idx_b[accept[idx_b]]
                    if alt.numel() > 0:
                        alt = alt[
                            (p_idx[alt] > fpos)
                            & (p_idx[alt] <= (fpos + _FRONTIER_PATIENCE_WINDOW))
                        ]
                    if alt.numel() > 0:
                        best_alt = alt[conf[alt].argmax()]
                        commit_b.append(b)
                        commit_p.append(int(p_idx[best_alt].item()))
                        commit_tok.append(_default_tok(int(best_alt.item())))
                        frontier_wait[b] += 1
                        continue
                commit_b.append(b)
                commit_p.append(fpos)
                commit_tok.append(int(winners[idx_f_g].item()))
                frontier_wait[b] = 0
                continue
            frontier_wait[b] = 0
            pos_to_kidx: Dict[int, int] = {int(p_idx[j].item()): int(j) for j in idx_b.tolist()}
            n_before = len(commit_b)
            pos = fpos
            while pos < max_length:
                if int(x[b, pos].item()) != ban_id:
                    pos += 1
                    continue
                ki = pos_to_kidx.get(pos)
                if ki is None:
                    break
                if not bool(accept[ki].item()):
                    break
                commit_b.append(b)
                commit_p.append(pos)
                commit_tok.append(_default_tok(ki))
                pos += 1
            if len(commit_b) == n_before:
                commit_b.append(b)
                commit_p.append(fpos)
                commit_tok.append(int(winners[idx_f_g].item()))

        if len(commit_b) == 0:
            if cand_mask.any():
                idx_c = torch.where(cand_mask)[0]
                best = idx_c[conf[idx_c].argmax()]
            else:
                best = conf.argmax()
            commit_b.append(int(b_idx[best].item()))
            commit_p.append(int(p_idx[best].item()))
            commit_tok.append(int(winners[best].item()))

        cb_t = torch.tensor(commit_b, device=device, dtype=torch.long)
        cp_t = torch.tensor(commit_p, device=device, dtype=torch.long)
        ctok_t = torch.tensor(commit_tok, device=device, dtype=torch.long)
        x[cb_t, cp_t] = ctok_t
        steps_taken_val = i + 1

        if enable_eos_early_stop and eos_token_id is not None:
            done_all = True
            for b in range(B):
                masks_b = x[b] == ban_id
                if not masks_b.any():
                    continue
                eos_pos = (x[b] == eos_token_id).nonzero(as_tuple=False)
                if eos_pos.numel() == 0:
                    done_all = False
                    break
                first_eos = int(eos_pos.min().item())
                if int(min_new_tokens_before_eos) > 0 and first_eos < (Lp + int(min_new_tokens_before_eos)):
                    done_all = False
                    break
                if masks_b[:first_eos].any():
                    done_all = False
                    break
            if done_all:
                break

    # Safety net: under the intended contract (steps >= max_new_tokens) the
    # frontier fallback commits >= 1 position per step, so the generation
    # region is always fully resolved. If the step budget is exhausted with
    # masks still present (e.g. steps < max_new_tokens and the model never
    # stabilizes), fill the stragglers deterministically so the returned
    # sequence never contains mask tokens.
    gen_region = x[:, Lp:max_length]
    leftover = gen_region == ban_id
    if bool(leftover.any().item()):
        fill_logits = adapter.forward(x, attention_mask)[:, Lp:max_length]
        fill_logits[..., ban_id] = torch.finfo(fill_logits.dtype).min
        fill_argmax = fill_logits.argmax(dim=-1)
        x[:, Lp:max_length] = torch.where(leftover, fill_argmax, gen_region)

    committed_early = bool(steps_taken_val < total_steps)
    stats: Dict[str, Any] = {
        "steps_taken": int(steps_taken_val),
        "Tmax": int(total_steps),
        "committed_early": committed_early,
        "progress": float(steps_taken_val / total_steps),
        "model_type": model_type,
        "conf_threshold": float(conf_threshold),
        "drift_threshold": float(drift_threshold),
        "persistence_len": int(persistence_len),
        "drift_topk": DRIFT_TOPK,
        "temperature": float(temperature),
        "top_p": float(top_p) if top_p is not None else None,
        "top_k": int(top_k) if top_k is not None else None,
        "eos_token_id": eos_token_id,
        "min_new_tokens_before_eos": int(min_new_tokens_before_eos),
        "enable_eos_early_stop": bool(enable_eos_early_stop),
    }
    stats_list: List[Dict[str, Any]] = [dict(stats) for _ in range(B)]
    for _b in range(B):
        stats_list[_b]["per_step_fallback"] = list(per_step_fallback[_b])
        stats_list[_b]["per_step_accepted_n"] = list(per_step_accepted_n[_b])
        stats_list[_b]["per_step_argmax_ids"] = [list(row) for row in per_step_argmax_ids[_b]]
    return (x, stats_list) if return_stats else x
