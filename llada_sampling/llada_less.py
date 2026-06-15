"""
llada_sampling/llada_less.py

LESS for LLaDA's block-wise discrete diffusion. The acceptance rule is the
conjunction of three orthogonal signals (confidence, multi-step argmax
persistence, coarse top-K JS drift) applied per masked position per step,
with all-accepted commitment within the current block and a top-K-by-conf
safeguard to ensure monotone progress. Mirrors dream_sampling/less.py.
"""
from __future__ import annotations

import os
import json

import torch
import torch.nn.functional as F

from less_core import DRIFT_TOPK, add_gumbel_noise, coarse_topk_js, get_num_transfer_tokens


LLADA_MASK_TOKEN_ID = 126336


@torch.no_grad()
def stable_less_decode(
    model,
    tokenizer,
    input_ids_original,
    gen_length,
    steps,
    block_length,
    temperature=0.0,
    mask_id=None,
    conf_threshold=0.75,
    drift_threshold=0.04,
    persistence_len: int = 2,
    # ---- infra ----
    step_save_dir=None,
    example_idx: int = 0,
):
    if mask_id is None:
        mask_id = LLADA_MASK_TOKEN_ID
    if int(persistence_len) < 1:
        raise ValueError("persistence_len must be >= 1")

    x = torch.full(
        (1, input_ids_original.shape[1] + gen_length),
        mask_id, dtype=torch.long,
    ).to(model.device)
    x[:, :input_ids_original.shape[1]] = input_ids_original.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    Lp = input_ids_original.shape[1]
    N = x.shape[1]
    used_steps = 0

    P = int(persistence_len)
    prev_ids_buf = torch.full((1, N, P), -1, device=x.device, dtype=torch.long)
    prev_topk_ids = torch.full((1, N, DRIFT_TOPK), -1, device=x.device, dtype=torch.long)
    prev_topk_probs = torch.zeros((1, N, DRIFT_TOPK), device=x.device, dtype=torch.float32)

    all_step_outputs = []

    for num_block in range(num_blocks):
        block_start = Lp + num_block * block_length
        block_end = Lp + (num_block + 1) * block_length

        prev_ids_buf[:, block_start:block_end] = -1
        prev_topk_ids[:, block_start:block_end] = -1
        prev_topk_probs[:, block_start:block_end] = 0.0

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask
            if not mask_index.any():
                break

            drift_thr_eff = float(drift_threshold)
            conf_thr_eff = float(conf_threshold)

            logits = model(x).logits
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)

            probs = F.softmax(logits.float(), dim=-1)
            topk_probs_full, topk_ids_full = torch.topk(probs, k=DRIFT_TOPK, dim=-1)
            topk_probs_full = topk_probs_full.float()
            topk_ids_full = topk_ids_full.long()
            x0 = torch.argmax(probs, dim=-1)
            curr_conf = topk_probs_full[:, :, 0]

            masked_positions = mask_index[0].nonzero(as_tuple=False).squeeze(-1)
            if masked_positions.dim() == 0:
                masked_positions = masked_positions.unsqueeze(0)
            K = masked_positions.shape[0]
            if K == 0:
                break
            p_idx = masked_positions

            winners = x0[0, p_idx]
            conf = curr_conf[0, p_idx]
            tk_ids = topk_ids_full[0, p_idx]
            tk_probs = topk_probs_full[0, p_idx]

            # generalized persistence
            buf = prev_ids_buf[0, p_idx]  # [K, P]
            valid = (buf >= 0).all(dim=-1)
            matches = (buf == winners.unsqueeze(-1)).all(dim=-1)
            persist = valid & matches

            prev_tk_ids = prev_topk_ids[0, p_idx]
            prev_tk_probs = prev_topk_probs[0, p_idx]
            has_prev = prev_tk_probs.sum(-1) > 0.0
            drift = coarse_topk_js(tk_ids, tk_probs, prev_tk_ids, prev_tk_probs)
            drift = torch.where(has_prev, drift, torch.ones_like(drift))

            accept = (
                persist
                & (conf >= conf_thr_eff)
                & (drift <= drift_thr_eff)
            )

            # Update history
            if P > 1:
                prev_ids_buf[0, p_idx] = torch.cat([buf[:, 1:], winners.unsqueeze(-1)], dim=-1)
            else:
                prev_ids_buf[0, p_idx, 0] = winners
            prev_topk_ids[0, p_idx] = tk_ids
            prev_topk_probs[0, p_idx] = tk_probs

            transfer_index = torch.zeros_like(x[0], dtype=torch.bool)
            accepted_positions = p_idx[accept]

            if accepted_positions.numel() > 0:
                # Commit every position that passed the joint rule.
                transfer_index[accepted_positions] = True
            else:
                # Fallback: top-k by confidence keeps progress monotone.
                k = int(num_transfer_tokens[0, step].item())
                available = int(mask_index[0].sum().item())
                k = max(1, min(k, available)) if available > 0 else 0
                if k > 0:
                    confidence = torch.where(mask_index[0], curr_conf[0], -float('inf'))
                    _, selected = torch.topk(confidence, k=k)
                    transfer_index[selected] = True

            commit_positions = transfer_index.nonzero(as_tuple=False).squeeze(-1)
            if commit_positions.dim() == 0:
                commit_positions = commit_positions.unsqueeze(0)
            x[0, commit_positions] = x0[0, commit_positions]

            if step_save_dir:
                decoded_text = tokenizer.batch_decode(
                    x[:, Lp:], skip_special_tokens=True
                )[0]
                # Per-step argmax over generation positions (for flip-rate
                # analysis). Cast to int list for JSON.
                argmax_gen = x0[0, Lp:Lp + gen_length].tolist()
                all_step_outputs.append({
                    "step": used_steps + 1,
                    "block": num_block,
                    "block_step": step,
                    "decoded_text": decoded_text,
                    "argmax_ids": argmax_gen,
                    "accepted_tokens_num": int(accepted_positions.numel()),
                    "fallback": accepted_positions.numel() == 0,
                    "drift_thr_eff": float(drift_thr_eff),
                    "conf_thr_eff": float(conf_thr_eff),
                })

            used_steps += 1

    if step_save_dir:
        with open(os.path.join(step_save_dir, f"all_steps_{example_idx}.json"), "w") as f:
            json.dump(all_step_outputs, f, indent=2)

    return x, used_steps
