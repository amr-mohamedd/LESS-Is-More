#!/usr/bin/env python3
"""LLaDA lm-evaluation-harness wrapper with LESS / origin generation.

Registers model type ``llada_less`` for use with ``lm_eval``.  Supports both
generative (``generate_until``) and likelihood (``loglikelihood``) tasks.

Usage (standalone):
    accelerate launch eval_llada_harness.py \
        --model llada_less \
        --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,alg=less,..." \
        --tasks hellaswag_generative \
        --num_fewshot 5 --batch_size 1 --output_path ./results
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# ---------------------------------------------------------------------------
# Ensure repo root is importable (needed for dream_sampling / llada_sampling)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (_REPO_ROOT, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llada_sampling.llada_less import (
    stable_less_decode,
    LLADA_MASK_TOKEN_ID,
    add_gumbel_noise,
    get_num_transfer_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().strip("'\"").lower()
    return s in ("1", "true", "yes", "y", "t")


# ---------------------------------------------------------------------------
# Default (origin) block-wise generation for LLaDA
# ---------------------------------------------------------------------------
@torch.no_grad()
def _origin_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    gen_length: int,
    steps: int,
    block_length: int,
    temperature: float = 0.0,
    mask_id: int = LLADA_MASK_TOKEN_ID,
) -> Tuple[torch.Tensor, int]:
    """Vanilla LLaDA block-wise generation (no early exit). Returns (x, steps_used)."""
    x = torch.full(
        (1, input_ids.shape[1] + gen_length),
        mask_id, dtype=torch.long, device=model.device,
    )
    x[:, :input_ids.shape[1]] = input_ids.clone()

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    Lp = input_ids.shape[1]
    used = 0

    for blk in range(num_blocks):
        bs = Lp + blk * block_length
        be = Lp + (blk + 1) * block_length
        block_mask = (x[:, bs:be] == mask_id)
        ntt = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            bm = torch.zeros_like(mask_index)
            bm[:, bs:be] = True
            mask_index = mask_index & bm
            if not mask_index.any():
                break

            logits = model(x).logits
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits, dim=-1)

            probs = F.softmax(logits.float(), dim=-1)
            conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(mask_index, conf, -torch.inf)

            k = int(ntt[0, step].item())
            avail = int(mask_index.sum().item())
            k = max(1, min(k, avail))
            _, sel = torch.topk(conf.view(-1), k=k)
            transfer = torch.zeros_like(x.view(-1), dtype=torch.bool)
            transfer[sel] = True
            transfer = transfer.view(x.shape)
            x = torch.where(transfer, x0, x)
            used += 1

    return x, used


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
@register_model("llada_less")
class LLaDALESSHarness(LM):
    """lm-eval wrapper for LLaDA with LESS / origin generation."""

    def __init__(
        self,
        model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        mask_id: int = LLADA_MASK_TOKEN_ID,
        max_length: int = 4096,
        batch_size: int = 1,
        mc_num: int = 128,
        is_check_greedy: bool = False,
        steps: int = 256,
        gen_length: int = 256,
        block_length: int = 64,
        temperature: float = 0.0,
        device: str = "cuda",
        # Algorithm selection
        alg: str = "less",
        # LESS knobs
        less_conf: float = 0.75,
        less_drift: float = 0.04,
        less_persistence: int = 2,
        # Chat template
        apply_chat_template: bool = False,
        # Metadata
        meta_dir: str = "eval_meta",
        **kwargs,
    ):
        super().__init__()
        # ---- Device / accelerate ----
        # NOTE: pass an extended NCCL timeout via InitProcessGroupKwargs to avoid
        # 600s default collective-op timeouts on long-running generative tasks
        # (e.g. hellaswag_generative with LESS sampling, where one rank
        # can run minutes longer than the others before hitting the per-task
        # wait_for_everyone() barrier at evaluator.py:592, which is a 1-element
        # NCCL ALLREDUCE under the hood).
        try:
            from datetime import timedelta
            import accelerate
            from accelerate import InitProcessGroupKwargs
            _ipg = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            acc = accelerate.Accelerator(kwargs_handlers=[_ipg])
            self._rank = acc.local_process_index
            self._world_size = acc.num_processes
            self.accelerator = acc if acc.num_processes > 1 else None
        except Exception:
            self._rank = 0
            self._world_size = 1
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs["device_map"] = {"": f"{self.accelerator.device}"}
        else:
            model_kwargs["device_map"] = device

        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, **model_kwargs,
        ).eval()
        self.device = next(self.model.parameters()).device

        if self.accelerator is not None:
            self.device = torch.device(f"{self.accelerator.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id = int(mask_id)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.mc_num = int(mc_num)
        self.is_check_greedy = _as_bool(is_check_greedy)

        # Generation knobs
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.temperature = float(temperature)

        # Algorithm
        self.alg = str(alg).strip().lower()
        self.less_conf = float(less_conf)
        self.less_drift = float(less_drift)
        self.less_persistence = int(less_persistence)

        self.always_chat_template = _as_bool(apply_chat_template)

        # Metadata logging
        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._meta_buffer: list = []
        self._run_header = {
            "model_path": model_path,
            "alg": self.alg,
            "steps": self.steps,
            "gen_length": self.gen_length,
            "block_length": self.block_length,
            "less_conf": self.less_conf,
            "less_drift": self.less_drift,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ---- LM interface ----
    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    # ---- Generation ----
    def _generate_one(self, prompt_ids: torch.Tensor) -> Tuple[str, int, float]:
        """Generate from prompt_ids [1, Lp]. Returns (text, steps_used, seconds)."""
        prompt_ids = prompt_ids.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()

        if self.alg == "less":
            x, used = stable_less_decode(
                self.model, self.tokenizer, prompt_ids,
                gen_length=self.gen_length,
                steps=self.steps,
                block_length=self.block_length,
                temperature=self.temperature,
                mask_id=self.mask_id,
                conf_threshold=self.less_conf,
                drift_threshold=self.less_drift,
                persistence_len=self.less_persistence,
            )
        else:
            x, used = _origin_generate(
                self.model, self.tokenizer, prompt_ids,
                gen_length=self.gen_length,
                steps=self.steps,
                block_length=self.block_length,
                temperature=self.temperature,
                mask_id=self.mask_id,
            )

        gen_ids = x[0, prompt_ids.shape[1]:]
        if torch.is_tensor(gen_ids):
            gen_ids = gen_ids.tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - _t0
        return text, used, seconds

    def generate_until(self, requests: List[Instance]):
        def _prep(q):
            if self.always_chat_template and isinstance(q, str):
                chat = [{"role": "user", "content": q}]
                return self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            if isinstance(q, (list, tuple)) and q and isinstance(q[0], dict):
                return self.tokenizer.apply_chat_template(
                    q, tokenize=False, add_generation_prompt=True
                )
            return q

        out = []
        for req in tqdm(requests, desc=f"Generating ({self.alg})"):
            raw_q = req.args[0]
            stop_tokens: List[str] = req.args[1].get("until", [])

            q_text = _prep(raw_q)
            q_ids = self.tokenizer(q_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

            text, used_steps, seconds = self._generate_one(q_ids)

            for stop in stop_tokens:
                if stop in text:
                    text = text.split(stop)[0]

            out.append(text)

            self._meta_buffer.append({
                "question": q_text[:200],
                "answer": text[:500],
                "alg": self.alg,
                "steps_used": used_steps,
                "steps_budget": self.steps,
                "seconds": seconds,
            })

            if self.rank == 0 and len(self._meta_buffer) <= 5:
                print(f"[LOG][Q] {q_text[:120]}")
                print(f"[LOG][A] {text[:200]}")
                print(f"[LOG][Steps] {used_steps}/{self.steps}  [Time] {seconds:.2f}s\n")

        self._flush_metadata()
        return out

    # ---- Loglikelihood (for loglikelihood-based tasks) ----
    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        pi = int(prompt_index.sum().item())
        target_len = l - pi
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
        is_mask = torch.cat(
            (torch.zeros(b, pi, dtype=torch.bool, device=batch.device), is_mask), dim=1
        )
        noisy = torch.where(is_mask, self.mask_id, batch)
        p_mask = (is_mask.float().cumsum(dim=1) / is_mask.sum(dim=1, keepdim=True).clamp_min(1)).clamp_(0, 1)
        return noisy, p_mask

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.cat([prefix, target])[None, :].repeat(self.batch_size, 1).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        loss_acc = []
        for _ in range(max(1, self.mc_num // self.batch_size)):
            perturbed, p_mask = self._forward_process(seq, prompt_index)
            mask_idx = perturbed == self.mask_id
            logits = self.model(perturbed).logits
            loss = F.cross_entropy(logits[mask_idx], seq[mask_idx], reduction="none") / p_mask[mask_idx]
            loss_acc.append((loss.sum() / self.batch_size).item())
        return -float(sum(loss_acc) / len(loss_acc))

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
        ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        return ctx, whole[len(ctx):]

    def loglikelihood(self, requests):
        out = []
        for req in tqdm(requests, desc="Loglikelihood"):
            prefix_enc, target_enc = self._encode_pair(req.args[0], req.args[1])
            prefix_t = torch.tensor(prefix_enc, dtype=torch.long)
            target_t = torch.tensor(target_enc, dtype=torch.long)
            ll = self.get_loglikelihood(prefix_t, target_t)
            out.append((ll, 0.0))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _flush_metadata(self):
        if not self._meta_buffer:
            return
        fname = f"metadata_rank_{self.rank}.json" if self.world_size > 1 else "metadata.json"
        out_path = self.meta_dir / fname
        with open(out_path, "w") as f:
            json.dump({"run": self._run_header, "samples": self._meta_buffer}, f, indent=2)


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
