#!/usr/bin/env python3
"""Dream lm-evaluation-harness wrapper with LESS / origin generation.

Registers model type ``dream_less`` for use with ``lm_eval``.

Usage:
    accelerate launch eval_dream_harness.py \
        --model dream_less \
        --model_args "model_path=Dream-org/Dream-v0-Instruct-7B,alg=less,..." \
        --tasks gsm8k_cot \
        --num_fewshot 8 --batch_size 1 --output_path ./results
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
from tqdm import tqdm

# Backwards-compatible transformers shim for Dream tokenizer
try:
    import transformers.tokenization_utils as _unused  # type: ignore
except Exception:
    import importlib
    try:
        _base = importlib.import_module("transformers.tokenization_utils_base")
    except Exception:
        _base = None
    else:
        if _base is not None and not hasattr(_base, "PreTrainedTokenizer"):
            BaseCls = getattr(_base, "PreTrainedTokenizerBase", None)
            if BaseCls is not None:
                class PreTrainedTokenizer(BaseCls):
                    pass
                PreTrainedTokenizer.__name__ = "PreTrainedTokenizer"
                setattr(_base, "PreTrainedTokenizer", PreTrainedTokenizer)
        sys.modules.setdefault("transformers.tokenization_utils", _base)

from transformers import AutoTokenizer, AutoModel

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (_REPO_ROOT, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dream_sampling.less import diffusion_generate_less


def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _as_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().strip("'\"").lower() in ("1", "true", "yes", "y", "t")


@register_model("dream_less")
class DreamLESSHarness(LM):
    """lm-eval wrapper for Dream with LESS / origin generation."""

    def __init__(
        self,
        model_path: str = "Dream-org/Dream-v0-Instruct-7B",
        max_length: int = 2048,
        batch_size: int = 1,
        steps: int = 256,
        gen_length: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        device: str = "cuda",
        alg: str = "less",
        less_conf: float = 0.75,
        less_drift: float = 0.04,
        less_persistence: int = 2,
        apply_chat_template: bool = False,
        meta_dir: str = "eval_meta",
        **kwargs,
    ):
        super().__init__()
        # NOTE: pass an extended NCCL timeout via InitProcessGroupKwargs to avoid
        # 600s default collective-op timeouts on long-running generative tasks
        # (e.g. hellaswag_generative with LESS sampling, where one rank can run
        # minutes longer than the others before hitting the per-task
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

        if self.accelerator is not None:
            _device_map = {"": f"{self.accelerator.device}"}
        else:
            _device_map = device

        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=_device_map,
        ).eval()
        self.device = next(self.model.parameters()).device
        if self.accelerator is not None:
            self.device = torch.device(f"{self.accelerator.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self.alg = str(alg).strip().lower()

        self.less_conf = float(less_conf)
        self.less_drift = float(less_drift)
        self.less_persistence = int(less_persistence)

        self.always_chat_template = _as_bool(apply_chat_template)

        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._meta_buffer: list = []
        self._run_header = {
            "model_path": model_path,
            "alg": self.alg,
            "steps": self.steps,
            "gen_length": self.gen_length,
            "less_conf": self.less_conf,
            "less_drift": self.less_drift,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_one(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[str, int, float]:
        prompt_ids = prompt_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()

        if self.alg == "less":
            sequences, stats = diffusion_generate_less(
                self.model, prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.gen_length,
                steps=self.steps,
                conf_threshold=self.less_conf,
                drift_threshold=self.less_drift,
                persistence_len=self.less_persistence,
                temperature=self.temperature,
                top_p=self.top_p,
                mask_token_id=self.mask_token_id,
                return_stats=True,
                model_type="dream",
            )
            used = int(stats[0].get("steps_taken", self.steps))
            gen = sequences[0, prompt_ids.shape[1]:]
            text = self.tokenizer.decode(gen.tolist(), skip_special_tokens=True)
        else:
            output = self.model.diffusion_generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.gen_length,
                steps=self.steps,
                temperature=self.temperature,
                top_p=self.top_p,
                alg="origin",
                output_history=True,
                return_dict_in_generate=True,
            )
            gen_ids = output.sequences[0, prompt_ids.shape[1]:]
            text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            used = self.steps

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
            stop_tokens = req.args[1].get("until", [])

            q_text = _prep(raw_q)
            enc = self.tokenizer(q_text, add_special_tokens=True, return_tensors="pt")
            q_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            text, used, seconds = self._generate_one(q_ids, attn)

            for stop in stop_tokens:
                if stop in text:
                    text = text.split(stop)[0]

            out.append(text)

            self._meta_buffer.append({
                "question": q_text[:200],
                "answer": text[:500],
                "alg": self.alg,
                "steps_used": used,
                "steps_budget": self.steps,
                "seconds": seconds,
            })

            if self.rank == 0 and len(self._meta_buffer) <= 5:
                print(f"[LOG][Q] {q_text[:120]}")
                print(f"[LOG][A] {text[:200]}")
                print(f"[LOG][Steps] {used}/{self.steps}  [Time] {seconds:.2f}s\n")

        self._flush_metadata()
        return out

    def loglikelihood(self, requests):
        raise NotImplementedError("Dream harness: loglikelihood not implemented — use generate_until tasks.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _flush_metadata(self):
        if not self._meta_buffer:
            return
        fname = f"metadata_rank_{self.rank}.json" if self.world_size > 1 else "metadata.json"
        with open(self.meta_dir / fname, "w") as f:
            json.dump({"run": self._run_header, "samples": self._meta_buffer}, f, indent=2)


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
