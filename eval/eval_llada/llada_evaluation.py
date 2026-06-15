#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaDA evaluation script (GSM8K / MATH / HumanEval / MBPP)

Drop-in replacement for your current eval script with the following RESULT-AFFECTING fixes:
  1) Chat inputs are tokenized directly via tokenizer.apply_chat_template(..., return_tensors="pt", return_dict=True)
     (prevents double-tokenization / double-special-token bugs).
  2) Decoding mirrors Dream-style decoding:
       - decode generated tokens with skip_special_tokens=False
       - truncate ONLY if tokenizer.eos_token is a valid non-empty string
     (prevents eos_token=None -> whitespace split catastrophe).
  3) Fixes the MATH evaluation bug where `prompt` was overwritten by the chat-templated prompt string.
     (evaluation now uses the raw dataset prompt for extract/compare).
  4) Forces a consistent LLaDA mask_id (LLADA_MASK_TOKEN_ID).
  5) Supports alg={less, origin}: LESS adaptive decoding vs. vanilla fixed-schedule
     block-wise decoding (the two paths compared in the paper).
  6) Optional torchrun distributed support (merges outputs on rank 0).

Repo assumptions:
  - utils.py provides:
      process_file, parse_ground_truth_answer, parse_answer, extract_math_answer, compare_answers, evaluate_task
  - human_eval is available:
      from human_eval.data import read_problems, write_jsonl
      from human_eval.evaluation import evaluate_functional_correctness
  - llada_sampling provides:
      stable_less_decode, LLADA_MASK_TOKEN_ID, add_gumbel_noise, get_num_transfer_tokens
"""

import os
import sys
import json
import re
import random
import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# --------------------
# Ensure repository root is on sys.path
# --------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from utils import *  # noqa: F401,F403
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

from llada_sampling.llada_less import (
    stable_less_decode as _stable_less_decode,
    LLADA_MASK_TOKEN_ID,
    add_gumbel_noise,
    get_num_transfer_tokens,
)


# --------------------
# Vanilla (origin) block-wise generation for LLaDA — the baseline LESS is compared against
# --------------------
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


# --------------------
# Distributed helpers
# --------------------
def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_world_size() -> int:
    return dist.get_world_size() if _dist_is_initialized() else 1


def _get_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main_process() -> bool:
    return _get_rank() == 0


def _barrier() -> None:
    if _dist_is_initialized():
        dist.barrier()


# --------------------
# Utility: only pass supported kwargs (prevents signature mismatch crashes)
# --------------------
def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    allowed = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


# --------------------
# Chat input preparation (fixed: no double-tokenization)
# --------------------
def _prepare_chat_inputs(
    tokenizer,
    messages: List[Dict[str, str]],
    device: torch.device,
) -> Tuple[str, torch.LongTensor, Optional[torch.LongTensor]]:
    """
    Build the chat template string with tokenize=False, then tokenize that
    string directly via tokenizer(prompt)["input_ids"] (avoids double-tokenization).
    Returns (prompt_text, input_ids, None)
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = torch.tensor(tokenizer(prompt_text)["input_ids"]).to(device).unsqueeze(0)
    return prompt_text, input_ids, None


# --------------------
# Decode generation like Dream (fixed: eos_token=None safe)
# --------------------
def _decode_generation_like_dream(
    tokenizer,
    prompt_input_ids: torch.LongTensor,  # [B, prompt_len]
    full_output_ids: torch.LongTensor,   # [B, prompt_len + gen_len]
) -> str:
    """
    Decode the generated portion with skip_special_tokens=True for GSM8K/MATH.
    EOS splitting is handled explicitly in HumanEval/MBPP paths only.
    """
    generations = tokenizer.batch_decode(
        full_output_ids[:, prompt_input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return generations[0] if generations else ""


# --------------------
# LESS config helper
# --------------------
def _validate_less_kwargs(less_kwargs: Dict[str, Any]) -> None:
    sig = inspect.signature(_stable_less_decode)
    allowed = set(sig.parameters.keys())
    allowed.discard("model")
    allowed.discard("tokenizer")
    allowed.discard("input_ids_original")
    unknown = set(less_kwargs.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown LESS kwargs: {sorted(unknown)}")


def _build_run_config(*, alg, dataset, gen_length, steps, block_length, less_kwargs):
    cfg = {
        "alg": alg,
        "dataset": dataset,
        "gen_length": int(gen_length),
        "steps": int(steps),
        "block_length": int(block_length),
    }
    if alg == "less":
        cfg["less_kwargs"] = dict(less_kwargs or {})
    return cfg


# --------------------
# Eval: GSM8K / MATH
# --------------------
def test_dataset(
    model,
    tokenizer,
    save_dir: str,
    dataset: str,
    gen_length: int,
    steps: int,
    block_length: int,
    alg: str = "less",
    temperature: float = 0.0,
    test_size: Optional[int] = None,
    random_sampling: bool = False,
    num_samples: int = 1,
    save_steps: bool = False,
    less_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    # Directory naming
    if alg == "less":
        lk = less_kwargs or {}
        c = lk.get("conf_threshold", 0.75)
        d = lk.get("drift_threshold", 0.04)
        run_dir = (
            f"{save_dir}/LLaDA/{dataset}/less/all/len_{gen_length}_block_{block_length}/steps_{steps}"
            f"/c{c}_d{d}_s{num_samples}"
        )
    else:
        run_dir = (
            f"{save_dir}/LLaDA/{dataset}/{alg}/len_{gen_length}_block_{block_length}/steps_{steps}"
            f"/s{num_samples}"
        )
    os.makedirs(run_dir, exist_ok=True)

    world_size = _get_world_size()
    rank = _get_rank()
    is_main = _is_main_process()

    device = torch.device(f"cuda:{_get_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    rank_dir = os.path.join(run_dir, f"rank{rank}") if world_size > 1 else run_dir
    if world_size > 1:
        os.makedirs(rank_dir, exist_ok=True)

    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(rank_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    # Load data
    data_path = f"./data/{dataset}_test.json"
    data = process_file(data_path)

    if test_size:
        random.seed(516)
        data = random.sample(data, test_size) if random_sampling else data[:test_size]

    # Partition by rank (keep global ids stable)
    indices = list(range(len(data)))
    if world_size > 1:
        indices = indices[rank::world_size]

    correct_count = 0
    used_steps_list: List[int] = []
    results: Dict[str, Any] = {"summary": {}, "results": []}

    for global_i in tqdm(
        indices,
        total=len(indices),
        desc=f"Generating completions for {dataset.capitalize()}",
        disable=not is_main,
    ):
        example = data[global_i]

        if dataset == "gsm8k":
            raw_prompt = example["question"]
            answer = example["answer"]
            ground_truth_answer = parse_ground_truth_answer(answer)
        elif dataset == "math":
            raw_prompt = example["problem"]
            solution = example["solution"]
            ground_truth_answer = extract_math_answer(raw_prompt, solution)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        example_samples: List[Dict[str, Any]] = []
        example_correct = False
        example_steps: List[int] = []

        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to answer the question below. Give step by step reasoning before you answer, "
                    "and when you're ready to answer, please use the format 'The final answer is'."
                ),
            },
            {"role": "user", "content": raw_prompt},
        ]
        prompt_chat, input_ids_original, _attention_mask = _prepare_chat_inputs(tokenizer, messages, device=device)

        for sample_idx in range(num_samples):
            if alg == "less":
                lk = dict(less_kwargs or {})
                lk.setdefault("mask_id", LLADA_MASK_TOKEN_ID)
                lk.setdefault("gen_length", int(gen_length))
                lk.setdefault("steps", int(steps))
                lk.setdefault("block_length", int(block_length))
                lk.setdefault("temperature", float(temperature))
                if save_steps and step_save_dir:
                    lk.setdefault("step_save_dir", step_save_dir)
                    lk.setdefault("example_idx", f"q{global_i}_s{sample_idx}")
                x_output, used_steps = _stable_less_decode(
                    model, tokenizer, input_ids_original, **lk,
                )
            else:
                x_output, used_steps = _origin_generate(
                    model, tokenizer, input_ids_original,
                    gen_length=int(gen_length),
                    steps=int(steps),
                    block_length=int(block_length),
                    temperature=float(temperature),
                    mask_id=LLADA_MASK_TOKEN_ID,
                )

            generated_text = _decode_generation_like_dream(tokenizer, input_ids_original, x_output)

            if dataset == "gsm8k":
                generated_answer = parse_answer(generated_text)
                is_correct = (generated_answer == ground_truth_answer)
            else:
                generated_answer = extract_math_answer(prompt_chat, generated_text)
                is_correct = compare_answers(prompt_chat, ground_truth_answer, generated_answer)

            if is_correct:
                example_correct = True

            example_steps.append(int(used_steps))
            example_samples.append(
                {
                    "task_id": global_i,
                    "sample_idx": sample_idx,
                    "used_steps": int(used_steps),
                    "generation": generated_text,
                    "parsed_answer": generated_answer,
                    "is_correct": bool(is_correct),
                }
            )

        if example_correct:
            correct_count += 1

        used_steps_list.extend(example_steps)
        results["results"].append(
            {
                "task_id": global_i,
                "input_prompt": prompt_chat,
                "ground_truth_answer": ground_truth_answer,
                "any_correct": example_correct,
                "avg_steps": round(float(np.mean(example_steps)), 2) if example_steps else 0.0,
                "samples": example_samples,
            }
        )

    accuracy = (correct_count / len(indices)) if indices else 0.0
    avg_steps = float(np.mean(used_steps_list)) if used_steps_list else 0.0
    results["summary"] = {
        "accuracy": round(accuracy * 100, 2),
        "average_steps": round(avg_steps, 2),
        "total_questions": len(indices),
        "correct_questions": correct_count,
        "num_samples_per_question": num_samples,
        "world_size": world_size,
        "rank": rank,
    }
    results["config"] = _build_run_config(
        alg=alg, dataset=dataset, gen_length=gen_length, steps=steps,
        block_length=block_length, less_kwargs=less_kwargs,
    )

    # Save results (merge on rank 0 if distributed)
    if world_size > 1:
        partial_path = os.path.join(rank_dir, f"all_results_rank{rank}.json")
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        _barrier()

        if is_main:
            merged: Dict[str, Any] = {"summary": {}, "results": []}
            for r in range(world_size):
                r_path = os.path.join(run_dir, f"rank{r}", f"all_results_rank{r}.json")
                if os.path.exists(r_path):
                    with open(r_path, "r", encoding="utf-8") as rf:
                        part = json.load(rf)
                    merged["results"].extend(part.get("results", []))

            total_count = len(merged["results"])
            total_correct = sum(1 for x in merged["results"] if x.get("any_correct", False))
            steps_mean = float(np.mean([x.get("avg_steps", 0.0) for x in merged["results"]])) if total_count else 0.0

            merged["summary"] = {
                "accuracy": round((total_correct / total_count) * 100, 2) if total_count else 0.0,
                "average_steps": round(steps_mean, 2),
                "total_questions": total_count,
                "correct_questions": total_correct,
                "num_samples_per_question": num_samples,
                "world_size": world_size,
            }
            merged["config"] = _build_run_config(
                alg=alg, dataset=dataset, gen_length=gen_length, steps=steps,
                block_length=block_length, less_kwargs=less_kwargs,
            )

            save_path = os.path.join(run_dir, "all_results.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)

            print(f"[{dataset.capitalize()}]")
            print(f"Accuracy: {merged['summary']['accuracy']}")
            print(f"Average steps: {merged['summary']['average_steps']}")
            print(f"Results saved to {save_path}")

        _barrier()
    else:
        save_path = os.path.join(run_dir, "all_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[{dataset.capitalize()}]")
        print(f"Accuracy: {results['summary']['accuracy']}")
        print(f"Average steps: {results['summary']['average_steps']}")
        print(f"Results saved to {save_path}")


# --------------------
# Eval: HumanEval
# --------------------
def _call_humaneval_evaluator(
    samples_file: str,
    k_tuple: Tuple[int, ...],
    n_workers: int,
    timeout: float,
) -> Dict[str, Any]:
    """
    human_eval forks differ: some expect k as a tuple/list, others accept a comma-separated string.
    We try tuple first, then fall back to string if needed.
    """
    try:
        return evaluate_functional_correctness(
            samples_file,
            k=k_tuple,
            n_workers=n_workers,
            timeout=timeout,
        )
    except Exception:
        k_str = ",".join(str(x) for x in k_tuple)
        return evaluate_functional_correctness(
            samples_file,
            k=k_str,
            n_workers=n_workers,
            timeout=timeout,
        )


def test_humaneval(
    model,
    tokenizer,
    save_dir: str,
    gen_length: int,
    steps: int,
    block_length: int,
    alg: str = "less",
    temperature: float = 0.0,
    k: Tuple[int, ...] = (1, 10, 100),
    n_workers: int = 4,
    timeout: float = 3.0,
    test_size: Optional[int] = None,
    random_sampling: bool = False,
    num_samples: int = 1,
    save_steps: bool = False,
    less_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    if alg == "less":
        lk = less_kwargs or {}
        c = lk.get("conf_threshold", 0.75)
        d = lk.get("drift_threshold", 0.04)
        run_dir = (
            f"{save_dir}/LLaDA/humaneval/less/all/len_{gen_length}_block_{block_length}/steps_{steps}"
            f"/c{c}_d{d}_s{num_samples}"
        )
    else:
        run_dir = (
            f"{save_dir}/LLaDA/humaneval/{alg}/len_{gen_length}_block_{block_length}/steps_{steps}"
            f"/s{num_samples}"
        )
    os.makedirs(run_dir, exist_ok=True)

    world_size = _get_world_size()
    rank = _get_rank()
    is_main = _is_main_process()
    device = torch.device(f"cuda:{_get_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    rank_dir = os.path.join(run_dir, f"rank{rank}") if world_size > 1 else run_dir
    if world_size > 1:
        os.makedirs(rank_dir, exist_ok=True)

    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(rank_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    problems = read_problems()
    items = list(problems.items())

    if test_size:
        random.seed(516)
        items = random.sample(items, test_size) if random_sampling else items[:test_size]

    if world_size > 1:
        items = items[rank::world_size]

    samples: List[Dict[str, Any]] = []
    steps_per_problem: List[Dict[str, Any]] = []

    for idx, (task_id, info) in enumerate(tqdm(items, desc="Generating completions for HumanEval", disable=not is_main)):
        raw_prompt = info["prompt"]
        task_samples: List[Dict[str, Any]] = []
        task_steps: List[int] = []

        messages = [
            {"role": "system", "content": "You complete only Python code."},
            {"role": "user", "content": raw_prompt},
        ]
        prompt_chat, input_ids_original, _attention_mask = _prepare_chat_inputs(tokenizer, messages, device=device)

        for sample_idx in range(num_samples):
            if alg == "less":
                lk = dict(less_kwargs or {})
                lk.setdefault("mask_id", LLADA_MASK_TOKEN_ID)
                lk.setdefault("gen_length", int(gen_length))
                lk.setdefault("steps", int(steps))
                lk.setdefault("block_length", int(block_length))
                lk.setdefault("temperature", float(temperature))
                if save_steps and step_save_dir:
                    lk.setdefault("step_save_dir", step_save_dir)
                    lk.setdefault("example_idx", f"q{idx}_s{sample_idx}")
                x_output, used_steps = _stable_less_decode(
                    model, tokenizer, input_ids_original, **lk,
                )
            else:
                x_output, used_steps = _origin_generate(
                    model, tokenizer, input_ids_original,
                    gen_length=int(gen_length),
                    steps=int(steps),
                    block_length=int(block_length),
                    temperature=float(temperature),
                    mask_id=LLADA_MASK_TOKEN_ID,
                )

            task_steps.append(int(used_steps))

            generations = tokenizer.batch_decode(
                x_output[:, input_ids_original.shape[1]:],
                skip_special_tokens=True,
            )
            decoded_text = generations[0]
            eos_tok = getattr(tokenizer, "eos_token", None)
            if isinstance(eos_tok, str) and eos_tok:
                decoded_text = decoded_text.split(eos_tok)[0]
            code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", decoded_text, re.DOTALL)
            code_only = code_match.group(1).strip() if code_match else decoded_text.strip()

            sample_data = {
                "task_id": task_id,
                "sample_idx": sample_idx,
                "completion": code_only,
                "used_steps": int(used_steps),
            }
            task_samples.append(sample_data)
            samples.append(sample_data)

        steps_per_problem.append(
            {
                "task_id": task_id,
                "input_prompt": prompt_chat,
                "avg_steps": float(np.mean(task_steps)) if task_steps else 0.0,
                "samples": task_samples,
            }
        )

    # Distributed merge + evaluate
    if world_size > 1:
        rank_samples_file = os.path.join(rank_dir, f"humaneval_samples_rank{rank}.jsonl")
        write_jsonl(rank_samples_file, samples)

        rank_steps_file = os.path.join(rank_dir, f"humaneval_steps_rank{rank}.json")
        with open(rank_steps_file, "w", encoding="utf-8") as f:
            json.dump({"results": steps_per_problem}, f, indent=2)

        _barrier()

        if is_main:
            merged_samples_file = os.path.join(run_dir, "humaneval_samples.jsonl")
            with open(merged_samples_file, "w", encoding="utf-8") as out_f:
                for r in range(world_size):
                    part_file = os.path.join(run_dir, f"rank{r}", f"humaneval_samples_rank{r}.jsonl")
                    if os.path.exists(part_file):
                        with open(part_file, "r", encoding="utf-8") as pf:
                            for line in pf:
                                out_f.write(line)

            merged_steps: List[Dict[str, Any]] = []
            for r in range(world_size):
                part_steps = os.path.join(run_dir, f"rank{r}", f"humaneval_steps_rank{r}.json")
                if os.path.exists(part_steps):
                    with open(part_steps, "r", encoding="utf-8") as psf:
                        merged_steps.extend(json.load(psf).get("results", []))

            results_eval = _call_humaneval_evaluator(
                merged_samples_file,
                k_tuple=k,
                n_workers=n_workers,
                timeout=timeout,
            )
            avg_steps = float(np.mean([entry.get("avg_steps", 0.0) for entry in merged_steps])) if merged_steps else 0.0

            all_results = {
                "summary": {
                    "accuracy": round(float(results_eval.get("pass@1", 0.0)) * 100, 2),
                    "average_steps": round(avg_steps, 2),
                },
                "config": _build_run_config(
                    alg=alg, dataset="humaneval", gen_length=gen_length, steps=steps,
                    block_length=block_length, less_kwargs=less_kwargs,
                ),
                "results": merged_steps,
            }

            save_path = os.path.join(run_dir, "all_results.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

            print("[HumanEval]")
            print("Accuracy:", round(float(results_eval.get("pass@1", 0.0)) * 100, 2))
            print(f"Average steps: {avg_steps:.2f}")
            print(f"Results saved to {save_path}")

        _barrier()
    else:
        samples_file = os.path.join(run_dir, "humaneval_samples.jsonl")
        write_jsonl(samples_file, samples)

        results_eval = _call_humaneval_evaluator(
            samples_file,
            k_tuple=k,
            n_workers=n_workers,
            timeout=timeout,
        )
        avg_steps = float(np.mean([entry.get("avg_steps", 0.0) for entry in steps_per_problem])) if steps_per_problem else 0.0

        all_results = {
            "summary": {
                "accuracy": round(float(results_eval.get("pass@1", 0.0)) * 100, 2),
                "average_steps": round(avg_steps, 2),
            },
            "config": _build_run_config(
                alg=alg, dataset="humaneval", gen_length=gen_length, steps=steps,
                block_length=block_length, less_kwargs=less_kwargs,
            ),
            "results": steps_per_problem,
        }

        save_path = os.path.join(run_dir, "all_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print("[HumanEval]")
        print("Accuracy:", round(float(results_eval.get("pass@1", 0.0)) * 100, 2))
        print(f"Average steps: {avg_steps:.2f}")
        print(f"Results saved to {save_path}")


# --------------------
# Eval: MBPP
# --------------------
def test_mbpp(
    model,
    tokenizer,
    save_dir: str,
    gen_length: int,
    steps: int,
    block_length: int,
    alg: str = "less",
    temperature: float = 0.0,
    eval_timeout: float = 3.0,
    test_size: Optional[int] = None,
    random_sampling: bool = False,
    num_samples: int = 1,
    save_steps: bool = False,
    less_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    if alg == "less":
        lk = less_kwargs or {}
        c = lk.get("conf_threshold", 0.75)
        d = lk.get("drift_threshold", 0.04)
        run_dir = (
            f"{save_dir}/LLaDA/mbpp/less/all/len_{gen_length}_block_{block_length}/steps_{steps}"
            f"/c{c}_d{d}_s{num_samples}"
        )
    else:
        run_dir = (
            f"{save_dir}/LLaDA/mbpp/{alg}/len_{gen_length}_block_{block_length}/steps_{steps}"
            f"/s{num_samples}"
        )
    os.makedirs(run_dir, exist_ok=True)

    world_size = _get_world_size()
    rank = _get_rank()
    is_main = _is_main_process()
    device = torch.device(f"cuda:{_get_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    rank_dir = os.path.join(run_dir, f"rank{rank}") if world_size > 1 else run_dir
    if world_size > 1:
        os.makedirs(rank_dir, exist_ok=True)

    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(rank_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    if test_size:
        random.seed(516)
        ds = random.sample(list(ds), test_size) if random_sampling else ds.select(range(test_size))

    # Partition by rank
    if world_size > 1:
        ds = list(ds)
        ds = ds[rank::world_size]

    steps_per_problem: List[Dict[str, Any]] = []

    for idx, ex in enumerate(tqdm(ds, desc="Generating completions for MBPP", disable=not is_main)):
        task_id = ex["task_id"]
        raw_prompt = ex["prompt"]
        tests = ex["test_list"]
        code = ex["code"]

        task_samples: List[Dict[str, Any]] = []
        task_steps: List[int] = []
        task_passed = False

        messages = [
            {
                "role": "user",
                "content": (
                    f"You are an expert Python programmer, and here is your task: {raw_prompt} "
                    f"Your code should pass these tests:\n\n{tests}\n[BEGIN]"
                ),
            }
        ]
        prompt_chat, input_ids_original, _attention_mask = _prepare_chat_inputs(tokenizer, messages, device=device)

        for sample_idx in range(num_samples):
            if alg == "less":
                lk = dict(less_kwargs or {})
                lk.setdefault("mask_id", LLADA_MASK_TOKEN_ID)
                lk.setdefault("gen_length", int(gen_length))
                lk.setdefault("steps", int(steps))
                lk.setdefault("block_length", int(block_length))
                lk.setdefault("temperature", float(temperature))
                if save_steps and step_save_dir:
                    lk.setdefault("step_save_dir", step_save_dir)
                    lk.setdefault("example_idx", f"q{idx}_s{sample_idx}")
                x_output, used_steps = _stable_less_decode(
                    model, tokenizer, input_ids_original, **lk,
                )
            else:
                x_output, used_steps = _origin_generate(
                    model, tokenizer, input_ids_original,
                    gen_length=int(gen_length),
                    steps=int(steps),
                    block_length=int(block_length),
                    temperature=float(temperature),
                    mask_id=LLADA_MASK_TOKEN_ID,
                )

            generations = tokenizer.batch_decode(
                x_output[:, input_ids_original.shape[1]:],
                skip_special_tokens=True,
            )
            decoded_text = generations[0] if generations else ""
            eos_tok = getattr(tokenizer, "eos_token", None)
            if isinstance(eos_tok, str) and eos_tok:
                decoded_text = decoded_text.split(eos_tok)[0]

            sample_data = {
                "task_id": task_id,
                "sample_idx": sample_idx,
                "used_steps": int(used_steps),
                "generation": decoded_text,
            }

            passed = evaluate_task(sample_data, tests, timeout=eval_timeout)
            if passed:
                task_passed = True
            sample_data["passed"] = bool(passed)

            task_samples.append(sample_data)
            task_steps.append(int(used_steps))

        steps_per_problem.append(
            {
                "task_id": task_id,
                "input_prompt": prompt_chat,
                "solution_code": code,
                "any_passed": task_passed,
                "avg_steps": float(np.mean(task_steps)) if task_steps else 0.0,
                "samples": task_samples,
            }
        )

    # Merge + summarize
    if world_size > 1:
        partial = os.path.join(rank_dir, f"mbpp_results_rank{rank}.json")
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"results": steps_per_problem}, f, indent=2)

        _barrier()

        if is_main:
            merged: List[Dict[str, Any]] = []
            for r in range(world_size):
                r_path = os.path.join(run_dir, f"rank{r}", f"mbpp_results_rank{r}.json")
                if os.path.exists(r_path):
                    with open(r_path, "r", encoding="utf-8") as rf:
                        merged.extend(json.load(rf).get("results", []))

            total_passed = sum(1 for t in merged if t.get("any_passed", False))
            accuracy = (total_passed / len(merged)) if merged else 0.0
            avg_steps = float(np.mean([t.get("avg_steps", 0.0) for t in merged])) if merged else 0.0

            all_results = {
                "summary": {
                    "accuracy": round(accuracy * 100, 2),
                    "average_steps": round(avg_steps, 2),
                    "total_tasks": len(merged),
                    "passed_tasks": total_passed,
                    "num_samples_per_task": num_samples,
                    "world_size": world_size,
                },
                "config": _build_run_config(
                    alg=alg, dataset="mbpp", gen_length=gen_length, steps=steps,
                    block_length=block_length, less_kwargs=less_kwargs,
                ),
                "results": merged,
            }

            save_path = os.path.join(run_dir, "all_results.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

            print("[MBPP]")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Average steps: {avg_steps:.2f}")
            print(f"Results saved to {save_path}")

        _barrier()
    else:
        total_passed = sum(1 for t in steps_per_problem if t.get("any_passed", False))
        accuracy = (total_passed / len(steps_per_problem)) if steps_per_problem else 0.0
        avg_steps = float(np.mean([t.get("avg_steps", 0.0) for t in steps_per_problem])) if steps_per_problem else 0.0

        all_results = {
            "summary": {
                "accuracy": round(accuracy * 100, 2),
                "average_steps": round(avg_steps, 2),
                "total_tasks": len(steps_per_problem),
                "passed_tasks": total_passed,
                "num_samples_per_task": num_samples,
            },
            "config": _build_run_config(
                alg=alg, dataset="mbpp", gen_length=gen_length, steps=steps,
                block_length=block_length, less_kwargs=less_kwargs,
            ),
            "results": steps_per_problem,
        }

        save_path = os.path.join(run_dir, "all_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print("[MBPP]")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average steps: {avg_steps:.2f}")
        print(f"Results saved to {save_path}")


# --------------------
# Main
# --------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLaDA model on gsm8k, math, humaneval, or mbpp.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math", "humaneval", "mbpp"], help="Dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--gen_length", type=int, default=256, help="Generation length")
    parser.add_argument("--block_length", type=int, default=64, help="Block length")
    parser.add_argument("--steps", type=int, default=256, help="Number of diffusion steps")

    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--test_size", type=int, default=None, help="Number of test examples")
    parser.add_argument("--random_sampling", action="store_true", help="Use random sampling")

    parser.add_argument(
        "--alg",
        type=str,
        default="less",
        choices=["less", "origin"],
        help="Algorithm to use",
    )

    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question/task")
    parser.add_argument("--save_steps", action="store_true", help="If set, save stepwise outputs (if supported).")

    # LESS args
    parser.add_argument("--less_conf", type=float, default=0.75, help="Confidence threshold for LESS")
    parser.add_argument("--less_drift", type=float, default=0.04, help="JS drift threshold for LESS")
    parser.add_argument("--less_config", type=str, default=None, help="Path to LESS config JSON (overrides CLI knobs)")

    # HumanEval-specific
    parser.add_argument("--humaneval_k", type=str, default="1", help="pass@k values, comma-separated (e.g., 1,10,100)")
    parser.add_argument("--humaneval_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--humaneval_timeout", type=float, default=3.0, help="Test timeout (seconds)")

    args = parser.parse_args()
    print("Parsed arguments:", args)

    # Initialize distributed if launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not _dist_is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = _get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Load model/tokenizer (fix: use torch_dtype)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # LESS kwargs. Merge order (low -> high precedence):
    #   1. CLI knobs (--less_conf, --less_drift)
    #   2. --less_config JSON (explicit override)
    less_kwargs = None
    if args.alg == "less":
        less_kwargs = {
            "conf_threshold": args.less_conf,
            "drift_threshold": args.less_drift,
        }
        if args.less_config is not None:
            with open(args.less_config, "r") as f:
                user_cfg = json.load(f)
            _validate_less_kwargs(user_cfg)
            less_kwargs.update(user_cfg)

    # Dispatch
    if args.dataset in ["gsm8k", "math"]:
        test_dataset(
            model=model,
            tokenizer=tokenizer,
            save_dir=args.save_dir,
            dataset=args.dataset,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            alg=args.alg,
            temperature=args.temperature,
            test_size=args.test_size,
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
            less_kwargs=less_kwargs,
        )
    elif args.dataset == "humaneval":
        ks = tuple(int(x.strip()) for x in args.humaneval_k.split(",") if x.strip())
        test_humaneval(
            model=model,
            tokenizer=tokenizer,
            save_dir=args.save_dir,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            alg=args.alg,
            temperature=args.temperature,
            k=ks,
            n_workers=args.humaneval_workers,
            timeout=args.humaneval_timeout,
            test_size=args.test_size,
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
            less_kwargs=less_kwargs,
        )
    elif args.dataset == "mbpp":
        test_mbpp(
            model=model,
            tokenizer=tokenizer,
            save_dir=args.save_dir,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            alg=args.alg,
            temperature=args.temperature,
            test_size=args.test_size,
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
            less_kwargs=less_kwargs,
        )

    if _dist_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
