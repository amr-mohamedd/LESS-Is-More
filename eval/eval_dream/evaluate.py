#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import inspect
import torch
import numpy as np
from tqdm import tqdm
import random
import re
import argparse
import torch.distributed as dist

# Dream checkpoint tokenizers still rely on the removed transformers.tokenization_utils
# module. Recreate a lightweight alias (and PreTrainedTokenizer shim) so that
# AutoTokenizer can load them under newer transformers versions.
try:  # pragma: no cover
    import transformers.tokenization_utils as _unused_tokenization_utils  # type: ignore
except Exception:  # pragma: no cover
    import importlib

    try:
        _tokenization_utils_base = importlib.import_module("transformers.tokenization_utils_base")
    except Exception:
        _tokenization_utils_base = None
    else:
        if _tokenization_utils_base is not None and not hasattr(
            _tokenization_utils_base, "PreTrainedTokenizer"
        ):
            BaseCls = getattr(_tokenization_utils_base, "PreTrainedTokenizerBase", None)

            if BaseCls is not None:

                class PreTrainedTokenizer(BaseCls):  # type: ignore
                    """Backwards-compatible alias removed in HF >= 4.45."""

                    pass

                PreTrainedTokenizer.__name__ = "PreTrainedTokenizer"
                setattr(_tokenization_utils_base, "PreTrainedTokenizer", PreTrainedTokenizer)

        sys.modules.setdefault("transformers.tokenization_utils", _tokenization_utils_base)

from transformers import AutoTokenizer, AutoModel

# Ensure repository root and eval/ are on sys.path so `dream_sampling` and `utils` can be imported
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.dirname(_CUR_DIR)
_REPO_ROOT = os.path.dirname(_EVAL_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from dream_sampling.less import diffusion_generate_less as _diffusion_generate_less
from datasets import load_dataset

from utils import *
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


# --------------------
# Distributed helpers
# --------------------
def _dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def _get_world_size():
    return dist.get_world_size() if _dist_is_initialized() else 1


def _get_rank():
    return dist.get_rank() if _dist_is_initialized() else 0


def _get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main_process():
    return _get_rank() == 0


def _barrier():
    if _dist_is_initialized():
        dist.barrier()


def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.dim() > 0 else obj.item()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj


# --------------------
# LESS config helpers
# --------------------
def _load_less_config(path: str) -> dict:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"--less_config must be a JSON object/dict, got: {type(obj)}")
    return obj


def _validate_less_kwargs(less_kwargs: dict) -> None:
    sig = inspect.signature(_diffusion_generate_less)
    allowed = set(sig.parameters.keys())
    allowed.discard("model")
    allowed.discard("input_ids")
    unknown = set(less_kwargs.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown LESS kwargs: {sorted(unknown)}")


def _build_run_config(*, alg, dataset, gen_length, steps, less_kwargs):
    cfg = {
        "alg": alg,
        "dataset": dataset,
        "gen_length": int(gen_length),
        "steps": int(steps),
    }
    if alg == "less":
        cfg["less_kwargs"] = dict(less_kwargs or {})
    return cfg


# --------------------
# Evaluation: GSM8K/MATH
# --------------------
def test_dataset(
    model,
    tokenizer,
    save_dir,
    dataset,
    gen_length,
    steps,
    alg="less",
    temperature=0.2,
    top_p=0.95,
    test_size=None,
    random_sampling=False,
    num_samples=1,
    save_steps=False,
    less_kwargs=None,
):
    if alg == "less":
        lk = less_kwargs or {}
        c = lk.get("conf_threshold", 0.75)
        d = lk.get("drift_threshold", 0.04)
        save_dir = (
            f"{save_dir}/Dream/{dataset}/less/all/len_{gen_length}/steps_{steps}"
            f"/c{c}_d{d}_s{num_samples}"
        )
    else:
        save_dir = f"{save_dir}/Dream/{dataset}/{alg}/len_{gen_length}/steps_{steps}/s{num_samples}"
    os.makedirs(save_dir, exist_ok=True)

    world_size = _get_world_size()
    rank = _get_rank()
    is_main = _is_main_process()
    rank_dir = os.path.join(save_dir, f"rank{rank}") if world_size > 1 else save_dir
    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(rank_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    data_path = f"./data/{dataset}_test.json"
    data = process_file(data_path)

    if test_size:
        random.seed(516)
        data = random.sample(data, test_size) if random_sampling else data[:test_size]

    # Partition data across ranks for distributed inference
    if world_size > 1:
        data = data[rank::world_size]

    correct_count = 0
    used_steps_list = []
    results = {"summary": {}, "results": []}

    for example_idx, example in tqdm(
        enumerate(data),
        total=len(data),
        desc=f"Generating completions for {dataset.capitalize()}",
        disable=not is_main,
    ):
        if dataset == "gsm8k":
            prompt = example["question"]
            answer = example["answer"]
            ground_truth_answer = parse_ground_truth_answer(answer)
        elif dataset == "math":
            prompt = example["problem"]
            solution = example["solution"]
            ground_truth_answer = extract_math_answer(prompt, solution)
        else:
            raise ValueError(f"Unsupported dataset in test_dataset: {dataset}")

        example_samples = []
        example_correct = False
        example_steps = []

        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to answer the question below. Give step by step reasoning before you answer, "
                    "and when you're ready to answer, please use the format 'The final answer is'."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device=f"cuda:{_get_local_rank()}")
        attention_mask = inputs.attention_mask.to(device=f"cuda:{_get_local_rank()}")

        for sample_idx in range(num_samples):
            if alg == "less":
                lk = dict(less_kwargs or {})
                lk.setdefault("mask_token_id", tokenizer.mask_token_id)
                lk.setdefault("max_new_tokens", gen_length)
                lk.setdefault("steps", steps)
                lk.setdefault("temperature", temperature)
                lk.setdefault("top_p", top_p)
                lk.setdefault("return_stats", True)
                lk.setdefault("collect_step_trace", bool(save_steps))
                sequences, stats_list = _diffusion_generate_less(
                    model, input_ids, attention_mask=attention_mask, **lk,
                )
                used_steps = int(stats_list[0].get("steps_taken", steps))
                generations = [tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, sequences)]
                output = None
                if save_steps and stats_list:
                    fb = stats_list[0].get("per_step_fallback", [])
                    an = stats_list[0].get("per_step_accepted_n", [])
                    ax_ids = stats_list[0].get("per_step_argmax_ids", [])
                    step_trace = [
                        {"step": _i, "fallback": bool(fb[_i]),
                         "accepted_tokens_num": int(an[_i]) if _i < len(an) else 0,
                         "argmax_ids": (list(ax_ids[_i]) if _i < len(ax_ids) else [])}
                        for _i in range(len(fb))
                    ]
                    all_steps_path = os.path.join(step_save_dir, f"q{example_idx}_s{sample_idx}.json")
                    with open(all_steps_path, "w") as _ftrace:
                        json.dump(step_trace, _ftrace)
            else:
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=temperature,
                    top_p=top_p,
                    alg=alg,
                    alg_temp=0.0,
                )
                used_steps = int(getattr(output, "used_steps", steps))
                generations = [tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, output.sequences)]

            generated_text = generations[0].split(tokenizer.eos_token)[0]

            if dataset == "gsm8k":
                generated_answer = parse_answer(generated_text)
                is_correct = generated_answer == ground_truth_answer
            else:  # math
                generated_answer = extract_math_answer(prompt, generated_text)
                is_correct = compare_answers(prompt, ground_truth_answer, generated_answer)

            if is_correct:
                example_correct = True

            example_steps.append(used_steps)
            example_samples.append(
                {
                    "task_id": example_idx,
                    "sample_idx": sample_idx,
                    "used_steps": used_steps,
                    "generation": generated_text,
                    "parsed_answer": generated_answer,
                    "is_correct": is_correct,
                }
            )

            if save_steps and output is not None:
                history = output.history
                decoded_history = []
                for step in history:
                    text_ids = step["text"][0].tolist()
                    decoded_text = (
                        tokenizer.decode(text_ids).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " ")
                    )
                    for token in step["decoded_tokens"]:
                        token["decoded_token"] = tokenizer.decode([token["token_id"]])
                    step_with_decoded = dict(step)
                    step_with_decoded["decoded_text"] = decoded_text
                    del step_with_decoded["text"]
                    step_with_decoded = tensor_to_python(step_with_decoded)
                    decoded_history.append(step_with_decoded)
                all_steps_path = os.path.join(step_save_dir, f"q{example_idx}_s{sample_idx}.json")
                with open(all_steps_path, "w") as f:
                    json.dump(decoded_history, f, indent=2)

        if example_correct:
            correct_count += 1

        used_steps_list.extend(example_steps)
        results["results"].append(
            {
                "task_id": example_idx,
                "input_prompt": prompt,
                "ground_truth_answer": ground_truth_answer,
                "any_correct": example_correct,
                "avg_steps": round(sum(example_steps) / len(example_steps), 2),
                "samples": example_samples,
            }
        )

    accuracy = correct_count / len(data) if len(data) else 0.0
    avg_steps = sum(used_steps_list) / len(used_steps_list) if used_steps_list else 0.0
    results["summary"] = {
        "accuracy": round(accuracy * 100, 2),
        "average_steps": round(avg_steps, 2),
        "total_questions": len(data),
        "correct_questions": correct_count,
        "num_samples_per_question": num_samples,
    }
    results["config"] = _build_run_config(
        alg=alg, dataset=dataset, gen_length=gen_length, steps=steps,
        less_kwargs=less_kwargs,
    )

    # Save results; if distributed, write partial per-rank and merge on rank 0
    if world_size > 1:
        os.makedirs(rank_dir, exist_ok=True)
        partial_path = os.path.join(rank_dir, f"all_results_rank{rank}.json")
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        _barrier()
        if is_main:
            merged = {"summary": {}, "results": []}
            for r in range(world_size):
                r_path = os.path.join(save_dir, f"rank{r}", f"all_results_rank{r}.json")
                if os.path.exists(r_path):
                    with open(r_path, "r") as rf:
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
            }
            merged["config"] = _build_run_config(
                alg=alg, dataset=dataset, gen_length=gen_length, steps=steps,
                less_kwargs=less_kwargs,
            )

            save_path = f"{save_dir}/all_results.json"
            with open(save_path, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"[{dataset.capitalize()}]")
            print(f"Accuracy: {merged['summary']['accuracy']}")
            print(f"Average steps: {merged['summary']['average_steps']}")
            print(f"Results saved to {save_path}")
        _barrier()
    else:
        save_path = f"{save_dir}/all_results.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{dataset.capitalize()}]")
        print(f"Accuracy: {round(accuracy * 100, 2)}")
        print(f"Average steps: {round(avg_steps, 2)}")
        print(f"Results saved to {save_path}")


# --------------------
# Evaluation: HumanEval
# --------------------
def test_humaneval(
    model,
    tokenizer,
    save_dir,
    gen_length,
    steps,
    alg="less",
    temperature=0.2,
    top_p=0.95,
    k=(1, 10, 100),
    n_workers=4,
    timeout=3.0,
    test_size=None,
    random_sampling=False,
    num_samples=1,
    save_steps=False,
    less_kwargs=None,
):
    if alg == "less":
        lk = less_kwargs or {}
        c = lk.get("conf_threshold", 0.75)
        d = lk.get("drift_threshold", 0.04)
        save_dir = (
            f"{save_dir}/Dream/humaneval/less/all/len_{gen_length}/steps_{steps}"
            f"/c{c}_d{d}_s{num_samples}"
        )
    else:
        save_dir = f"{save_dir}/Dream/humaneval/{alg}/len_{gen_length}/steps_{steps}/s{num_samples}"

    os.makedirs(save_dir, exist_ok=True)

    world_size = _get_world_size()
    rank = _get_rank()
    is_main = _is_main_process()
    rank_dir = os.path.join(save_dir, f"rank{rank}") if world_size > 1 else save_dir

    if save_steps:
        step_save_dir = os.path.join(rank_dir, "stepwise") if world_size > 1 else os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    problems = read_problems()

    if test_size:
        random.seed(516)
        problems = list(problems.items())
        data_list = random.sample(problems, test_size) if random_sampling else problems[:test_size]
    else:
        data_list = list(problems.items())

    if world_size > 1:
        data_list = data_list[rank::world_size]
    data = dict(data_list)

    samples = []
    steps_per_problem = []
    i = 0

    for task_id, info in tqdm(data.items(), desc="Generating completions for HumanEval", disable=not is_main):
        prompt = info["prompt"]
        task_samples = []
        task_steps = []

        messages = [{"role": "system", "content": "You complete only Python code."}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = inputs.input_ids.to(f"cuda:{_get_local_rank()}")
        attention_mask = inputs.attention_mask.to(f"cuda:{_get_local_rank()}")

        for sample_idx in range(num_samples):
            if alg == "less":
                lk = dict(less_kwargs or {})
                lk.setdefault("mask_token_id", tokenizer.mask_token_id)
                lk.setdefault("max_new_tokens", gen_length)
                lk.setdefault("steps", steps)
                lk.setdefault("temperature", temperature)
                lk.setdefault("top_p", top_p)
                lk.setdefault("return_stats", True)
                lk.setdefault("collect_step_trace", bool(save_steps))
                sequences, stats_list = _diffusion_generate_less(
                    model, input_ids, attention_mask=attention_mask, **lk,
                )
                used_steps = int(stats_list[0].get("steps_taken", steps))
                task_steps.append(used_steps)
                generations = [tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, sequences)]
                output = None
                if save_steps and stats_list:
                    fb = stats_list[0].get("per_step_fallback", [])
                    an = stats_list[0].get("per_step_accepted_n", [])
                    ax_ids = stats_list[0].get("per_step_argmax_ids", [])
                    step_trace = [
                        {"step": _i, "fallback": bool(fb[_i]),
                         "accepted_tokens_num": int(an[_i]) if _i < len(an) else 0,
                         "argmax_ids": (list(ax_ids[_i]) if _i < len(ax_ids) else [])}
                        for _i in range(len(fb))
                    ]
                    all_steps_path = os.path.join(step_save_dir, f"q{i}_s{sample_idx}.json")
                    with open(all_steps_path, "w") as _ftrace:
                        json.dump(step_trace, _ftrace)
            else:
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=temperature,
                    top_p=top_p,
                    alg=alg,
                    alg_temp=0.0,
                )
                used_steps = int(getattr(output, "used_steps", steps))
                task_steps.append(used_steps)
                generations = [tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, output.sequences)]

            decoded_text = generations[0].split(tokenizer.eos_token)[0]
            code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", decoded_text, re.DOTALL)
            code_only = code_match.group(1).strip() if code_match else decoded_text.strip()

            sample_data = {"task_id": task_id, "sample_idx": sample_idx, "completion": code_only, "used_steps": used_steps}
            task_samples.append(sample_data)
            samples.append(sample_data)

            if save_steps and output is not None:
                history = output.history
                decoded_history = []
                for step in history:
                    text_ids = step["text"][0].tolist()
                    decoded_step_text = (
                        tokenizer.decode(text_ids)
                        .split(tokenizer.eos_token)[0]
                        .replace(tokenizer.mask_token, " ")
                    )
                    for token in step["decoded_tokens"]:
                        token["decoded_token"] = tokenizer.decode([token["token_id"]])
                    step_with_decoded = dict(step)
                    step_with_decoded["decoded_text"] = decoded_step_text
                    del step_with_decoded["text"]
                    step_with_decoded = tensor_to_python(step_with_decoded)
                    decoded_history.append(step_with_decoded)
                all_steps_path = os.path.join(step_save_dir, f"q{i}_s{sample_idx}.json")
                with open(all_steps_path, "w") as f:
                    json.dump(decoded_history, f, indent=2)

        steps_per_problem.append(
            {
                "task_id": task_id,
                "input_prompt": prompt,
                "avg_steps": sum(task_steps) / len(task_steps) if task_steps else 0.0,
                "samples": task_samples,
            }
        )
        i += 1

    if world_size > 1:
        os.makedirs(rank_dir, exist_ok=True)
        rank_samples_file = os.path.join(rank_dir, f"humaneval_samples_rank{rank}.jsonl")
        write_jsonl(rank_samples_file, samples)
        rank_steps_file = os.path.join(rank_dir, f"humaneval_steps_rank{rank}.json")
        with open(rank_steps_file, "w") as f:
            json.dump({"results": steps_per_problem}, f, indent=2)

        _barrier()
        if is_main:
            merged_samples_file = os.path.join(save_dir, "humaneval_samples.jsonl")
            with open(merged_samples_file, "w") as out_f:
                for r in range(world_size):
                    part_file = os.path.join(save_dir, f"rank{r}", f"humaneval_samples_rank{r}.jsonl")
                    if os.path.exists(part_file):
                        with open(part_file, "r") as pf:
                            for line in pf:
                                out_f.write(line)

            merged_steps = []
            for r in range(world_size):
                part_steps = os.path.join(save_dir, f"rank{r}", f"humaneval_steps_rank{r}.json")
                if os.path.exists(part_steps):
                    with open(part_steps, "r") as psf:
                        merged_steps.extend(json.load(psf).get("results", []))

            results_eval = evaluate_functional_correctness(merged_samples_file, k=k, n_workers=n_workers, timeout=timeout)
            avg_steps = float(np.mean([entry["avg_steps"] for entry in merged_steps])) if merged_steps else 0.0
            all_results = {
                "summary": {"accuracy": round(results_eval["pass@1"] * 100, 2), "average_steps": round(avg_steps, 2)},
                "config": _build_run_config(
                    alg=alg, dataset="humaneval", gen_length=gen_length, steps=steps,
                    less_kwargs=less_kwargs,
                ),
                "results": merged_steps,
            }
            save_path = os.path.join(save_dir, "all_results.json")
            with open(save_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print("[HumanEval]")
            print("Accuracy:", round(results_eval["pass@1"] * 100, 2))
            print(f"Average steps: {avg_steps:.2f}")
            print(f"Results saved to {save_path}")
        _barrier()
    else:
        samples_file = os.path.join(save_dir, "humaneval_samples.jsonl")
        write_jsonl(samples_file, samples)
        results = evaluate_functional_correctness(samples_file, k=k, n_workers=n_workers, timeout=timeout)
        avg_steps = float(np.mean([entry["avg_steps"] for entry in steps_per_problem])) if steps_per_problem else 0.0
        all_results = {
            "summary": {"accuracy": round(results["pass@1"] * 100, 2), "average_steps": round(avg_steps, 2)},
            "config": _build_run_config(
                alg=alg, dataset="humaneval", gen_length=gen_length, steps=steps,
                less_kwargs=less_kwargs,
            ),
            "results": steps_per_problem,
        }
        save_path = os.path.join(save_dir, "all_results.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print("[HumanEval]")
        print("Accuracy:", round(results["pass@1"] * 100, 2))
        print(f"Average steps: {avg_steps:.2f}")
        print(f"Results saved to {save_path}")


# --------------------
# Evaluation: MBPP
# --------------------
def test_mbpp(
    model,
    tokenizer,
    save_dir,
    gen_length,
    steps,
    alg="less",
    temperature=0.2,
    top_p=0.95,
    eval_timeout=3.0,
    test_size=None,
    random_sampling=False,
    num_samples=1,
    save_steps=False,
    less_kwargs=None,
):
    if alg == "less":
        lk = less_kwargs or {}
        c = lk.get("conf_threshold", 0.75)
        d = lk.get("drift_threshold", 0.04)
        save_dir = (
            f"{save_dir}/Dream/mbpp/less/all/len_{gen_length}/steps_{steps}"
            f"/c{c}_d{d}_s{num_samples}"
        )
    else:
        save_dir = f"{save_dir}/Dream/mbpp/{alg}/len_{gen_length}/steps_{steps}/s{num_samples}"
    os.makedirs(save_dir, exist_ok=True)

    world_size = _get_world_size()
    rank = _get_rank()
    is_main = _is_main_process()
    rank_dir = os.path.join(save_dir, f"rank{rank}") if world_size > 1 else save_dir

    if save_steps:
        step_save_dir = os.path.join(rank_dir, "stepwise") if world_size > 1 else os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    if test_size:
        random.seed(516)
        ds = random.sample(list(ds), test_size) if random_sampling else ds.select(range(test_size))

    # Partition across ranks if distributed
    if world_size > 1:
        ds = list(ds)
        ds = ds[rank::world_size]

    steps_per_problem = []

    for idx, ex in enumerate(tqdm(ds, desc="Generating completions for MBPP", disable=not is_main)):
        task_id = ex["task_id"]
        prompt = ex["prompt"]
        tests = ex["test_list"]
        code = ex["code"]

        task_samples = []
        task_steps = []
        task_passed = False

        inputs = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        f"You are an expert Python programmer, and here is your task: {prompt} "
                        f"Your code should pass these tests:\n\n{tests}\n[BEGIN]"
                    ),
                }
            ],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device=f"cuda:{_get_local_rank()}")
        attention_mask = inputs.attention_mask.to(device=f"cuda:{_get_local_rank()}")

        for sample_idx in range(num_samples):
            if alg == "less":
                lk = dict(less_kwargs or {})
                lk.setdefault("mask_token_id", tokenizer.mask_token_id)
                lk.setdefault("max_new_tokens", gen_length)
                lk.setdefault("steps", steps)
                lk.setdefault("temperature", temperature)
                lk.setdefault("top_p", top_p)
                lk.setdefault("return_stats", True)
                lk.setdefault("collect_step_trace", bool(save_steps))
                sequences, stats_list = _diffusion_generate_less(
                    model, input_ids, attention_mask=attention_mask, **lk,
                )
                used_steps = int(stats_list[0].get("steps_taken", steps))
                task_steps.append(used_steps)
                generations = [tokenizer.decode(g[len(p):].tolist()) for p, g in zip(input_ids, sequences)]
                output = None
                if save_steps and stats_list:
                    fb = stats_list[0].get("per_step_fallback", [])
                    an = stats_list[0].get("per_step_accepted_n", [])
                    ax_ids = stats_list[0].get("per_step_argmax_ids", [])
                    step_trace = [
                        {"step": _i, "fallback": bool(fb[_i]),
                         "accepted_tokens_num": int(an[_i]) if _i < len(an) else 0,
                         "argmax_ids": (list(ax_ids[_i]) if _i < len(ax_ids) else [])}
                        for _i in range(len(fb))
                    ]
                    all_steps_path = os.path.join(step_save_dir, f"q{idx}_s{sample_idx}.json")
                    with open(all_steps_path, "w") as _ftrace:
                        json.dump(step_trace, _ftrace)
            else:
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    steps=steps,
                    temperature=temperature,
                    top_p=top_p,
                    alg=alg,
                    output_history=True,
                    return_dict_in_generate=True,
                )
                used_steps = int(getattr(output, "used_steps", steps))
                task_steps.append(used_steps)
                generations = [tokenizer.decode(g[len(p) :].tolist()) for p, g in zip(input_ids, output.sequences)]

            decoded_text = generations[0].split(tokenizer.eos_token)[0]
            sample_data = {"task_id": task_id, "sample_idx": sample_idx, "used_steps": used_steps, "generation": decoded_text}

            # Save stepwise info if requested
            if save_steps and output is not None:
                history = output.history
                decoded_history = []
                for step in history:
                    text_ids = step["text"][0].tolist()
                    decoded_step_text = (
                        tokenizer.decode(text_ids)
                        .split(tokenizer.eos_token)[0]
                        .replace(tokenizer.mask_token, " ")
                    )
                    for token in step["decoded_tokens"]:
                        token["decoded_token"] = tokenizer.decode([token["token_id"]])
                    step_with_decoded = dict(step)
                    step_with_decoded["decoded_text"] = decoded_step_text
                    del step_with_decoded["text"]
                    step_with_decoded = tensor_to_python(step_with_decoded)
                    decoded_history.append(step_with_decoded)
                all_steps_path = os.path.join(step_save_dir, f"q{idx}_s{sample_idx}.json")
                with open(all_steps_path, "w") as f:
                    json.dump(decoded_history, f, indent=2)

            # Evaluate this sample
            passed = evaluate_task(sample_data, tests, timeout=eval_timeout)
            if passed:
                task_passed = True
            sample_data["passed"] = passed
            task_samples.append(sample_data)

        # Aggregate results for this task
        steps_per_problem.append(
            {
                "task_id": task_id,
                "input_prompt": prompt,
                "solution_code": code,
                "any_passed": task_passed,
                "avg_steps": sum(task_steps) / len(task_steps) if task_steps else 0.0,
                "samples": task_samples,
            }
        )

    total_passed_tasks = sum(1 for task_data in steps_per_problem if task_data.get("any_passed"))
    accuracy = total_passed_tasks / len(steps_per_problem) if steps_per_problem else 0.0
    average_steps = float(np.mean([entry.get("avg_steps", 0.0) for entry in steps_per_problem])) if steps_per_problem else 0.0

    if world_size > 1:
        os.makedirs(rank_dir, exist_ok=True)
        partial = os.path.join(rank_dir, f"mbpp_results_rank{rank}.json")
        with open(partial, "w") as f:
            json.dump({"results": steps_per_problem}, f, indent=2)
        _barrier()
        if is_main:
            merged = []
            for r in range(world_size):
                r_path = os.path.join(save_dir, f"rank{r}", f"mbpp_results_rank{r}.json")
                if os.path.exists(r_path):
                    with open(r_path, "r") as rf:
                        merged.extend(json.load(rf).get("results", []))

            total_passed_tasks = sum(1 for t in merged if t.get("any_passed", False))
            accuracy = (total_passed_tasks / len(merged)) if merged else 0.0
            average_steps = float(np.mean([t.get("avg_steps", 0.0) for t in merged])) if merged else 0.0

            all_results = {
                "summary": {
                    "accuracy": round(accuracy * 100, 2),
                    "average_steps": round(average_steps, 2),
                    "total_tasks": len(merged),
                    "passed_tasks": total_passed_tasks,
                    "num_samples_per_task": num_samples,
                },
                "config": _build_run_config(
                    alg=alg, dataset="mbpp", gen_length=gen_length, steps=steps,
                    less_kwargs=less_kwargs,
                ),
                "results": merged,
            }
            samples_file = os.path.join(save_dir, "all_results.json")
            with open(samples_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print("[MBPP]")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Average steps: {average_steps:.2f}")
            print(f"Results saved to {samples_file}")
        _barrier()
    else:
        all_results = {
            "summary": {
                "accuracy": round(accuracy * 100, 2),
                "average_steps": round(average_steps, 2),
                "total_tasks": len(steps_per_problem),
                "passed_tasks": total_passed_tasks,
                "num_samples_per_task": num_samples,
            },
            "config": _build_run_config(
                alg=alg, dataset="mbpp", gen_length=gen_length, steps=steps,
                less_kwargs=less_kwargs,
            ),
            "results": steps_per_problem,
        }
        samples_file = os.path.join(save_dir, "all_results.json")
        with open(samples_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print("[MBPP]")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average steps: {average_steps:.2f}")
        print(f"Results saved to {samples_file}")


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Dream model on math, GSM8K, HumanEval, or MBPP.")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math", "humaneval", "mbpp"], help="Dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--gen_length", type=int, default=256, help="Generation length")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps")

    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")

    parser.add_argument("--test_size", type=int, default=None, help="Number of test examples")
    parser.add_argument("--random_sampling", action="store_true", help="Use random sampling")

    parser.add_argument(
        "--alg",
        type=str,
        choices=["less", "origin"],
        default="less",
        help="Algorithm to use",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question/task")
    parser.add_argument("--save_steps", action="store_true", help="If set, save the results of each step.")

    # LESS knobs (3 parameters: conf, drift, persist is fixed at 2)
    parser.add_argument("--less_conf", type=float, default=0.75, help="Confidence threshold for LESS")
    parser.add_argument("--less_drift", type=float, default=0.04, help="JS drift threshold for LESS")

    # Optional: arbitrary LESS kwargs via JSON (overrides CLI knobs)
    parser.add_argument(
        "--less_config",
        type=str,
        default=None,
        help="Path to JSON file containing kwargs for diffusion_generate_less (overrides CLI knobs).",
    )

    # HumanEval-specific arguments
    parser.add_argument("--humaneval_k", type=str, default="1", help="pass@k values, comma-separated")
    parser.add_argument("--humaneval_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--humaneval_timeout", type=float, default=3.0, help="Test timeout (seconds)")

    args = parser.parse_args()

    # Initialize distributed if launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not _dist_is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = _get_local_rank()
    torch.cuda.set_device(local_rank)

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(f"cuda:{local_rank}").eval()

    # Build LESS kwargs (if alg=less). Merge order (low -> high precedence):
    #   1. CLI knobs (--less_conf, --less_drift)
    #   2. --less_config JSON (explicit override)
    less_kwargs = None
    if args.alg == "less":
        less_kwargs = {
            "conf_threshold": args.less_conf,
            "drift_threshold": args.less_drift,
        }
        if args.less_config is not None:
            user_cfg = _load_less_config(args.less_config)
            _validate_less_kwargs(user_cfg)
            less_kwargs.update(user_cfg)

    if args.dataset == "humaneval":
        ks = tuple(map(int, args.humaneval_k.split(",")))
        test_humaneval(
            model,
            tokenizer,
            args.save_dir,
            num_samples=args.num_samples,
            k=ks,
            n_workers=args.humaneval_workers,
            timeout=args.humaneval_timeout,
            gen_length=args.gen_length,
            steps=args.steps,
            alg=args.alg,
            top_p=args.top_p,
            temperature=args.temperature,
            save_steps=args.save_steps,
            less_kwargs=less_kwargs,
            test_size=args.test_size,
        )
    elif args.dataset == "mbpp":
        test_mbpp(
            model,
            tokenizer,
            args.save_dir,
            gen_length=args.gen_length,
            steps=args.steps,
            alg=args.alg,
            temperature=args.temperature,
            test_size=args.test_size,
            top_p=args.top_p,
            save_steps=args.save_steps,
            num_samples=args.num_samples,
            less_kwargs=less_kwargs,
        )
    else:
        test_dataset(
            model,
            tokenizer,
            save_dir=args.save_dir,
            dataset=args.dataset,
            gen_length=args.gen_length,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg,
            test_size=args.test_size,
            random_sampling=args.random_sampling,
            num_samples=args.num_samples,
            save_steps=args.save_steps,
            less_kwargs=less_kwargs,
        )


if __name__ == "__main__":
    main()
