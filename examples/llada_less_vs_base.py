#!/usr/bin/env python
"""
examples/llada_less_vs_base.py

Side-by-side comparison of vanilla fixed-schedule block decoding vs. LESS on
LLaDA-8B. Both decoders share the same (steps, block_length) budget. Vanilla
commits a fixed top-k-by-confidence set every step and always spends the full
budget; LESS commits only positions that pass the joint confidence /
persistence / drift rule and exits each block as soon as it is resolved.

The vanilla baseline is inlined here so the example is fully self-contained
(it mirrors `eval/eval_llada/llada_evaluation.py:_origin_generate`).

Usage:
    python examples/llada_less_vs_base.py
    python examples/llada_less_vs_base.py --prompt "Write a haiku about autumn." --steps 256
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoTokenizer

from llada_sampling import LLADA_MASK_TOKEN_ID, stable_less_decode
from llada_sampling.llada_less import add_gumbel_noise, get_num_transfer_tokens


@torch.no_grad()
def origin_generate(model, input_ids, gen_length, steps, block_length,
                    temperature=0.0, mask_id=LLADA_MASK_TOKEN_ID):
    """Vanilla LLaDA block-wise decoding (fixed schedule, no early exit)."""
    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()
    Lp = input_ids.shape[1]
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    used = 0
    for blk in range(num_blocks):
        bs, be = Lp + blk * block_length, Lp + (blk + 1) * block_length
        ntt = get_num_transfer_tokens((x[:, bs:be] == mask_id), steps_per_block)
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
            conf = torch.where(mask_index, probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1), -torch.inf)
            k = max(1, min(int(ntt[0, step].item()), int(mask_index.sum().item())))
            _, sel = torch.topk(conf.view(-1), k=k)
            transfer = torch.zeros_like(x.view(-1), dtype=torch.bool)
            transfer[sel] = True
            x = torch.where(transfer.view(x.shape), x0, x)
            used += 1
    return x, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--steps", type=int, default=256, help="Step budget given to BOTH decoders")
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--less_conf", type=float, default=0.75)
    ap.add_argument("--less_drift", type=float, default=0.04)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found — LLaDA-8B on CPU will be extremely slow.\n")

    print(f"Loading {args.model_path} on {device} ...")
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    torch.set_grad_enabled(False)

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=False,
    )
    input_ids = torch.tensor(tokenizer(prompt_text)["input_ids"], device=device).unsqueeze(0)
    prompt_len = input_ids.shape[1]

    def synced_timer():
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # Warm up.
    _ = origin_generate(model, input_ids, gen_length=args.block_length, steps=args.block_length,
                        block_length=args.block_length, mask_id=LLADA_MASK_TOKEN_ID)

    # 1) Vanilla fixed-schedule block decoding.
    t0 = synced_timer()
    base, steps_base = origin_generate(
        model, input_ids, gen_length=args.gen_length, steps=args.steps,
        block_length=args.block_length, mask_id=LLADA_MASK_TOKEN_ID,
    )
    t_base = synced_timer() - t0
    text_base = tokenizer.decode(base[0, prompt_len:], skip_special_tokens=True).strip()

    # 2) LESS adaptive decoding, same budget.
    t0 = synced_timer()
    seq_less, steps_less = stable_less_decode(
        model, tokenizer, input_ids,
        gen_length=args.gen_length, steps=args.steps, block_length=args.block_length,
        conf_threshold=args.less_conf, drift_threshold=args.less_drift,
        mask_id=LLADA_MASK_TOKEN_ID,
    )
    t_less = synced_timer() - t0
    text_less = tokenizer.decode(seq_less[0, prompt_len:], skip_special_tokens=True).strip()

    bar = "─" * 68
    print(f"\n{bar}")
    print(f"Prompt: {args.prompt}")
    print(bar)
    print(f"{'':18}{'steps':>10}{'wall-clock':>14}")
    print(f"{'vanilla (origin)':18}{steps_base:>10}{t_base:>12.2f}s")
    print(f"{'LESS':18}{steps_less:>10}{t_less:>12.2f}s")
    print(bar)
    if steps_less > 0:
        print(f"step reduction : {100 * (1 - steps_less / steps_base):5.1f}%  "
              f"({steps_base} -> {steps_less})")
    if t_less > 0:
        print(f"speed-up       : {t_base / t_less:5.2f}x")
    print(bar)
    print(f"\nAnswer (vanilla): {text_base}")
    print(f"\nAnswer (LESS)   : {text_less}")


if __name__ == "__main__":
    main()
