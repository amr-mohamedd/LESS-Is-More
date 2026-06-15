#!/usr/bin/env python
"""
examples/dream_less_vs_base.py

Side-by-side comparison of vanilla fixed-schedule decoding vs. LESS on Dream-7B.

Both decoders are given the same step budget. Vanilla (`alg="origin"`) always
spends the full budget; LESS stops as soon as every position has stabilized
under the joint confidence / persistence / drift rule. The script prints the
steps actually taken, wall-clock time, and the decoded answer for each so the
step + latency savings (at matched output quality) are visible directly.

Usage:
    python examples/dream_less_vs_base.py
    python examples/dream_less_vs_base.py --prompt "Write a haiku about autumn." --steps 256
"""
import argparse
import os
import sys
import time

import torch

# Allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoTokenizer

from dream_sampling import diffusion_generate_less


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--steps", type=int, default=256, help="Step budget given to BOTH decoders")
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--less_conf", type=float, default=0.75, help="LESS confidence threshold (c)")
    ap.add_argument("--less_drift", type=float, default=0.04, help="LESS drift threshold (d)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found — Dream-7B on CPU will be extremely slow.\n")

    print(f"Loading {args.model_path} on {device} ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    torch.set_grad_enabled(False)

    chat = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device) if "attention_mask" in enc else None
    prompt_len = input_ids.shape[1]

    def synced_timer():
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # Warm up so the first kernel launch / autotune cost isn't charged to run #1.
    _ = model.diffusion_generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=8, steps=8,
        alg="origin", output_history=False, return_dict_in_generate=True,
    )

    # 1) Vanilla fixed-schedule decoding — always spends the full step budget.
    t0 = synced_timer()
    base = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.gen_length,
        steps=args.steps,
        alg="origin",
        output_history=False,
        return_dict_in_generate=True,
    )
    t_base = synced_timer() - t0
    steps_base = args.steps  # vanilla never exits early
    text_base = tokenizer.decode(
        base.sequences[0, prompt_len:], skip_special_tokens=True
    ).strip()

    # 2) LESS — adaptive stability-gated decoding, same step budget.
    t0 = synced_timer()
    seq_less, stats = diffusion_generate_less(
        model,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.gen_length,
        steps=args.steps,
        conf_threshold=args.less_conf,
        drift_threshold=args.less_drift,
        mask_token_id=tokenizer.mask_token_id,
        model_type="dream",
        return_stats=True,
    )
    t_less = synced_timer() - t0
    steps_less = stats[0]["steps_taken"]
    text_less = tokenizer.decode(
        seq_less[0, prompt_len:], skip_special_tokens=True
    ).strip()

    # ── Report ──────────────────────────────────────────────────────────────
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
    print(f"\nLESS stats: steps_taken={steps_less}/{stats[0]['Tmax']}, "
          f"committed_early={stats[0]['committed_early']}")


if __name__ == "__main__":
    main()
