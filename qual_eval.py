#!/usr/bin/env python
"""
Part 3 evals: score the four checkpoints and generate the before/after examples.

Runs on a single L4. Does two things in one pass so I only start the pod once:
  - runs the lm-eval harness (0-shot suite + gsm8k 5-shot) on V1/V2 SFT and GRPO
  - generates greedy answers to a set of simple questions, SFT vs GRPO, so I can
    see the V2 degradation instead of just reading it off the perplexity number

Pythia-410M numbers are included as a reference point. They come from the Pythia
paper (arXiv 2304.01373, Table 8), not from my own runs. If I want a reference
that went through the exact same harness as my models, --run-refs will pull
gpt2 and pythia-410m from HF and score them here instead.

    pip install lm-eval
    python run_part3_evals.py                # evals + generations
    python run_part3_evals.py --skip-evals   # generations only (quick)
    python run_part3_evals.py --skip-gen     # evals only
    python run_part3_evals.py --run-refs     # also run gpt2 + pythia-410m locally
"""

import os
import json
import time
import argparse

import torch
import torch.nn.functional as F
import tiktoken

from src.eval.harness_adapter import CustomGPTLM, build_eval_model


# ---------------------------------------------------------------------------
# Checkpoints and per-model architecture
# ---------------------------------------------------------------------------
# Paths came without a leading slash; I've written them absolute since that's
# what the rest of the project used. Fix if yours are relative.
#
# build_eval_model defaults are V1 (MHA, flags off). V2 has to override the GQA
# and the two norm/attention flags or the state dict won't load.

V1_ARCH = dict(
    d_model=1024, n_layers=24, n_heads=16, n_kv_heads=16,
    use_qk_norm=False, use_diff_attn=False, use_xsa=False,
)

V2_ARCH = dict(
    d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4,   # GQA 4:1
    use_qk_norm=True, use_diff_attn=True, use_xsa=False,
)

CHECKPOINTS = {
    "v1_sft":  dict(path="/workspace/checkpoints_v1_sft/sft_final.pt",                         arch=V1_ARCH),
    "v1_grpo": dict(path="/workspace/checkpoints_grpo_v1/percentage/grpo_final_percentage.pt", arch=V1_ARCH),
    "v2_sft":  dict(path="/workspace/checkpoints_sft_v2/sft_final_v2.pt",                       arch=V2_ARCH),
    "v2_grpo": dict(path="/workspace/checkpoints_grpo_v2/percentage/grpo_final_percentage.pt",  arch=V2_ARCH),
}

ZERO_SHOT_TASKS = ["lambada_openai", "hellaswag", "arc_easy", "arc_challenge",
                   "winogrande", "piqa", "wikitext"]
GSM8K_FEWSHOT = 5

# L4 is 24GB. 350M models are fine at 16; drop to 8 if gsm8k generation OOMs.
EVAL_BATCH_SIZE = 16

OUTPUT_DIR = "/workspace/eval_results"

# Simple questions for the before/after. All use the SFT chat template.
# Kept simple on purpose: these are the kind of prompts GRPO's arithmetic
# training could plausibly have knocked around, plus a couple of pure
# language ones to show general capability holding or breaking.
GEN_PROMPTS = [
    "What causes rainbows?",
    "What is the capital of France?",
    "Why is the sky blue?",
    "What is 7 plus 5?",
    "What is 12 times 3?",
    "Name three colors.",
    "What is the largest planet in the solar system?",
    "Write one sentence about dogs.",
    "What is 15% of 80?",
    "Who wrote Romeo and Juliet?",
]

def as_chat(question):
    return f"<|endoftext|>User: {question}<|endoftext|>Assistant:"


# ---------------------------------------------------------------------------
# Pythia-410M reference (Pythia paper, arXiv 2304.01373, Table 8, 0-shot).
# Not my runs. hellaswag isn't in that table, so it's left out rather than
# guessed. wikitext/gsm8k also left out (not comparable / not in the table).
# ---------------------------------------------------------------------------
PYTHIA_410M = {
    "lambada_openai": 0.516,
    "arc_easy": 0.521,
    "arc_challenge": 0.213,
    "winogrande": 0.537,
    "piqa": 0.668,
}


# ---------------------------------------------------------------------------
# Greedy generation (same loop as GRPO, temp 0 so the examples reproduce)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, enc, prompt, max_new_tokens=120, device="cuda"):
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = len(ids)
    for _ in range(max_new_tokens):
        crop = input_ids[:, -model.config.max_seq_len:]
        logits, _ = model(crop)
        logits = logits[:, -1, :]
        logits[:, enc.n_vocab:] = -float("inf")
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if nxt.item() == enc.eot_token:
            break
        input_ids = torch.cat([input_ids, nxt], dim=1)
    out = input_ids[0, prompt_len:].tolist()
    if enc.eot_token in out:
        out = out[:out.index(enc.eot_token)]
    return enc.decode(out).strip()


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------
def metric_of(task_result):
    """Pull the primary metric out of a task result, across harness versions."""
    for key in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                "exact_match,flexible-extract", "word_perplexity,none",
                "acc_norm", "acc", "exact_match", "word_perplexity"):
        if key in task_result:
            return task_result[key]
    for k, v in task_result.items():
        if "stderr" not in k and isinstance(v, (int, float)):
            return v
    return float("nan")


def eval_my_checkpoint(name, spec, device):
    import lm_eval
    print(f"\n{'='*70}\n{name}  {spec['path']}\n{'='*70}")
    t0 = time.time()

    model = build_eval_model(spec["path"], device=device, **spec["arch"])
    lm = CustomGPTLM(model=model, device=device, batch_size=EVAL_BATCH_SIZE, max_length=1024)

    r0 = lm_eval.simple_evaluate(model=lm, tasks=ZERO_SHOT_TASKS, num_fewshot=0)
    rg = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"], num_fewshot=GSM8K_FEWSHOT)

    merged = {}
    merged.update(r0["results"])
    merged.update(rg["results"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w") as f:
        json.dump(merged, f, indent=2, default=str)

    print(f"  {(time.time()-t0)/60:.1f} min")
    del model, lm
    torch.cuda.empty_cache()
    return merged


def eval_hf_reference(hf_name, device):
    """Run a stock HF model (gpt2, EleutherAI/pythia-410m) through the same
    harness, so its numbers come from my protocol instead of a paper."""
    import lm_eval
    print(f"\n{'='*70}\nreference: {hf_name}\n{'='*70}")
    t0 = time.time()
    r0 = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_name},dtype=float",
        tasks=ZERO_SHOT_TASKS,
        num_fewshot=0,
        batch_size=EVAL_BATCH_SIZE,
        device=device,
    )
    rg = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_name},dtype=float",
        tasks=["gsm8k"],
        num_fewshot=GSM8K_FEWSHOT,
        batch_size=EVAL_BATCH_SIZE,
        device=device,
    )
    merged = {}
    merged.update(r0["results"])
    merged.update(rg["results"])
    print(f"  {(time.time()-t0)/60:.1f} min")
    torch.cuda.empty_cache()
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-evals", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--run-refs", action="store_true",
                    help="also score gpt2 and pythia-410m through this harness")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    # ---- before/after generations ------------------------------------------
    if not args.skip_gen:
        print(f"\n{'#'*70}\n# BEFORE / AFTER GENERATIONS (greedy)\n{'#'*70}")
        for name in ("v1_sft", "v1_grpo", "v2_sft", "v2_grpo"):
            spec = CHECKPOINTS[name]
            print(f"\n----- {name} -----")
            model = build_eval_model(spec["path"], device=args.device, **spec["arch"])
            for q in GEN_PROMPTS:
                ans = generate(model, enc, as_chat(q), device=args.device)
                print(f"\nQ: {q}\nA: {ans}")
            del model
            torch.cuda.empty_cache()

    # ---- harness evals -----------------------------------------------------
    results = {}
    if not args.skip_evals:
        results = {name: eval_my_checkpoint(name, spec, args.device)
                for name, spec in CHECKPOINTS.items()}

    refs = {}
    if args.run_refs:
        for hf_name in ("gpt2", "EleutherAI/pythia-410m"):
            refs[hf_name] = eval_hf_reference(hf_name, args.device)

    if results or refs:
        # ---- table ---------------------------------------------------------
        all_tasks = ZERO_SHOT_TASKS + ["gsm8k"]
        mine = ["v1_sft", "v1_grpo", "v2_sft", "v2_grpo"]
        col = lambda s: f"{str(s):>12}"

        header_cols = list(mine)
        if args.run_refs:
            header_cols += ["gpt2", "pythia-410m"]
        else:
            header_cols += ["pythia-410m*"]

        print(f"\n{'='*100}\nFINAL COMPARISON\n{'='*100}")
        header = f"{'task':<18}" + "".join(col(c) for c in header_cols)
        print(header)
        print("-" * len(header))
        for task in all_tasks:
            row = f"{task:<18}"
            for name in mine:
                row += col(f"{metric_of(results[name].get(task, {})):.4f}")
            if args.run_refs:
                for hf_name in ("gpt2", "EleutherAI/pythia-410m"):
                    v = metric_of(refs[hf_name].get(task, {}))
                    row += col(f"{v:.4f}")
            else:
                v = PYTHIA_410M.get(task)
                row += col(f"{v:.4f}" if v is not None else "-")
            print(row)

        if not args.run_refs:
            print("\n* pythia-410m column is from the Pythia paper (0-shot), not my runs.")
            print("  hellaswag/wikitext/gsm8k blank: not in that table / not comparable.")
            print("  Use --run-refs to score gpt2 + pythia-410m through this exact harness.")

        with open(os.path.join(OUTPUT_DIR, "ALL.json"), "w") as f:
            json.dump({"mine": results, "refs": refs}, f, indent=2, default=str)
        print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()