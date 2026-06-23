"""Full benchmark eval for the four 350M checkpoints: V1/V2 x SFT/GRPO.

Runs the standard 0-shot suite + GSM8K (5-shot) on each checkpoint, saves
per-checkpoint JSON results, and prints a final comparison table.

Run (from repo root), logging to file AND console:
    python -u eval_all.py 2>&1 | tee /workspace/eval_results/eval_all_$(date +%Y%m%d_%H%M%S).log

The -u flag (unbuffered) + tee means you see progress live AND get a saved log.

Notes:
  - All four checkpoints have only 'model_weights' (no EMA), so use_ema=False
    everywhere -> uniform, fair comparison (all raw weights).
  - V2 checkpoints need arch overrides (GQA n_kv_heads=4, QK-Norm, DiffAttn);
    V1 uses defaults. Muon is an optimizer, not arch, so it's irrelevant here.
  - GRPO checkpoints were fine-tuned on a narrow math curriculum, so expect them
    to score LOWER than SFT on general tasks (HellaSwag/LAMBADA/PIQA) -- that's
    catastrophic forgetting, an expected and reportable finding, not a bug.
  - The clean V1-vs-V2 architecture comparison is the SFT pair (same recipe,
    same data, only architecture differs).
"""

import os
import sys
import json
import time
import traceback

import torch
import lm_eval

from src.eval.harness_adapter import CustomGPTLM, build_eval_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "/workspace/eval_results"

# Standard 0-shot suite (multiple-choice + perplexity)
TASKS_0SHOT = [
    "lambada_openai",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "wikitext",
]

# GSM8K runs separately at its conventional 5-shot
GSM8K_FEWSHOT = 5

V2_ARCH = {"n_kv_heads": 4, "use_qk_norm": True, "use_diff_attn": True}

# (label, checkpoint_path, arch_overrides)
CONFIGS = [
    ("v1_sft",
     "/workspace/checkpoints_sft/sft_final_v1v2.pt",
     {}),
    ("v2_sft",
     "/workspace/checkpoints_sft_v2/sft_final_v2.pt",
     V2_ARCH),
    ("v1_grpo",
     "/workspace/checkpoints_grpo_v1/percentage/grpo_final_percentage.pt",
     {}),
    ("v2_grpo",
     "/workspace/checkpoints_grpo_v2/percentage/grpo_final_percentage.pt",
     V2_ARCH),
]

DEVICE = "cuda"
BATCH_SIZE = 16
MAX_LENGTH = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg):
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def summarize(results_dict):
    """Pull the headline metric per task into a flat {task: value} dict."""
    flat = {}
    for task, m in results_dict.items():
        if "acc_norm,none" in m:
            flat[task] = ("acc_norm", m["acc_norm,none"])
        elif "acc,none" in m:
            flat[task] = ("acc", m["acc,none"])
        elif "word_perplexity,none" in m:
            flat[task] = ("word_ppl", m["word_perplexity,none"])
        elif "exact_match,flexible-extract" in m:
            flat[task] = ("em_flex", m["exact_match,flexible-extract"])
        elif "exact_match,strict-match" in m:
            flat[task] = ("em_strict", m["exact_match,strict-match"])
        else:
            # fallback: first numeric-looking metric
            for k, v in m.items():
                if k.endswith(",none") and isinstance(v, (int, float)):
                    flat[task] = (k, v)
                    break
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"0-shot tasks: {TASKS_0SHOT}", flush=True)
    print(f"GSM8K: {GSM8K_FEWSHOT}-shot", flush=True)

    all_summaries = {}   # label -> flat summary (for the final table)
    overall_start = time.time()

    for label, ckpt, arch in CONFIGS:
        banner(f"CHECKPOINT: {label}")
        print(f"path: {ckpt}", flush=True)
        print(f"arch overrides: {arch if arch else '(V1 defaults)'}", flush=True)

        if not os.path.exists(ckpt):
            print(f"!! SKIP {label}: checkpoint not found", flush=True)
            continue

        t_ckpt = time.time()

        try:
            print("Loading model...", flush=True)
            model = build_eval_model(ckpt, device=DEVICE, use_ema=False, **arch)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  loaded OK | params: {n_params:,}", flush=True)

            lm = CustomGPTLM(model=model, device=DEVICE,
                             batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

            # --- 0-shot suite ---
            print(f"\n[{label}] running 0-shot suite ({len(TASKS_0SHOT)} tasks)...", flush=True)
            t0 = time.time()
            r0 = lm_eval.simple_evaluate(
                model=lm,
                tasks=TASKS_0SHOT,
                num_fewshot=0,
            )
            print(f"[{label}] 0-shot done in {(time.time()-t0)/60:.1f} min", flush=True)

            # --- GSM8K 5-shot ---
            print(f"\n[{label}] running GSM8K ({GSM8K_FEWSHOT}-shot)...", flush=True)
            t1 = time.time()
            rg = lm_eval.simple_evaluate(
                model=lm,
                tasks=["gsm8k"],
                num_fewshot=GSM8K_FEWSHOT,
            )
            print(f"[{label}] GSM8K done in {(time.time()-t1)/60:.1f} min", flush=True)

            # --- save ---
            out = {
                "label": label,
                "checkpoint": ckpt,
                "arch": arch,
                "n_params": n_params,
                "0shot": r0["results"],
                "gsm8k_5shot": rg["results"],
            }
            out_path = os.path.join(OUTPUT_DIR, f"{label}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"\n[{label}] saved -> {out_path}", flush=True)

            # --- per-checkpoint summary print ---
            merged = dict(r0["results"])
            merged.update(rg["results"])
            flat = summarize(merged)
            all_summaries[label] = flat
            print(f"\n[{label}] SUMMARY:", flush=True)
            for task, (metric, val) in flat.items():
                try:
                    print(f"    {task:<18} {metric:<10} {float(val):.4f}", flush=True)
                except (TypeError, ValueError):
                    print(f"    {task:<18} {metric:<10} {val}", flush=True)

        except Exception as e:
            print(f"\n!! ERROR evaluating {label}: {e}", flush=True)
            traceback.print_exc()
            print("Continuing to next checkpoint...", flush=True)
            continue
        finally:
            # free GPU between checkpoints
            try:
                del model, lm
            except Exception:
                pass
            torch.cuda.empty_cache()

        print(f"\n[{label}] total time: {(time.time()-t_ckpt)/60:.1f} min", flush=True)

    # ---------------------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------------------
    banner("FINAL COMPARISON TABLE")
    if not all_summaries:
        print("No results produced.", flush=True)
        return

    # collect all task names that appeared
    all_tasks = []
    for flat in all_summaries.values():
        for t in flat:
            if t not in all_tasks:
                all_tasks.append(t)

    labels = list(all_summaries.keys())
    header = f"{'task':<18}" + "".join(f"{lab:>14}" for lab in labels)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for task in all_tasks:
        row = f"{task:<18}"
        for lab in labels:
            cell = all_summaries[lab].get(task)
            if cell is None:
                row += f"{'-':>14}"
            else:
                _, val = cell
                try:
                    row += f"{float(val):>14.4f}"
                except (TypeError, ValueError):
                    row += f"{str(val):>14}"
        print(row, flush=True)

    # combined dump
    combined_path = os.path.join(OUTPUT_DIR, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nCombined summary -> {combined_path}", flush=True)

    print(f"\nTOTAL WALL TIME: {(time.time()-overall_start)/60:.1f} min", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()