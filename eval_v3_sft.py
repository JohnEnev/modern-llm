
"""Run the full benchmark suite on the V3 SFT model.

Uses the existing harness_adapter (no changes needed there) — just passes the
V3 architecture args to build_eval_model.

Run:
  python eval_v3_sft.py 2>&1 | tee /workspace/eval_v3_sft.log
"""

import json
import lm_eval
from src.eval.harness_adapter import CustomGPTLM, build_eval_model

CHECKPOINT = "/workspace/checkpoints_sft_v3/sft_final_v3.pt"
DEVICE = "cuda"
BATCH_SIZE = 8

TASKS = [
    "lambada_openai",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "wikitext",
    "gsm8k",
]

# gsm8k needs few-shot (standard is 5-shot); the MC/LM tasks are 0-shot.
# lm_eval applies each task's default fewshot when num_fewshot is None.
NUM_FEWSHOT = None  # use each task's standard default (gsm8k=5, others=0)


def main():
    print(f"Loading V3 SFT model: {CHECKPOINT}")
    model = build_eval_model(
        CHECKPOINT,
        device=DEVICE,
        # ---- V3 architecture (override the V1 defaults) ----
        vocab_size=50304,
        d_model=1536,
        n_layers=24,
        n_heads=12,
        n_kv_heads=3,
        max_seq_len=1024,
        use_qk_norm=True,
        use_diff_attn=False,
        use_xsa=True,         
        use_ema=False,       
        use_mhc=False,
        n_streams=2,
        mhc_every_n_layers=1,
    )
    print("✓ Model loaded\n")

    lm = CustomGPTLM(model=model, device=DEVICE, batch_size=BATCH_SIZE, max_length=1024)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=NUM_FEWSHOT,
        # no limit -> full datasets, publishable numbers with tight error bars
    )

    print("\n" + "=" * 60)
    print("V3 SFT EVAL RESULTS")
    print("=" * 60)
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Save full results json for the blog / comparison table
    out_path = "/workspace/eval_v3_sft_results.json"
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2, default=str)
    print(f"\n>>> Full results saved to {out_path}")


if __name__ == "__main__":
    main()