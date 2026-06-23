# eval_v1_full.py
import json
import lm_eval
from src.eval.harness_adapter import CustomGPTLM, build_eval_model

CHECKPOINT = "/workspace/checkpoints_grpo_v1/percentage/grpo_final_percentage.pt"
OUTPUT = "/workspace/eval_results/v1_grpo_percentage.json"

# V1 architecture: defaults (d_model 1024, n_layers 24, n_heads 16,
# n_kv_heads 16, all flags False). Confirm these match your V1 config.
model = build_eval_model(CHECKPOINT, device="cuda")
lm = CustomGPTLM(model=model, device="cuda", batch_size=16, max_length=1024)

# 0-shot suite (full datasets, no limit)
results_0shot = lm_eval.simple_evaluate(
    model=lm,
    tasks=["lambada_openai", "hellaswag", "arc_easy", "arc_challenge",
           "winogrande", "piqa", "wikitext"],
    num_fewshot=0,
)

# GSM8K at standard 5-shot, separately
results_gsm8k = lm_eval.simple_evaluate(
    model=lm,
    tasks=["gsm8k"],
    num_fewshot=5,
)

# Merge and save
all_results = {
    "0shot": results_0shot["results"],
    "gsm8k_5shot": results_gsm8k["results"],
    "config": {
        "checkpoint": CHECKPOINT,
        "model": "V1 GRPO (percentage)",
    },
}

import os
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(json.dumps(all_results, indent=2, default=str))
print(f"\nSaved to {OUTPUT}")