"""
Three-way comparison: base pretrained model vs original SFT vs math-heavy combined SFT.

Run on the H100 pod where all three checkpoints are accessible
(should be /workspace/step_020000.pt, /workspace/checkpoints_sft/sft_final.pt,
and /workspace/checkpoints_sft/sft_final_v1v2.pt).
"""

import torch
import tiktoken
from src.model.gpt import GPT, GPTConfig

enc = tiktoken.get_encoding("gpt2")
model_config = GPTConfig(use_qk_norm=False, use_diff_attn=False, use_mhc=False, n_kv_heads=16)

# Base model uses raw prompts (no instruction format — it was never trained on chat format)
base_prompts = [
    "The capital of France is",
    "Here is a Python function to reverse a string:\n\ndef reverse_string(s):",
    "15% of 80 is",
    "A transformer model is",
    "Rainbows are caused by",
    "A store sells apples for $3 each. If I buy 7 apples, I spend",
    "27 + 48 =",
    "A train travels 60 miles per hour for 3 hours. It travels",
]

# SFT models use instruction format
sft_prompts = [
    "User: What is the capital of France?\nAssistant:",
    "User: Write a Python function to reverse a string.\nAssistant:",
    "User: What is 15% of 80?\nAssistant:",
    "User: Explain what a transformer model is.\nAssistant:",
    "User: What causes rainbows?\nAssistant:",
    "User: A store sells apples for $3 each. If I buy 7 apples, how much do I spend?\nAssistant:",
    "User: What is 27 + 48?\nAssistant:",
    "User: If a train travels 60 miles per hour for 3 hours, how far does it go?\nAssistant:",
]


def run_model(ckpt_path, label, prompts, is_base=False):
    print(f"\n{'='*70}")
    print(f"{label}")
    print('='*70)

    model = GPT(model_config)
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_weights"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    for prompt in prompts:
        tokens = enc.encode(prompt)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100, temperature=0.4)
        output_tokens = output[0].tolist()[len(tokens):]
        eot = enc.eot_token
        if eot in output_tokens:
            output_tokens = output_tokens[:output_tokens.index(eot)]
        print(f"\n>>> {prompt}")
        print(enc.decode(output_tokens))
        print("---")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    run_model("/workspace/step_020000.pt", "1. BASE MODEL (pretrained only, no SFT)", base_prompts, is_base=True)
    run_model("/workspace/checkpoints_sft/sft_final.pt", "2. ORIGINAL SFT (V1, general-heavy: 600K OpenHermes + 300K MetaMath + 20K CodeAlpaca)", sft_prompts)
    run_model("/workspace/checkpoints_sft/sft_final_v1v2.pt", "3. MATH-HEAVY SFT (combined: 600K OpenHermes + 395K MetaMath + 7.5K GSM8K + 20K CodeAlpaca)", sft_prompts)