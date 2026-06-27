"""
V3 before/after: base pretrained model vs SFT model.

Base model gets raw-continuation prompts (it was never trained on chat format).
SFT model gets instruction-format prompts (User:/Assistant:) — same format SFT used.

Run on the pod where both checkpoints are accessible:
  base: /workspace/checkpoints_v3_700m_xsa_30b/step_028610.pt   (key: model_weights)
  sft:  /workspace/checkpoints_sft_v3/sft_final_v3.pt           (key: model_weights)
"""

import torch
import tiktoken
from src.model.gpt import GPT, GPTConfig

enc = tiktoken.get_encoding("gpt2")

# V3 architecture — must match what was trained (NOT the V1/V2 config!)
model_config = GPTConfig(
    vocab_size=50304,
    d_model=1536,
    n_layers=24,
    n_heads=12,
    n_kv_heads=3,
    dropout=0.0,
    max_seq_len=1024,
    use_flash=True,
    tie_weights=True,
    use_qk_norm=True,
    use_diff_attn=False,
    use_xsa=True,
    use_mhc=False,
)

# Base model: raw prompts (no instruction format — base only knows text continuation)
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

# SFT model: instruction format (MUST match the template SFTDataset used)
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


def run_model(ckpt_path, label, prompts):
    print(f"\n{'='*70}")
    print(label)
    print('='*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(model_config).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    # both checkpoints store raw (DDP-unwrapped / single-GPU) weights -> strip
    # _orig_mod. from compile; strip module. too for safety (harmless if absent).
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in ckpt["model_weights"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    for prompt in prompts:
        tokens = enc.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_model(
        "/workspace/checkpoints_v3_700m_xsa_30b/step_028610.pt",
        "1. V3 BASE MODEL (pretrained only, no SFT) — raw continuation prompts",
        base_prompts,
    )
    run_model(
        "/workspace/checkpoints_sft_v3/sft_final_v3.pt",
        "2. V3 SFT MODEL (instruction-tuned) — User/Assistant prompts",
        sft_prompts,
    )