#!/usr/bin/env python3
"""Quick sample-generation from a V3 checkpoint. Loads model, runs prompts, prints completions."""
import argparse
import torch
import tiktoken
from src.model.gpt import GPT, GPTConfig

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    # V3 architecture config (matches the 30B run)
    config = GPTConfig(
        vocab_size=50304,
        d_model=1536, n_layers=24, n_heads=12, n_kv_heads=3,
        max_seq_len=1024,
        use_flash=True, tie_weights=True,
        use_qk_norm=True, use_xsa=True, use_diff_attn=False, use_mhc=False,
    )
    model = GPT(config)

    # pick weights: prefer EMA if present (smoother), else model_weights
    if "ema" in ckpt and ckpt["ema"] is not None:
        state = ckpt["ema"]
        print(">>> using EMA weights")
    else:
        state = ckpt["model_weights"]
        print(">>> using model_weights")

    # strip torch.compile prefix if present
    state = { k.replace("_orig_mod.", ""): v for k, v in state.items() }
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.checkpoint, args.device)

    prompts = [
        "The meaning of life is",
        "In a distant galaxy,",
        "The president announced that",
        "Here is a simple Python function to compute the factorial of a number:",
        "The three most important things to know about machine learning are",
        "def fibonacci(n):",
    ]

    for p in prompts:
        ids = torch.tensor([enc.encode(p)], device=args.device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        text = enc.decode(out[0].tolist())
        print("=" * 70)
        print(text)
    print("=" * 70)

if __name__ == "__main__":
    main()