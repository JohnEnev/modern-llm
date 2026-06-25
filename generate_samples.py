#!/usr/bin/env python3
"""Sample generation from a V3 checkpoint. Reuses training-time generate_sample logic."""
import argparse
import torch
import torch.nn.functional as F
import tiktoken
from src.model.gpt import GPT, GPTConfig

@torch.no_grad()
def generate_sample(model, enc, device, prompt, max_tokens=200, temperature=0.8):
    model.eval()
    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    real_vocab_size = enc.n_vocab  # 50257; mask the 50257..50304 padding slots
    for _ in range(max_tokens):
        input_crop = input_ids[:, -model.config.max_seq_len:]
        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature
        logits[:, real_vocab_size:] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == enc.eot_token:
            break
    return enc.decode(input_ids[0].tolist())

def load_model(checkpoint_path, device, use_ema):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = GPTConfig(
        vocab_size=50304,
        d_model=1536, n_layers=24, n_heads=12, n_kv_heads=3,
        dropout=0.0, max_seq_len=1024,
        use_flash=True, tie_weights=True,
        use_qk_norm=True, use_diff_attn=False, use_xsa=True, use_mhc=False,
    )
    model = GPT(config)
    if use_ema and "ema" in ckpt:
        state = ckpt["ema"]
        print(">>> using EMA weights")
    else:
        state = ckpt["model_weights"]
        print(">>> using raw model_weights")
    # strip torch.compile prefix (model was compiled before DDP, so keys have _orig_mod.)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--ema", action="store_true", help="use EMA weights instead of raw")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.checkpoint, args.device, args.ema)

    prompts = [
        "The meaning of life is",
        "In a distant galaxy,",
        "The president announced that",
        "Here is a simple Python function to compute the factorial of a number:",
        "The three most important things to know about machine learning are",
        "def fibonacci(n):",
    ]
    for p in prompts:
        text = generate_sample(model, enc, args.device, p,
                               max_tokens=args.max_new_tokens, temperature=args.temperature)
        print("=" * 70)
        print(text)
    print("=" * 70)

if __name__ == "__main__":
    main()