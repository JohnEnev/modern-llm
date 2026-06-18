"""GRPO baseline script — generation only, no training.

Loads the math-SFT checkpoint, generates K completions per prompt on a
fixed eval set, scores with the reward function, and prints a summary
table. The goal is to answer two questions BEFORE writing any GRPO code:

  1. Does the model ever get arithmetic problems right?
  2. Is baseline reward high enough to make GRPO viable?

Target conditions (from feedback document):
    reward_mean > 0.05  ->  acceptable
    reward_mean > 0.15  ->  good
    all_zero_group_rate < 70%  ->  acceptable
    all_zero_group_rate < 40%  ->  good

Run:
    python -m src.grpo.baseline \
        --checkpoint /workspace/checkpoints_sft/sft_final_v1v2.pt \
        --stage single_digit \
        --n-prompts 20 \
        --k 8
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.grpo.reward import compute_reward
from src.grpo.synthetic_math import make_eval_set


# ---- Model loading -------------------------------------------------------

def load_sft_model(checkpoint_path: str, device: str) -> GPT:
    """Load a V1-architecture SFT checkpoint."""
    model_config = GPTConfig(
        vocab_size=50304,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=16,   # V1 architecture — no GQA
        dropout=0.0,
        max_seq_len=1024,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=False,
        use_diff_attn=False,
        use_mhc=False,
    )
    model = GPT(model_config)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in ckpt["model_weights"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ---- Generation ----------------------------------------------------------

@torch.no_grad()
def generate_completion(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    device: str = "cuda",
) -> str:
    """Generate one completion for a prompt string.

    Returns the RESPONSE ONLY (not the prompt), as decoded text.
    """
    real_vocab_size = enc.n_vocab  # 50257 — never sample padding tokens

    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    prompt_len = len(token_ids)

    for _ in range(max_new_tokens):
        # Crop to context window if needed
        input_crop = input_ids[:, -model.config.max_seq_len:]

        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature

        # Mask padding tokens — never sample IDs beyond real vocab
        logits[:, real_vocab_size:] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == enc.eot_token:
            break

    # Return only the generated tokens, decoded
    generated_ids = input_ids[0, prompt_len:].tolist()
    # Strip EOT if present
    if enc.eot_token in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(enc.eot_token)]

    return enc.decode(generated_ids)


# ---- Baseline evaluation -------------------------------------------------

def run_baseline(
    checkpoint_path: str,
    stage: str,
    n_prompts: int,
    k: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    eval_seed: int = 1234,
):
    enc = tiktoken.get_encoding("gpt2")
    model = load_sft_model(checkpoint_path, device)

    eval_set = make_eval_set(stage, n=n_prompts, seed=eval_seed)
    print(f"\n✓ Eval set: {n_prompts} prompts, stage='{stage}', K={k}")
    print(f"  Temperature: {temperature}, max_new_tokens: {max_new_tokens}")
    print()

    # Accumulators for summary stats
    all_rewards = []
    all_zero_groups = 0
    strict_hits = 0
    lenient_hits = 0
    format_only_hits = 0

    for prompt_idx, (prompt, ground_truth) in enumerate(eval_set):
        print(f"{'='*70}")
        print(f"Prompt {prompt_idx+1}/{n_prompts}  |  ground truth: {ground_truth}")
        print(f"{'='*70}")

        group_rewards = []

        for k_idx in range(k):
            completion = generate_completion(
                model, enc, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            reward = compute_reward(completion, ground_truth)
            group_rewards.append(reward)

            # Classify for summary stats
            if reward == 1.0:
                strict_hits += 1
            elif reward == 0.5:
                lenient_hits += 1
            elif reward == 0.05:
                format_only_hits += 1

            # Print completion (truncated) and its reward
            completion_preview = completion.replace("\n", " ").strip()[:120]
            print(f"  k={k_idx+1} | reward={reward:.2f} | {completion_preview}")

        # Group stats
        group_mean = sum(group_rewards) / k
        all_zero = all(r == 0.0 for r in group_rewards)
        if all_zero:
            all_zero_groups += 1
        all_rewards.extend(group_rewards)

        print(f"  --> group mean reward: {group_mean:.3f}  |  all-zero: {all_zero}")
        print()

    # ---- Summary ----
    total_completions = n_prompts * k
    reward_mean = sum(all_rewards) / len(all_rewards)
    all_zero_rate = all_zero_groups / n_prompts

    print(f"{'='*70}")
    print("BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Stage:               {stage}")
    print(f"  Prompts:             {n_prompts}")
    print(f"  K per prompt:        {k}")
    print(f"  Total completions:   {total_completions}")
    print()
    print(f"  reward_mean:         {reward_mean:.4f}")
    print(f"  strict_rate:         {strict_hits/total_completions:.3f}  ({strict_hits}/{total_completions} correct + right format)")
    print(f"  lenient_rate:        {lenient_hits/total_completions:.3f}  ({lenient_hits}/{total_completions} correct + wrong format)")
    print(f"  format_only_rate:    {format_only_hits/total_completions:.3f}  ({format_only_hits}/{total_completions} wrong answer + right format)")
    print(f"  all_zero_group_rate: {all_zero_rate:.3f}  ({all_zero_groups}/{n_prompts} groups with no signal)")
    print()

    # Viability check
    if reward_mean > 0.15 and all_zero_rate < 0.40:
        print("  ✓ GOOD starting conditions — GRPO should have signal")
    elif reward_mean > 0.05 and all_zero_rate < 0.70:
        print("  ⚠ ACCEPTABLE starting conditions — GRPO may work, watch closely")
    else:
        print("  ✗ POOR starting conditions — task may be too hard, consider easier stage")
        print("    Try: --stage single_digit, or lower --temperature")


# ---- Entry point ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO generation-only baseline")
    parser.add_argument("--checkpoint", type=str,
                        default="/workspace/checkpoints_sft/sft_final_v1v2.pt")
    parser.add_argument("--stage", type=str, default="single_digit",
                        choices=["single_digit", "two_digit", "multiplication", "percentage"])
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        checkpoint_path=args.checkpoint,
        stage=args.stage,
        n_prompts=args.n_prompts,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        eval_seed=args.eval_seed,
    )