"""GRPO training loop — skeleton with TODOs.

All the plumbing (config, model loading, wandb, argparse) is provided.
Your job: implement the five marked TODO blocks, which map directly
to the Q&A you just answered.

Run (smoke test first):
    python -m src.grpo.grpo_train \
        --checkpoint /workspace/checkpoints_sft/sft_final_v1v2.pt \
        --stage addition_easy \
        --num-prompts 2 --k 4 --max-steps 5

Full run:
    python -m src.grpo.grpo_train \
        --checkpoint /workspace/checkpoints_sft/sft_final_v1v2.pt \
        --stage addition_easy \
        --num-prompts 8 --k 8 --max-steps 500 \
        --run-name grpo_addition_easy_v1
"""

import os
import time
import argparse
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.grpo.reward import compute_reward
from src.grpo.synthetic_math import make_prompt, make_eval_set

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    checkpoint: str = "/workspace/checkpoints_sft/sft_final_v1v2.pt"
    checkpoint_dir: str = "/workspace/checkpoints_grpo"
    run_name: str = "grpo_debug"

    # Task
    stage: str = "addition_easy"

    # GRPO core
    num_prompts: int = 8        # prompts sampled per step (N)
    k: int = 8                  # completions per prompt (K)
    max_new_tokens: int = 96
    temperature: float = 0.8
    kl_beta: float = 0.02       # beta — KL penalty strength
    adv_eps: float = 1e-6       # epsilon for std guard when all rewards identical

    # Optimizer
    lr: float = 2e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: int = 500
    warmup_steps: int = 20

    # Logging / eval
    log_interval: int = 1
    eval_interval: int = 50
    save_interval: int = 100
    eval_n: int = 100
    eval_seed: int = 1234

    # V1 architecture (matches sft_final_v1v2.pt)
    vocab_size: int = 50304
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16       # V1 — no GQA
    max_seq_len: int = 1024
    use_qk_norm: bool = False
    use_diff_attn: bool = False
    use_mhc: bool = False

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Model loading — provided, no TODOs here
# ---------------------------------------------------------------------------

def build_model(config: GRPOConfig) -> GPT:
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        dropout=0.0,
        max_seq_len=config.max_seq_len,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=config.use_qk_norm,
        use_diff_attn=config.use_diff_attn,
        use_mhc=config.use_mhc,
    )
    model = GPT(model_config)
    ckpt = torch.load(config.checkpoint, weights_only=False, map_location=config.device)
    state_key = "model_weights" if "model_weights" in ckpt else "ema"
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt[state_key].items()}
    model.load_state_dict(state_dict)
    model.to(config.device)
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completions(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> list[dict]:
    """Generate K independent completions for one prompt.

    Returns a list of K dicts, each containing:
        token_ids  : LongTensor [seq_len]  — full sequence (prompt + response)
        prompt_len : int                   — number of prompt tokens
        text       : str                   — decoded response only (for reward)
    """

    # Tokenize prompt string -> list of ints, store length to slice response later
    prompt_ids = enc.encode(prompt)
    prompt_len = len(prompt_ids)          # number of prompt tokens

    completions = []

    for _ in range(k):
        # Reset to prompt at the start of each completion — each of the K
        # completions is independent, all starting from the same prompt
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device
        )  # [1, prompt_len]

        # Autoregressive inner loop — one token per forward pass
        for _ in range(max_new_tokens):

            # Crop to context window in case sequence grows too long
            input_crop = input_ids[:, -model.config.max_seq_len:]  # [1, seq_len]

            # Full forward pass, but we only need the LAST position's logits
            # because that's the distribution over the next token
            logits, _ = model(input_crop)       # [1, seq_len, vocab_size]
            logits = logits[:, -1, :]           # [1, vocab_size]
            logits = logits / temperature       # flatten (high temp) or sharpen (low temp)

            # Mask padding token ids (50257-50303) to -inf so they can never be sampled
            # Without this, enc.decode() crashes on out-of-range token ids
            logits[:, enc.n_vocab:] = -float("inf")  # [1, vocab_size]

            # Sample one token from the distribution
            probs = F.softmax(logits, dim=-1)                    # [1, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1) # [1, 1]

            # Append sampled token — becomes part of context for next forward pass
            input_ids = torch.cat([input_ids, next_token], dim=1)  # [1, seq_len+1]

            # Stop early if model naturally ends its response
            if next_token.item() == enc.eot_token:
                break

        # Drop batch dimension: [1, seq_len] -> [seq_len]
        full_ids = input_ids[0]  # [seq_len] = prompt tokens + response tokens

        # Slice off prompt tokens — keep only the response portion
        response_ids = full_ids[prompt_len:].tolist()  # list of ints, length varies

        # Strip EOT token from response if present (shouldn't be decoded as text)
        if enc.eot_token in response_ids:
            response_ids = response_ids[:response_ids.index(enc.eot_token)]

        # Decode response token ids back to string — this is what reward.py scores
        text = enc.decode(response_ids)

        completions.append({
            "token_ids": full_ids,   # [seq_len] — needed for logprob scoring in grpo_step
            "prompt_len": prompt_len, # int — needed to slice response positions in grpo_step
            "text": text,            # str — passed to compute_reward()
        })

    return completions  # list of k dicts


# ---------------------------------------------------------------------------
# Per-token log-probs
# ---------------------------------------------------------------------------

def compute_token_logprobs(
    model: GPT,
    token_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Score an already-generated sequence in ONE forward pass.

    Args:
        token_ids : LongTensor [seq_len] — full sequence (prompt + response)

    Returns:
        logprobs : FloatTensor [seq_len - 1]
            logprobs[t] = log P(token_ids[t+1] | token_ids[:t+1])
    """
    input_ids = token_ids.unsqueeze(0).to(device) # shape [1, seq_len]
    logits, _ = model(input_ids) # [1, seq_len, vocab]
    logits = logits.squeeze(0) # [seq_len, vocab]

    log_probs = F.log_softmax(logits[:-1, :], dim=-1) # Log probs over the vocab dim, we don't need the last token (as it predicts the i+1)

    # Only need the log probs of the token actually generated
    targets = token_ids[1:].to(device) # [seq_len -1]
    targets = targets.unsqueeze(-1) # [seq_len -1, 1] - gather needs 2D index

    token_logprobs = log_probs.gather(dim=-1, index=targets) # index into last dim (vocab), returns which vocab entry to pick at each position

    return token_logprobs.squeeze(-1)


# ---------------------------------------------------------------------------
# k3 KL estimator
# ---------------------------------------------------------------------------

def k3_kl(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
) -> torch.Tensor:
    """Per-token unbiased KL estimator (always >= 0).

    delta = logp_ref - logp_policy
    kl    = exp(delta) - delta - 1
    """
    delta = logp_ref - logp_policy
    return torch.exp(delta) - delta - 1


# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    policy_model: GPT,
    ref_model: GPT,
    enc: tiktoken.Encoding,
    config: GRPOConfig,
    rng: random.Random,
    device: str,
) -> tuple[torch.Tensor | None, dict | None]:
    """One GRPO training step.

    Samples config.num_prompts prompts, generates config.k completions each,
    scores them, group-normalizes advantages, and computes the GRPO loss.

    Returns (loss_tensor, metrics_dict), or (None, None) if no valid
    completions were produced this step.
    """

    policy_model.train()

    all_zero_groups = 0       # count of groups with no reward variance (no signal)
    n = 0                     # count of valid completions trained on this step

    # Split the loss into PG and KL so we can log them separately —
    # useful for debugging (want KL small, PG driving the learning)
    total_pg = 0.0
    total_kl = 0.0

    # Metric accumulators across ALL completions this step
    all_rewards = []          # every reward, for reward_mean / strict / lenient
    completion_lengths = []   # response length (in tokens) per completion

    for _ in range(config.num_prompts):
        prompt, ground_truth = make_prompt(stage=config.stage, rng=rng)  # (prompt, answer)

        # Generate K independent completions from the CURRENT policy
        completions = generate_completions(
            model=policy_model,
            enc=enc,
            prompt=prompt,
            k=config.k,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            device=device,
        )

        # Score each completion against the ground truth
        scores = []
        for completion_id in range(config.k):
            score = compute_reward(completions[completion_id]["text"], ground_truth)
            scores.append(score)
        all_rewards.extend(scores)  # stash for step-level metrics

        # Group-normalize the K rewards into advantages
        scores_mean = sum(scores) / len(scores)
        scores_stddev = (sum((s - scores_mean) ** 2 for s in scores) / len(scores)) ** 0.5

        if scores_stddev < config.adv_eps:
            # All K rewards identical -> std ~ 0 -> no signal for this prompt
            all_zero_groups += 1
            advantages = [0] * config.k
        else:
            # advantage_i = (reward_i - group_mean) / (group_std + eps)
            # positive -> push policy toward this completion; negative -> away
            advantages = [(scores[i] - scores_mean) / (scores_stddev + config.adv_eps)
                          for i in range(config.k)]

        # Each completion contributes a PG term (scaled by ITS advantage) + KL term
        for completion, advantage in zip(completions, advantages):

            # Policy log-probs — gradients flow (this is what we train)
            logp_policy = compute_token_logprobs(
                model=policy_model,
                token_ids=completion["token_ids"],
                device=device,
            )

            # Reference log-probs — frozen, no gradients (anchor for KL)
            with torch.no_grad():
                logp_ref = compute_token_logprobs(
                    model=ref_model,
                    token_ids=completion["token_ids"],
                    device=device,
                )

            # Slice to RESPONSE positions only. logprob[t] scores token[t+1],
            # so the first response token (abs index prompt_len) sits at
            # logprob index prompt_len - 1.
            resp_start = completion["prompt_len"] - 1
            logp_policy_resp = logp_policy[resp_start:]
            logp_ref_resp = logp_ref[resp_start:]

            # Skip degenerate completions with no response tokens
            if logp_policy_resp.shape[0] == 0:
                continue

            completion_lengths.append(logp_policy_resp.shape[0])

            # Policy gradient term: -A * mean(logprob), length-normalized
            pg_term = -advantage * logp_policy_resp.mean()

            # KL penalty term: beta * mean(k3_kl), keeps policy near reference
            kl_term = config.kl_beta * k3_kl(
                logp_policy=logp_policy_resp,
                logp_ref=logp_ref_resp,
            ).mean()

            total_pg = total_pg + pg_term
            total_kl = total_kl + kl_term
            n += 1

    # No valid completions this step (everything degenerate) -> signal skip
    if n == 0:
        return None, None

    # Total loss, normalized over valid completions
    loss = (total_pg + total_kl) / n

    # Step-level reward stats
    reward_mean = sum(all_rewards) / len(all_rewards)
    reward_std = (sum((r - reward_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5

    metrics = {
        "loss": loss.item(),
        "pg_loss": (total_pg / n).item(),
        "kl_loss": (total_kl / n).item(),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "all_zero_group_rate": all_zero_groups / config.num_prompts,
        "strict_rate": sum(1 for r in all_rewards if r == 1.0) / len(all_rewards),
        "lenient_rate": sum(1 for r in all_rewards if r == 0.5) / len(all_rewards),
        "completion_len_mean": sum(completion_lengths) / max(1, len(completion_lengths)),
    }
    return loss, metrics

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    policy_model: GPT,
    enc: tiktoken.Encoding,
    config: GRPOConfig,
    device: str,
) -> dict:
    """Held-out evaluation on a fixed set, low temperature.

    Uses make_eval_set(config.stage, n=config.eval_n, seed=config.eval_seed)
    — same problems every call, never seen during training.
    Generates ONE completion per prompt at temperature 0.3 (near-greedy).
    """

    strict_acc = 0
    lenient_acc = 0
    rewards = []

    # Build the fixed eval set FIRST (same seed -> same problems every call),
    # then take its length for computing rates at the end.
    eval_set = make_eval_set(stage=config.stage, n=config.eval_n, seed=config.eval_seed)
    n_eval = len(eval_set)

    for prompt, ground_truth in eval_set:
        # k=1 -> generate_completions returns a list with ONE dict.
        # [0] grabs that dict, ["text"] grabs the decoded response string.
        completion = generate_completions(
            model=policy_model,
            enc=enc,
            prompt=prompt,
            k=1,
            max_new_tokens=config.max_new_tokens,
            temperature=0.3,   # near-greedy: stable, repeatable eval
            device=device,
        )[0]["text"]

        reward = compute_reward(completion, ground_truth)
        rewards.append(reward)

        # Tally by reward tier (strict 1.0 = correct + right format,
        # lenient 0.5 = correct number, wrong format)
        if reward == 1.0:
            strict_acc += 1
        elif reward == 0.5:
            lenient_acc += 1
        # (no else needed — anything else just isn't counted as a hit)

    # "Any correct" = got the right number, in either format
    any_correct_acc = strict_acc + lenient_acc

    return {
        "eval/reward_mean": sum(rewards) / len(rewards),
        "eval/strict_acc": strict_acc / n_eval,          # fraction, not raw count
        "eval/lenient_acc": lenient_acc / n_eval,
        "eval/any_correct_acc": any_correct_acc / n_eval,
    }


# ---------------------------------------------------------------------------
# Training loop — provided, no TODOs here
# ---------------------------------------------------------------------------

def train(config: GRPOConfig):
    torch.manual_seed(42)
    device = config.device
    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(42)

    print(f"Loading policy from {config.checkpoint}")
    policy_model = build_model(config)
    policy_model.train()

    print("Loading frozen reference (same checkpoint)")
    ref_model = build_model(config)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"  Params: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  Stage={config.stage} | N={config.num_prompts} | K={config.k} | "
          f"lr={config.lr} | beta={config.kl_beta} | steps={config.max_steps}")

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    if HAS_WANDB:
        wandb.init(
            project="llm-350m-grpo",
            name=config.run_name,
            config=vars(config),
        )

    for step in range(config.max_steps):
        t0 = time.time()

        # Linear LR warmup
        lr = config.lr * min(1.0, (step + 1) / max(1, config.warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss, metrics = grpo_step(policy_model, ref_model, enc, config, rng, device)

        if loss is None:
            print(f"Step {step:>4d}: no valid completions, skipping")
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(), config.grad_clip
        )
        optimizer.step()

        dt = time.time() - t0

        if step % config.log_interval == 0:
            print(
                f"Step {step:>4d} | loss {metrics['loss']:.4f} | "
                f"pg {metrics['pg_loss']:.4f} | kl {metrics['kl_loss']:.4f} | "
                f"reward {metrics['reward_mean']:.3f} | "
                f"zero_grp {metrics['all_zero_group_rate']:.2f} | "
                f"strict {metrics['strict_rate']:.2f} | "
                f"len {metrics['completion_len_mean']:.0f} | "
                f"gn {grad_norm:.2f} | dt {dt:.1f}s"
            )
            if HAS_WANDB:
                wandb.log({
                    "train/loss": metrics["loss"],
                    "train/pg_loss": metrics["pg_loss"],
                    "train/kl_loss": metrics["kl_loss"],
                    "train/reward_mean": metrics["reward_mean"],
                    "train/reward_std": metrics["reward_std"],
                    "train/all_zero_group_rate": metrics["all_zero_group_rate"],
                    "train/strict_rate": metrics["strict_rate"],
                    "train/lenient_rate": metrics["lenient_rate"],
                    "train/completion_len_mean": metrics["completion_len_mean"],
                    "train/grad_norm": grad_norm.item()
                        if torch.is_tensor(grad_norm) else grad_norm,
                    "train/lr": lr,
                }, step=step)

        if step > 0 and step % config.eval_interval == 0:
            policy_model.eval()
            eval_metrics = evaluate(policy_model, enc, config, device)
            policy_model.train()
            print(
                f"  >>> EVAL @ {step}: "
                f"reward {eval_metrics['eval/reward_mean']:.3f} | "
                f"strict {eval_metrics['eval/strict_acc']:.3f} | "
                f"any_correct {eval_metrics['eval/any_correct_acc']:.3f}"
            )
            if HAS_WANDB:
                wandb.log(eval_metrics, step=step)

        if step > 0 and step % config.save_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                config.checkpoint_dir, f"grpo_step_{step:06d}.pt"
            )
            torch.save({
                "model_weights": policy_model.state_dict(),
                "step": step,
            }, path)
            print(f"  >>> saved {path}")

    # Final save
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(
        config.checkpoint_dir, f"grpo_final_{config.stage}.pt"
    )
    torch.save({
        "model_weights": policy_model.state_dict(),
        "step": config.max_steps,
        "config": vars(config),
    }, path)
    print(f"GRPO complete. Final: {path}")

    if HAS_WANDB:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="/workspace/checkpoints_sft/sft_final_v1v2.pt")
    p.add_argument("--checkpoint-dir", type=str,
                   default="/workspace/checkpoints_grpo")
    p.add_argument("--run-name", type=str, default="grpo_debug")
    p.add_argument("--stage", type=str, default="addition_easy",
                   choices=["addition_easy", "single_digit", "two_digit",
                            "multiplication", "percentage"])
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=100)
    args = p.parse_args()
    return GRPOConfig(
        **{k.replace("-", "_"): v for k, v in vars(args).items()}
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)