"""GRPO auto-curriculum runner.

Runs GRPO across a sequence of stages, advancing automatically when the
model masters each stage (mastery = eval metric above threshold for N
consecutive evals), chaining checkpoints stage-to-stage.

This wraps the existing grpo_train building blocks (build_model, grpo_step,
evaluate). Rather than calling train() once per stage manually, it drives
the loop itself so it can check mastery and decide when to advance.

Run:
    python -m src.grpo.curriculum \
        --base-checkpoint /workspace/checkpoints_sft/sft_final_v1v2.pt \
        --checkpoint-root /workspace/checkpoints_grpo_v1 \
        --run-prefix grpo_v1 \
        --ref-mode fixed_sft
"""

import os
import argparse
import random
import time
from dataclasses import dataclass, field

import torch
import tiktoken

from src.model.gpt import GPT
from src.grpo.grpo_train import (
    GRPOConfig,
    build_model,
    grpo_step,
    evaluate,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    base_checkpoint: str = "/workspace/checkpoints_sft/sft_final_v1v2.pt"
    checkpoint_root: str = "/workspace/checkpoints_grpo_v1"
    run_prefix: str = "grpo_v1"

    # Stage ladder, easiest -> hardest
    stages: list = field(default_factory=lambda: [
        "addition_easy",
        "single_digit",
        "two_digit",
        "multiplication",
        "percentage",
    ])

    # Reference model for KL: "fixed_sft" (always original SFT) or
    # "chained" (previous stage's final checkpoint)
    ref_mode: str = "fixed_sft"

    # Mastery criterion
    mastery_metric: str = "eval/strict_acc"   # which eval metric to gate on
    mastery_threshold: float = 0.85           # must reach this...
    mastery_patience: int = 3                 # ...for this many consecutive evals

    # Per-stage limits (safety exits)
    max_steps_per_stage: int = 600            # hard cap
    stall_patience: int = 6                   # evals with no improvement -> give up
    eval_every: int = 50                      # run eval every N steps

    # On stage failure (hit max_steps / stalled without mastery):
    # "stop" (halt curriculum) or "advance" (move on anyway)
    on_failure: str = "stop"

    # GRPO hyperparams passed through to each stage
    num_prompts: int = 8
    k: int = 8
    lr: float = 2e-6
    kl_beta: float = 0.02
    max_new_tokens: int = 96
    temperature: float = 0.8
    warmup_steps: int = 20

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helper: build a per-stage GRPOConfig
# ---------------------------------------------------------------------------

def make_stage_config(
    cc: CurriculumConfig,
    stage: str,
    policy_checkpoint: str,
    stage_dir: str,
) -> GRPOConfig:
    """Build the GRPOConfig for a single stage from the curriculum config."""
    return GRPOConfig(
        checkpoint=policy_checkpoint,
        checkpoint_dir=stage_dir,
        run_name=f"{cc.run_prefix}_{stage}",
        stage=stage,
        num_prompts=cc.num_prompts,
        k=cc.k,
        lr=cc.lr,
        kl_beta=cc.kl_beta,
        max_new_tokens=cc.max_new_tokens,
        temperature=cc.temperature,
        warmup_steps=cc.warmup_steps,
        max_steps=cc.max_steps_per_stage,
        eval_interval=cc.eval_every,
        device=cc.device,
    )


# ---------------------------------------------------------------------------
# Mastery check
# ---------------------------------------------------------------------------

def check_mastery(eval_history: list[float], cc: CurriculumConfig) -> bool:
    """Decide whether the stage is mastered.

    Args:
        eval_history: list of the gated metric's value at each eval so far
                      (e.g. strict_acc at eval 1, eval 2, ...)

    Returns True if the LAST `mastery_patience` evals are ALL >= threshold.
    """
    patience = cc.mastery_patience 

    if len(eval_history) < patience:
        return False
    
    last_entries = eval_history[-patience:]
    result = all(eval >= cc.mastery_threshold for eval in last_entries)

    return result


# ---------------------------------------------------------------------------
# Stall check
# ---------------------------------------------------------------------------

def check_stall(eval_history: list[float], cc: CurriculumConfig) -> bool:
    """Decide whether the stage has stalled (no improvement for too long).

    Args:
        eval_history: list of the gated metric's value at each eval so far

    Returns True if the metric hasn't improved over the last
    `stall_patience` evals (the model is stuck — stage likely too hard).
    """
    patience = cc.stall_patience

    # Need enough history: the recent window PLUS at least one prior eval
    # to establish a "best so far" baseline to compare against.
    if len(eval_history) <= patience + 1:
        return False

    pre_stall = eval_history[:-patience]    # everything before the recent window
    post_stall = eval_history[-patience:]   # the recent `patience` evals

    best_before = max(pre_stall)            # best the model achieved earlier

    # Stalled if NONE of the recent evals beat the earlier best
    return all(e <= best_before for e in post_stall)


# ---------------------------------------------------------------------------
# Reference checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_reference_checkpoint(
    cc: CurriculumConfig,
    prev_stage_final: str | None,
) -> str:
    """Decide which checkpoint the frozen KL reference should load.

    Args:
        stage_idx: index of the current stage in cc.stages
        prev_stage_final: path to the previous stage's final checkpoint,
                          or None if this is the first stage

    Returns the checkpoint path to use for the reference model.
    """

    if cc.ref_mode == "fixed_sft":
        return cc.base_checkpoint
    elif cc.ref_mode == "chained":
        return prev_stage_final if prev_stage_final is not None else cc.base_checkpoint
    else:
        raise ValueError(f"Unknown ref_mode: {cc.ref_mode!r} (expected 'fixed_sft' or 'chained')")


# ---------------------------------------------------------------------------
# Single-stage training loop with mastery/stall checks
# ---------------------------------------------------------------------------

def run_stage(
    cc: CurriculumConfig,
    stage: str,
    policy_checkpoint: str,
    ref_checkpoint: str,
    stage_dir: str,
    enc: tiktoken.Encoding,
) -> tuple[str, str]:
    """Train one stage until mastery, stall, or max_steps.

    Returns (outcome, final_checkpoint_path) where outcome is one of
    "mastered", "stalled", "max_steps".

    This is essentially the train() loop from grpo_train.py, BUT instead of
    always running max_steps, it checks mastery/stall after each eval and
    exits early when appropriate.
    """

    config = make_stage_config(cc, stage, policy_checkpoint, stage_dir)

    torch.manual_seed(42)
    device = config.device
    rng = random.Random(42)

    print(f"Loading policy from {config.checkpoint}")
    policy_model = build_model(config, policy_checkpoint)
    policy_model.train()

    print(f"Loading frozen reference from {ref_checkpoint}")
    ref_model = build_model(config, ref_checkpoint)
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

    eval_history = []

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

            eval_history.append(eval_metrics[cc.mastery_metric])

            if check_mastery(eval_history, cc):
                path = os.path.join(stage_dir, f"grpo_final_{stage}.pt")
                torch.save({
                    "model_weights": policy_model.state_dict(),
                    "step": step,
                    "config": vars(config),
                }, path)
                print(f"GRPO complete. Final: {path}")
                
                if HAS_WANDB:
                    wandb.finish()

                return ("mastered", path)

            if check_stall(eval_history, cc):
                path = os.path.join(stage_dir, f"grpo_final_{stage}.pt")
                torch.save({
                    "model_weights": policy_model.state_dict(),
                    "step": step,
                    "config": vars(config),
                }, path)
                print(f"GRPO complete. Final: {path}")

                if HAS_WANDB:
                    wandb.finish()

                return ("stalled", path)

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
    path = os.path.join(stage_dir, f"grpo_final_{stage}.pt")
    torch.save({
        "model_weights": policy_model.state_dict(),
        "step": step,
        "config": vars(config),
    }, path)
    print(f"GRPO complete. Final: {path}")

    if HAS_WANDB:
        wandb.finish()
    
    return ("max_steps", path)


# ---------------------------------------------------------------------------
# Curriculum driver
# ---------------------------------------------------------------------------

def run_curriculum(cc: CurriculumConfig):
    """Drive the whole curriculum: stage by stage, advancing on mastery."""
    enc = tiktoken.get_encoding("gpt2")
    policy_checkpoint = cc.base_checkpoint
    prev_stage_final = None

    results = []  # (stage, outcome) for the final summary

    for stage_idx, stage in enumerate(cc.stages):
        stage_dir = os.path.join(cc.checkpoint_root, stage)
        os.makedirs(stage_dir, exist_ok=True)          # Create path for stage

        ref_checkpoint = resolve_reference_checkpoint(
            cc, prev_stage_final            
        )

        print(f"\n{'='*70}\nSTAGE {stage_idx+1}/{len(cc.stages)}: {stage}\n{'='*70}")
        outcome, final_ckpt = run_stage(
            cc, stage, policy_checkpoint, ref_checkpoint, stage_dir, enc
        )
        print(f"Outcome for stage '{stage}': {outcome}")
        results.append((stage, outcome))

        if outcome == "mastered":
            # Chain this stage's weights into the next stage
            policy_checkpoint = final_ckpt
            prev_stage_final = final_ckpt
        else:
            # "stalled" or "max_steps" — Bug 3 fix: honor the on_failure flag
            print(f"Stage '{stage}' did not reach mastery (outcome: {outcome})")
            if cc.on_failure == "stop":
                print("on_failure=stop -> halting curriculum.")
                break
            else:  # "advance"
                print("on_failure=advance -> continuing anyway.")
                policy_checkpoint = final_ckpt
                prev_stage_final = final_ckpt

    # Final summary
    print(f"\n{'='*70}\nCURRICULUM SUMMARY\n{'='*70}")
    for stage, outcome in results:
        mark = "✓" if outcome == "mastered" else "✗"
        print(f"  {mark} {stage:<16} {outcome}")
    n_mastered = sum(1 for _, o in results if o == "mastered")
    print(f"\nMastered {n_mastered}/{len(cc.stages)} stages.")
    print(f"Final policy checkpoint: {policy_checkpoint}")




# ---------------------------------------------------------------------------
# Entry point — provided
# ---------------------------------------------------------------------------

def parse_args() -> CurriculumConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=str,
                   default="/workspace/checkpoints_sft/sft_final_v1v2.pt")
    p.add_argument("--checkpoint-root", type=str,
                   default="/workspace/checkpoints_grpo_v1")
    p.add_argument("--run-prefix", type=str, default="grpo_v1")
    p.add_argument("--ref-mode", type=str, default="fixed_sft",
                   choices=["fixed_sft", "chained"])
    p.add_argument("--mastery-threshold", type=float, default=0.85)
    p.add_argument("--mastery-patience", type=int, default=2)
    p.add_argument("--max-steps-per-stage", type=int, default=600)
    p.add_argument("--stall-patience", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--on-failure", type=str, default="stop",
                   choices=["stop", "advance"])
    args = p.parse_args()
    return CurriculumConfig(
        base_checkpoint=args.base_checkpoint,
        checkpoint_root=args.checkpoint_root,
        run_prefix=args.run_prefix,
        ref_mode=args.ref_mode,
        mastery_threshold=args.mastery_threshold,
        mastery_patience=args.mastery_patience,
        max_steps_per_stage=args.max_steps_per_stage,
        stall_patience=args.stall_patience,
        eval_every=args.eval_every,
        on_failure=args.on_failure,
    )


if __name__ == "__main__":
    cc = parse_args()
    run_curriculum(cc)