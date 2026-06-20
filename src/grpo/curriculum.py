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
    stage_idx: int,
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
        return prev_stage_final if prev_stage_final else cc.base_checkpoint
    else:
        raise ValueError


# ---------------------------------------------------------------------------
# TODO 4 — single-stage training loop with mastery/stall checks
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

    TODOs:
        1. Build the stage GRPOConfig via make_stage_config.
        2. Load policy from policy_checkpoint (build_model with policy cfg).
           Load reference from ref_checkpoint (build_model, then freeze:
           .eval() + requires_grad_(False) on all params).
           NOTE: build_model currently loads cc.checkpoint for the model it
           builds — you'll need policy and ref to load DIFFERENT checkpoints.
           Either temporarily set config.checkpoint before each build_model
           call, or refactor build_model to take an explicit path.
        3. Create the AdamW optimizer over policy params only.
        4. (optional) wandb.init for this stage.
        5. Loop up to cc.max_steps_per_stage:
           - grpo_step -> loss, metrics  (skip if loss is None)
           - loss.backward(); clip grads; optimizer.step(); zero_grad()
           - log metrics every step
           - every cc.eval_every steps:
               * run evaluate(...) -> eval_metrics
               * append eval_metrics[cc.mastery_metric] to eval_history
               * if check_mastery(eval_history, cc): save + return ("mastered", path)
               * elif check_stall(eval_history, cc): save + return ("stalled", path)
        6. If the loop finishes without early exit, save + return ("max_steps", path).

        Save the final checkpoint as
        os.path.join(stage_dir, f"grpo_final_{stage}.pt").
    """
    # TODO: implement the single-stage loop
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 5 — curriculum driver
# ---------------------------------------------------------------------------

def run_curriculum(cc: CurriculumConfig):
    """Drive the whole curriculum: stage by stage, advancing on mastery.

    TODOs:
        - enc = tiktoken.get_encoding("gpt2")
        - policy_checkpoint starts as cc.base_checkpoint
        - prev_stage_final = None
        - for stage_idx, stage in enumerate(cc.stages):
            * stage_dir = os.path.join(cc.checkpoint_root, stage); makedirs
            * ref_checkpoint = resolve_reference_checkpoint(cc, stage_idx,
                                                            prev_stage_final)
            * outcome, final_ckpt = run_stage(cc, stage, policy_checkpoint,
                                              ref_checkpoint, stage_dir, enc)
            * print the outcome for this stage
            * if outcome == "mastered":
                - policy_checkpoint = final_ckpt   (chain into next stage)
                - prev_stage_final = final_ckpt
                - continue
              else (stalled / max_steps):
                - if cc.on_failure == "stop": print + break
                - else ("advance"): chain anyway and continue
        - print a final summary of which stages were mastered
    """
    # TODO: implement the curriculum driver
    raise NotImplementedError


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