# train.py
"""Training loop for 350M parameter language model."""

import os
import time
import math
import copy
import glob
import argparse
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import PretrainDataset
from src.optim.muon import configure_optimizers
from src.model.attention import DifferentialAttention
from src.model.mhc import MHCResidual

torch.set_float32_matmul_precision('high') # for torch optimization

# Logging through WandB
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

class EMA:
    """Exponential Moving Average of model weights.
    Use ema.model for eval and inference, not for training.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.model = copy.deepcopy(model)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_param, train_param in zip(self.model.parameters(), model.parameters()):
            # EMA high level is ema_param = decay * ema_param + (1 - decay) * train_param
            # Doing this in line for memory optimization
            ema_param.data.mul_(self.decay).add_(train_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 50304
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16
    use_flash: bool = True
    tie_weights: bool = True
    use_qk_norm: bool = False
    use_diff_attn: bool = False
    use_mhc: bool = False
    n_streams: int = 2
    mhc_every_n_layers: int = 1
    compile_model: bool = True

    # Run metadata
    run_name: str = "debug"

    # Data
    data_dir: str = "data/fineweb-edu/train"
    seq_len: int = 1024

    # Batch size
    micro_batch_size: int = 16       # how many sequences per forward pass
    grad_accum_steps: int = 32       # how many micro-steps before optimizer update
    # 1024 * 16 * 32 = 524,288, so about 500k tokens per step

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5             # 10% of max_lr usually
    weight_decay: float = 0.1
    muon_lr: float = 1.5e-4
    grad_clip: float = 1.0
    use_muon: bool = True

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9995

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 200           # approx total_tokens / tokens_per_step - 200 for smoke test

    # Logging and Checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    checkpoint_dir: str = "/workspace/checkpoints_v2"


    # Hardware
    device: str = "cuda"

    # Evaluation
    eval_interval: int = 500
    val_dir: str = "data/fineweb-edu/val"
    eval_use_ema: bool = False  # for short ablations, default to raw model

EVAL_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The president announced that",
    "def fibonacci(n):",
]


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        # Linear warmup from min_lr to max_lr
        slope = (max_lr - min_lr) / warmup_steps
        return slope * step + min_lr
    if step >= max_steps:
        return min_lr
    # Cosine decay from max_lr down to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return (1 + math.cos(math.pi * progress)) * (max_lr - min_lr) / 2 + min_lr


def save_checkpoint(model, muon_optimizer, adamw_optimizer, step, config, ema=None, keep_last_n=3):
    """Save everything needed to resume training.
    
    Automatically deletes old checkpoints, keeping only the most recent
    `keep_last_n` files, to avoid filling the network volume.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"step_{step:06d}.pt")
    checkpoint = {
        "model_weights": model.state_dict(),
        "adamw_state": adamw_optimizer.state_dict(),
        "step": step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    if muon_optimizer:
        checkpoint["muon_state"] = muon_optimizer.state_dict()
    if ema:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"  >>> Saved checkpoint: {path}")

    # Clean up old checkpoints, keep only the most recent N
    all_checkpoints = sorted(glob.glob(os.path.join(config.checkpoint_dir, "step_*.pt")))
    for old_ckpt in all_checkpoints[:-keep_last_n]:
        try:
            os.remove(old_ckpt)
            print(f"  >>> Removed old checkpoint: {old_ckpt}")
        except OSError as e:
            print(f"  >>> Warning: could not remove {old_ckpt}: {e}")

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the most recent checkpoint in the directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints)  # max on filename works since they're zero-padded

def load_checkpoint(path, model, muon_optimizer, adamw_optimizer):
    """Resume training from a checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_weights"])
    if muon_optimizer and "muon_state" in checkpoint:
        muon_optimizer.load_state_dict(checkpoint["muon_state"])
    adamw_optimizer.load_state_dict(checkpoint["adamw_state"])
    torch.random.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    print(f"  >>> Resumed from step {checkpoint['step']}")
    return checkpoint["step"]

@torch.no_grad()
def evaluate(model, val_dataset, config, device, dtype, num_batches=20):
    """Compute average validation loss."""
    # Set model to eval mode (no dropout, etc.)
    model.eval()

    # Loading a validation dataset (no shuffle, etc.)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    # Running loss
    total_loss = 0.0

    # Loop over batches
    for i, (val_inputs, val_targets) in enumerate(val_dataloader):
        if i >= num_batches:
            break

        # Load inputs and targets and move them to device
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)


        # Forward pass in dtype precision
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(val_inputs, val_targets)

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    return total_loss / min(num_batches, len(val_dataloader))

@torch.no_grad()
def print_differential_lambdas(model, step: int):
    lambdas = get_differential_lambdas(model)

    if lambdas is None:
        return None

    print(
        f"Step {step} | diff_attn lambda "
        f"min={lambdas.min().item():.4f} "
        f"max={lambdas.max().item():.4f} "
        f"mean={lambdas.mean().item():.4f}"
    )

    return lambdas

@torch.no_grad()
def get_differential_lambdas(model):
    raw_model = model.module if hasattr(model, "module") else model

    values = []

    for name, module in raw_model.named_modules():
        if isinstance(module, DifferentialAttention):
            lam = module.compute_lambda().detach().float().cpu().item()
            values.append(lam)

    if not values:
        return None

    return torch.tensor(values)

@torch.no_grad()
def get_mhc_stats(model):
    raw_model = model.module if hasattr(model, "module") else model

    stats = {}

    row_errors = []
    col_errors = []
    write_mins = []
    write_maxs = []
    read_entropies = []

    for name, module in raw_model.named_modules():
        if isinstance(module, MHCResidual):
            A = module.mixing_matrix().detach().float()

            row_error = (A.sum(dim=-1) - 1).abs().max().item()
            col_error = (A.sum(dim=-2) - 1).abs().max().item()

            row_errors.append(row_error)
            col_errors.append(col_error)

            write = module.write_gates.detach().float()
            write_mins.append(write.min().item())
            write_maxs.append(write.max().item())

            read = F.softmax(module.read_logits.detach().float(), dim=0)
            entropy = -(read * (read + 1e-8).log()).sum().item()
            read_entropies.append(entropy)

    if not row_errors:
        return None

    return {
        "mhc/row_error_max": max(row_errors),
        "mhc/col_error_max": max(col_errors),
        "mhc/write_gate_min": min(write_mins),
        "mhc/write_gate_max": max(write_maxs),
        "mhc/read_entropy_mean": sum(read_entropies) / len(read_entropies),
    }

@torch.no_grad()
def generate_sample(model, enc, device, prompt="The", max_tokens=100, temperature=0.8):
    model.eval()

    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    real_vocab_size = enc.n_vocab  # GPT-2 tiktoken: 50257

    for _ in range(max_tokens):
        input_crop = input_ids[:, -model.config.max_seq_len:]

        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature

        # Important: model vocab is padded to 50304, tokenizer vocab is 50257.
        # Never sample padded token IDs.
        logits[:, real_vocab_size:] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == enc.eot_token:
            break

    return enc.decode(input_ids[0].tolist())


def train(config: TrainConfig):
    # Initial setup
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = config.device
    dtype = torch.bfloat16
    enc = tiktoken.get_encoding("gpt2")

    # Model creation, using default GPTConfig for now
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        dropout=0.0,
        max_seq_len=config.seq_len,
        use_flash=config.use_flash,
        tie_weights=config.tie_weights,
        use_qk_norm=config.use_qk_norm,
        use_diff_attn=config.use_diff_attn,
        use_mhc=config.use_mhc,
        n_streams=config.n_streams,
        mhc_every_n_layers=config.mhc_every_n_layers,
    )
    model = GPT(model_config).to(device)

    # Print architecture summary
    print("="*60)
    print("V2 ARCHITECTURE CONFIG")
    print("="*60)
    print(f"Parameters:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"d_model:         {model_config.d_model}")
    print(f"n_layers:        {model_config.n_layers}")
    print(f"n_heads:         {model_config.n_heads}")
    print(f"n_kv_heads:      {model_config.n_kv_heads} {'(GQA)' if model_config.n_kv_heads < model_config.n_heads else '(no GQA)'}")
    print(f"vocab_size:      {model_config.vocab_size}")
    print(f"max_seq_len:     {model_config.max_seq_len}")
    print(f"use_flash:       {model_config.use_flash}")
    print(f"use_qk_norm:     {model_config.use_qk_norm}")
    print(f"use_diff_attn:   {model_config.use_diff_attn}")
    print(f"use_mhc:         {model_config.use_mhc}")
    if model_config.use_mhc:
        print(f"n_streams:       {model_config.n_streams}")
    if model_config.mhc_every_n_layers:
        print(f"mhc_every_n_layers:       {model_config.mhc_every_n_layers}")
    print("-"*60)
    print("TRAINING CONFIG")
    print("-"*60)
    print(f"max_steps:       {config.max_steps}")
    print(f"max_lr:          {config.max_lr}")
    print(f"muon_lr:         {config.muon_lr}")
    print(f"use_muon:        {config.use_muon}")
    print(f"use_ema:         {config.use_ema}")
    print(f"warmup_steps:    {config.warmup_steps}")
    print(f"micro_batch:     {config.micro_batch_size}")
    print(f"grad_accum:      {config.grad_accum_steps}")   
    print(f"batch (tokens):  {config.micro_batch_size * config.grad_accum_steps * config.seq_len:,}")
    print(f"checkpoint_dir:  {config.checkpoint_dir}")
    print("="*60)
    print()

    print("Compiling model...")
    if config.compile_model:
        model = torch.compile(model)
        print("✓ Model compiled")
    else:
        print("✓ Skipping torch.compile")

    # EMA
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

    # Dataset and dataloader
    dataset = PretrainDataset(config.data_dir, config.seq_len)
    val_dataset = PretrainDataset(config.val_dir, config.seq_len)
    print(f"✓ Data loaded: {len(dataset):,} train chunks, {len(val_dataset):,} val chunks")

    dataloader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    train_iter = iter(dataloader)

    # Optimizers
    if config.use_muon:
        muon_optimizer, adamw_optimizer = configure_optimizers(
            model,
            lr=config.max_lr,
            muon_lr=config.muon_lr,
            weight_decay=config.weight_decay,
        )
    else:
        adamw_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=True
        )
        muon_optimizer = None

    print(f"✓ Optimizer ready")
    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"  Effective batch: {config.micro_batch_size * config.grad_accum_steps * config.seq_len:,} tokens/step")
    print(f"  LR: {config.min_lr} → {config.max_lr} → {config.min_lr}")
    print(f"  Warmup: {config.warmup_steps} steps")
    print()

    # Resume from latest checkpoint if available
    start_step = 0
    latest_ckpt = find_latest_checkpoint(config.checkpoint_dir)
    if latest_ckpt:
        checkpoint = torch.load(latest_ckpt, weights_only=False)
        model.load_state_dict(checkpoint["model_weights"])
        if muon_optimizer:
            muon_optimizer.load_state_dict(checkpoint["muon_state"])
        adamw_optimizer.load_state_dict(checkpoint["adamw_state"])
        if ema and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        start_step = checkpoint["step"]
        torch.random.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        print(f"✓ Resumed from {latest_ckpt} (step {start_step})")
    else:
        print("✓ Starting fresh training")

    if HAS_WANDB:
        wandb.init(project="llm-350m", name=config.run_name, config={
            "max_steps": config.max_steps,
            "micro_batch_size": config.micro_batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "max_lr": config.max_lr,
            "seq_len": config.seq_len,
            "use_qk_norm": config.use_qk_norm,
            "use_diff_attn": config.use_diff_attn,
            "use_mhc": config.use_mhc,
            "n_streams": config.n_streams,
            "n_kv_heads": config.n_kv_heads,
            "use_muon": config.use_muon,
            "use_ema": config.use_ema,
        })

    # Training loop
    model.train()
    for step in range(start_step, config.max_steps):
        t_start = time.time()

        # Get the learning rate for the specific step
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for param_group in adamw_optimizer.param_groups:
            param_group["lr"] = lr
        # If using Muon
        if muon_optimizer:
            muon_lr = lr * (config.muon_lr / config.max_lr)
            for param_group in muon_optimizer.param_groups:
                param_group["lr"] = muon_lr

        # Zero gradients
        if muon_optimizer:
            muon_optimizer.zero_grad()
        adamw_optimizer.zero_grad()

        # Gradient accumulation loop (micro-steps)
        running_loss = 0.0
        for micro_step in range(config.grad_accum_steps):
            # Get batch, restart iterator if exhausted
            try:
                batch_inputs, batch_targets = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                batch_inputs, batch_targets = next(train_iter)

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass in mixed precision
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(batch_inputs, batch_targets)

            # Scale loss by accum steps so gradients average correctly
            loss = loss / config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Optimizers step
        if muon_optimizer:
            muon_optimizer.step()
        adamw_optimizer.step()

        # EMA update
        if ema:
            ema.update(model)


        # Timing
        step_time = time.time() - t_start
        tokens_per_sec = (config.seq_len * config.micro_batch_size * config.grad_accum_steps) / step_time

        completed_step = step + 1

        # Logging
        if completed_step % config.log_interval == 0:
            
            lambdas = get_differential_lambdas(model)
            
            print(
                f"Step {step:>6d} | "
                f"Loss {running_loss:.4f} | "
                f"LR {lr:.2e} | "
                f"Grad Norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"dt {step_time*1000:.0f}ms"
            )

            log_dict = {
                "train/loss": running_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "train/tokens_per_sec": tokens_per_sec,
                "train/step_time_ms": step_time * 1000,
                "train/muon_lr": muon_lr if muon_optimizer else 0.0,
            }

            if lambdas is not None:
                log_dict.update({
                    "diff_attn/lambda_min": lambdas.min().item(),
                    "diff_attn/lambda_max": lambdas.max().item(),
                    "diff_attn/lambda_mean": lambdas.mean().item(),
                })

            mhc_stats = get_mhc_stats(model)
            if mhc_stats is not None:
                log_dict.update(mhc_stats)

            if HAS_WANDB:
                wandb.log(log_dict, step=completed_step)

        # Checkpointing
        if completed_step % config.save_interval == 0:
            save_checkpoint(model, muon_optimizer, adamw_optimizer, completed_step, config, ema=ema)

        if completed_step % config.eval_interval == 0:
            if config.eval_use_ema and ema is not None:
                eval_model = ema.model
            else:
                eval_model = model
            eval_model_name = "EMA" if (config.eval_use_ema and ema is not None) else "raw"
            print(f"  >>> evaluating {eval_model_name} model")
            val_loss = evaluate(eval_model, val_dataset, config, device, dtype)
            print(f"  >>> val_loss: {val_loss:.4f}")
            lambdas = print_differential_lambdas(model, completed_step)

            if HAS_WANDB:
                eval_log = {"val/loss": val_loss}
                
                if lambdas is not None:
                    eval_log.update({
                        "diff_attn/lambda_min": lambdas.min().item(),
                        "diff_attn/lambda_max": lambdas.max().item(),
                        "diff_attn/lambda_mean": lambdas.mean().item(),
                    })
                
                wandb.log(eval_log, step=completed_step)

            for prompt in EVAL_PROMPTS:
                sample = generate_sample(eval_model, enc, device, prompt=prompt)
                print(f"  >>> [{prompt}] {sample[:150]}")
                # Go back to train mode
            model.train()

    # Final checkpoint saved        
    save_checkpoint(model, muon_optimizer, adamw_optimizer, config.max_steps, config, ema=ema)
    print("✓ Final checkpoint saved")

    if HAS_WANDB:
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="debug")

    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=32)

    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=16)

    parser.add_argument("--use-qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-diff-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mhc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-streams", type=int, default=2)
    parser.add_argument("--mhc-every-n-layers", type=int, default=1)

    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-use-ema", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints_v2")

    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    return TrainConfig(**vars(args))

if __name__ == "__main__":
    config = parse_args()
    train(config)