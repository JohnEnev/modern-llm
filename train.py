# train.py
"""Training loop for 350M parameter language model."""

import os
import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import PretrainDataset

torch.set_float32_matmul_precision('high') # for torch optimization

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/fineweb-edu"
    seq_len: int = 1024

    # Batch size
    micro_batch_size: int = 16       # how many sequences per forward pass
    grad_accum_steps: int = 32       # how many micro-steps before optimizer update
    # 1024 * 16 * 32 = 524,288, so about 500k tokens per step

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5             # 10% of max_lr usually
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 20000           # approx total_tokens / tokens_per_step

    # Logging and Checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Hardware
    device: str = "cuda"

    # Evaluation
    eval_interval: int = 500
    val_dir: str = "data/fineweb-edu/val"

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


def save_checkpoint(model, optimizer, step, config):
    """Save everything needed to resume training."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"step_{step:06d}.pt")
    checkpoint = {
        "model_weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    torch.save(checkpoint, path)
    print(f"  >>> Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer):
    """Resume training from a checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_weights"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
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
def generate_sample(model, enc, device, prompt="The", max_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    # Set model to eval mode
    model.eval()

    # Encode prompt into a tensor and on device
    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Loop through max_tokens
    for i in range(max_tokens):
        # Crop to context window
        input_crop = input_ids[:, -model.config.max_seq_len:]

        # Go through forward pass
        logits, _ = model(input_crop)

        # Only take last position and add temperature
        logits = logits[:, -1, :] / temperature

        # Sample from the logits
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to input ids
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == enc.eot_token:
            break

    # Return the decoded answer
    return enc.decode(input_ids[0].tolist())


def train(config: TrainConfig):
    # Initial setup
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = config.device
    dtype = torch.bfloat16
    enc = tiktoken.get_encoding("gpt2")
    val_dataset = PretrainDataset(config.val_dir, config.seq_len)

    # Model creation, using default GPTConfig for now
    model_config = GPTConfig()
    model = GPT(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Compiling model...")
    model = torch.compile(model)
    print("✓ Model compiled")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    print(f"✓ Optimizer ready")
    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"  Effective batch: {config.micro_batch_size * config.grad_accum_steps * config.seq_len:,} tokens/step")
    print(f"  LR: {config.min_lr} → {config.max_lr} → {config.min_lr}")
    print(f"  Warmup: {config.warmup_steps} steps")
    print()

    # Training loop
    model.train()
    for step in range(config.max_steps):
        t_start = time.time()

        # Get the learning rate for the specific step
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Zero gradients
        optimizer.zero_grad()

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

        # Optimizer step
        optimizer.step()

        # Timing
        step_time = time.time() - t_start
        tokens_per_sec = (config.seq_len * config.micro_batch_size * config.grad_accum_steps) / step_time

        # Logging
        if step % config.log_interval == 0:
            print(
                f"Step {step:>6d} | "
                f"Loss {running_loss:.4f} | "
                f"LR {lr:.2e} | "
                f"Grad Norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"dt {step_time*1000:.0f}ms"
            )

        # Checkpointing
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(model, optimizer, step, config)

        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, config, device, dtype)
            print(f"  >>> val_loss: {val_loss:.4f}")
            for prompt in EVAL_PROMPTS:
                sample = generate_sample(model, enc, device, prompt=prompt)
                print(f"  >>> [{prompt}] {sample[:150]}")
                # Go back to train mode
            model.train()


    



if __name__ == "__main__":
    config = TrainConfig()
    train(config)