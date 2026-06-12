# sft.py
"""Supervised Fine-Tuning for the 350M LLM."""

import os
import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.data.sft_dataset import SFTDataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class SFTConfig:
    # Base model to start from
    base_checkpoint: str = "checkpoints/step_020000.pt"
    
    # Output
    checkpoint_dir: str = "checkpoints_sft"
    
    # Data — mix of general, math, code
    # Ratios: 60% OpenHermes, 30% MetaMath, 10% CodeAlpaca
    data_sources: list = field(default=None)  # defined in __post_init__
    
    # Training
    max_seq_len: int = 1024
    batch_size: int = 16
    grad_accum_steps: int = 4      # smaller than pretraining — SFT data is smaller
    num_epochs: int = 3
    
    # Optimizer — much lower LR than pretraining
    lr: float = 2e-5              # 15x lower than pretraining max_lr
    weight_decay: float = 0.01   # lower than pretraining — don't regularize as aggressively
    grad_clip: float = 1.0
    warmup_steps: int = 100       # short warmup — model is already trained
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    
    # Hardware
    device: str = "cuda"
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                {
                    "name": "teknium/OpenHermes-2.5",
                    "split": "train",
                    "instruction_key": "conversations",  # needs special handling
                    "max_examples": 600_000,  # 60%
                },
                {
                    "name": "meta-math/MetaMathQA",
                    "split": "train",
                    "instruction_key": "query",
                    "response_key": "response",
                    "max_examples": 300_000,  # 30%
                },
                {
                    "name": "sahil2801/CodeAlpaca-20k",
                    "split": "train",
                    "instruction_key": "instruction",
                    "response_key": "output",
                    "max_examples": 20_000,  # 100%
                },
            ]


def sft_loss(logits, targets, loss_mask):
    """Cross-entropy loss applied only to non-masked positions."""
    # Apply loss_mask: where mask is -100, set target to -100 (ignored by cross_entropy)
    masked_targets = targets.clone()
    masked_targets[loss_mask == -100] = -100
    
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        masked_targets.view(-1),
        ignore_index=-100
    )


def get_sft_lr(step, warmup_steps, total_steps, lr):
    """Linear warmup + cosine decay for SFT."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (1 + math.cos(math.pi * progress)) / 2


def sft_train(config: SFTConfig):
    """Main SFT training loop."""
    torch.manual_seed(42)
    device = config.device
    enc = tiktoken.get_encoding("gpt2")
    
    # Initialize wandb
    if HAS_WANDB:
        wandb.init(project="llm-350m-sft", config=vars(config))
    
    # Load base model
    model_config = GPTConfig()
    model = GPT(model_config).to(device)
    
    print(f"Loading base model from {config.base_checkpoint}...")
    checkpoint = torch.load(config.base_checkpoint, weights_only=False)
    state_dict = checkpoint["model_weights"]
    # Strip torch.compile prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print("✓ Base model loaded")
    
    # Compile for speed
    model = torch.compile(model)
    
    # SFT dataset
    train_dataset = SFTDataset(config.data_sources, config.max_seq_len, enc)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    
    # Total steps across epochs
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.num_epochs
    print(f"✓ Training for {config.num_epochs} epochs, {total_steps} steps")
    
    # Optimizer — single AdamW, no Muon (optional for SFT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    
    # Training loop
    model.train()
    step = 0
    train_iter = iter(train_loader)
    
    for step in range(total_steps):
        t_start = time.time()
        
        # LR schedule
        lr = get_sft_lr(step, config.warmup_steps, total_steps, config.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        optimizer.zero_grad()
        running_loss = 0.0
        
        for _ in range(config.grad_accum_steps):
            try:
                input_ids, targets, loss_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets, loss_mask = next(train_iter)
            
            input_ids = input_ids.to(device)
            targets   = targets.to(device)
            loss_mask = loss_mask.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(input_ids)
                loss = sft_loss(logits, targets, loss_mask)
            
            loss = loss / config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        step_time = time.time() - t_start
        
        if step % config.log_interval == 0:
            print(
                f"Step {step:>6d} | Loss {running_loss:.4f} | "
                f"LR {lr:.2e} | Grad Norm {grad_norm:.2f} | dt {step_time*1000:.0f}ms"
            )
            if HAS_WANDB:
                wandb.log({"train/loss": running_loss, "train/lr": lr}, step=step)
        
        if step > 0 and step % config.save_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            path = os.path.join(config.checkpoint_dir, f"sft_step_{step:06d}.pt")
            torch.save({
                "model_weights": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
            }, path)
            print(f"  >>> Saved: {path}")
    
    # Final save
    path = os.path.join(config.checkpoint_dir, "sft_final.pt")
    torch.save({"model_weights": model.state_dict(), "step": total_steps}, path)
    print(f"\nSFT complete! Final model: {path}")
    
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    config = SFTConfig()
    sft_train(config)