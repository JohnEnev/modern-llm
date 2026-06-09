"""Smoke test: verify the full training pipeline works end-to-end."""

import os
import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import PretrainDataset

# ---- Config ----
@dataclass
class SmokeConfig:
    data_dir: str = "data/test/train"
    val_dir: str = "data/test/val"
    seq_len: int = 128  # short for speed
    micro_batch_size: int = 2
    grad_accum_steps: int = 2
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2
    max_steps: int = 10
    log_interval: int = 1
    eval_interval: int = 5
    save_interval: int = 5
    checkpoint_dir: str = "checkpoints_smoke"
    device: str = "cpu"


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        slope = (max_lr - min_lr) / warmup_steps
        return slope * step + min_lr
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return (1 + math.cos(math.pi * progress)) * (max_lr - min_lr) / 2 + min_lr


def save_checkpoint(model, optimizer, step, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"step_{step:06d}.pt")
    checkpoint = {
        "model_weights": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "rng_state": torch.random.get_rng_state(),
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_weights"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.random.set_rng_state(checkpoint["rng_state"])
    return checkpoint["step"]


@torch.no_grad()
def evaluate(model, val_dataset, config, device, num_batches=5):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=config.micro_batch_size, shuffle=False, drop_last=True)
    total_loss = 0.0
    count = 0
    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids, targets)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def smoke_test():
    config = SmokeConfig()
    device = config.device
    torch.manual_seed(42)

    print("=" * 60)
    print("SMOKE TEST — Full Training Pipeline")
    print("=" * 60)

    # ---- Tiny model ----
    model_config = GPTConfig(
        vocab_size=32768,
        d_model=128,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_seq_len=128,
        use_flash=False,
        tie_weights=True,
    )
    model = GPT(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created: {total_params:,} params (tiny config for testing)")

    # ---- Check initial loss ----
    expected_init_loss = math.log(model_config.vocab_size)
    print(f"  Expected initial loss: {expected_init_loss:.2f} (= ln({model_config.vocab_size}))")

    # ---- Data ----
    train_dataset = PretrainDataset(config.data_dir, config.seq_len)
    val_dataset = PretrainDataset(config.val_dir, config.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.micro_batch_size, shuffle=True, drop_last=True)
    train_iter = iter(train_loader)
    print(f"✓ Dataset loaded: {len(train_dataset)} train chunks, {len(val_dataset)} val chunks")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )
    print("✓ Optimizer created")

    # ---- Training loop ----
    print(f"\n--- Training for {config.max_steps} steps ---\n")
    model.train()
    losses = []

    for step in range(config.max_steps):
        t_start = time.time()

        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        running_loss = 0.0

        for micro_step in range(config.grad_accum_steps):
            try:
                batch_inputs, batch_targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_inputs, batch_targets = next(train_iter)

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            _, loss = model(batch_inputs, batch_targets)
            loss = loss / config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        step_time = time.time() - t_start
        losses.append(running_loss)

        if step % config.log_interval == 0:
            print(
                f"Step {step:>3d} | Loss {running_loss:.4f} | "
                f"LR {lr:.2e} | Grad Norm {grad_norm:.2f} | dt {step_time*1000:.0f}ms"
            )

        # Eval
        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, config, device)
            print(f"  >>> val_loss: {val_loss:.4f}")
            model.train()

        # Checkpoint
        if step > 0 and step % config.save_interval == 0:
            ckpt_path = save_checkpoint(model, optimizer, step, config)
            print(f"  >>> Saved: {ckpt_path}")

    # ---- Checks ----
    print("\n" + "=" * 60)
    print("CHECKS")
    print("=" * 60)

    # 1. Initial loss sanity
    init_loss = losses[0]
    diff = abs(init_loss - expected_init_loss)
    status = "✓" if diff < 1.0 else "✗"
    print(f"{status} Initial loss: {init_loss:.4f} (expected ~{expected_init_loss:.2f}, diff={diff:.2f})")

    # 2. Loss decreased
    status = "✓" if losses[-1] < losses[0] else "✗"
    print(f"{status} Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")

    # 3. Gradient flow
    print("\n  Gradient check (after last step):")
    all_have_grad = True
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"  ✗ {name}: NO GRADIENT")
            all_have_grad = False
    if all_have_grad:
        print("  ✓ All parameters have gradients")

    # 4. Checkpoint save/load
    ckpt_path = save_checkpoint(model, optimizer, config.max_steps, config)
    model2 = GPT(model_config).to(device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=config.max_lr)
    loaded_step = load_checkpoint(ckpt_path, model2, opt2)
    status = "✓" if loaded_step == config.max_steps else "✗"
    print(f"{status} Checkpoint save/load works (loaded step {loaded_step})")

    # 5. Eval function works
    val_loss = evaluate(model, val_dataset, config, device)
    status = "✓" if val_loss > 0 else "✗"
    print(f"{status} Evaluation works: val_loss={val_loss:.4f}")

    # 6. Overfit test
    print("\n--- Overfit test (50 steps on single batch) ---")
    model_overfit = GPT(model_config).to(device)
    opt_overfit = torch.optim.AdamW(model_overfit.parameters(), lr=1e-3)
    single_x, single_y = next(iter(train_loader))
    single_x, single_y = single_x.to(device), single_y.to(device)

    model_overfit.train()
    first_loss = None
    for s in range(50):
        opt_overfit.zero_grad()
        _, loss = model_overfit(single_x, single_y)
        loss.backward()
        opt_overfit.step()
        if s == 0:
            first_loss = loss.item()
        if s % 10 == 0:
            print(f"  step {s:>3d}: loss={loss.item():.4f}")

    final_loss = loss.item()
    status = "✓" if final_loss < first_loss * 0.5 else "✗"
    print(f"{status} Overfit test: {first_loss:.4f} → {final_loss:.4f}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)

    # Cleanup
    import shutil
    if os.path.exists(config.checkpoint_dir):
        shutil.rmtree(config.checkpoint_dir)


if __name__ == "__main__":
    smoke_test()
