# train_v3.py
"""Training loop for 1.508B parameter language model with DDP (V3)."""

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
from src.data.manifest_dataset import ManifestDataset
from src.optim.muon import configure_optimizers
from src.model.attention import DifferentialAttention
from src.model.mhc import MHCResidual
import ddp_utils

torch.set_float32_matmul_precision('high')

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class EMA:
    """Exponential Moving Average of model weights (eval/inference only)."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_param, train_param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(train_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 50304
    d_model: int = 2048
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: int = 4
    use_flash: bool = True
    tie_weights: bool = True
    use_qk_norm: bool = False
    use_diff_attn: bool = False
    use_mhc: bool = False
    n_streams: int = 2
    mhc_every_n_layers: int = 1
    compile_model: bool = True

    run_name: str = "debug"

    # Data — manifest drives train/val split and source mix
    manifest_path: str = "data/v3/manifest.jsonl"
    seq_len: int = 2048

    # Batch
    micro_batch_size: int = 16
    grad_accum_steps: int = 32

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    muon_lr: float = 1.5e-4
    grad_clip: float = 1.0
    use_muon: bool = True

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9995

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 40000

    # Logging / checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    checkpoint_dir: str = "/workspace/checkpoints_v3"

    device: str = "cuda"

    # Eval
    eval_interval: int = 500
    eval_use_ema: bool = False


EVAL_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The president announced that",
    "def fibonacci(n):",
]


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        slope = (max_lr - min_lr) / warmup_steps
        return slope * step + min_lr
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return (1 + math.cos(math.pi * progress)) * (max_lr - min_lr) / 2 + min_lr


def save_checkpoint(model, muon_optimizer, adamw_optimizer, step, config, ema=None, keep_last_n=3):
    """Save full training state — ONLY on the main process (rank 0).

    Inlines the rank-0 guard + DDP unwrap rather than using
    ddp_utils.save_checkpoint_main, because our checkpoint carries more
    than the helper saves (EMA, muon, RNG states).
    """
    if not ddp_utils.is_main_process():
        return  # non-main ranks skip checkpointing entirely

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"step_{step:06d}.pt")

    # Unwrap DDP so keys have no 'module.' prefix (loadable without DDP).
    raw = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_weights": raw.state_dict(),
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

    # Keep only the most recent N checkpoints.
    all_checkpoints = sorted(glob.glob(os.path.join(config.checkpoint_dir, "step_*.pt")))
    for old_ckpt in all_checkpoints[:-keep_last_n]:
        try:
            os.remove(old_ckpt)
            print(f"  >>> Removed old checkpoint: {old_ckpt}")
        except OSError as e:
            print(f"  >>> Warning: could not remove {old_ckpt}: {e}")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints)


@torch.no_grad()
def evaluate(model, val_dataset, config, device, dtype, num_batches=20):
    """Compute average validation loss."""
    model.eval()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )
    total_loss = 0.0
    n = 0
    for i, (val_inputs, val_targets) in enumerate(val_dataloader):
        if i >= num_batches:
            break
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(val_inputs, val_targets)
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


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
def print_differential_lambdas(model, step: int):
    lambdas = get_differential_lambdas(model)
    if lambdas is None:
        return None
    print(
        f"Step {step} | diff_attn lambda "
        f"min={lambdas.min().item():.4f} max={lambdas.max().item():.4f} "
        f"mean={lambdas.mean().item():.4f}"
    )
    return lambdas


@torch.no_grad()
def get_mhc_stats(model):
    raw_model = model.module if hasattr(model, "module") else model
    row_errors, col_errors = [], []
    write_mins, write_maxs, read_entropies = [], [], []
    for name, module in raw_model.named_modules():
        if isinstance(module, MHCResidual):
            A = module.mixing_matrix().detach().float()
            row_errors.append((A.sum(dim=-1) - 1).abs().max().item())
            col_errors.append((A.sum(dim=-2) - 1).abs().max().item())
            write = module.write_gates.detach().float()
            write_mins.append(write.min().item())
            write_maxs.append(write.max().item())
            read = F.softmax(module.read_logits.detach().float(), dim=0)
            read_entropies.append(-(read * (read + 1e-8).log()).sum().item())
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
    raw = model.module if hasattr(model, "module") else model  # for .config access
    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    real_vocab_size = enc.n_vocab
    for _ in range(max_tokens):
        input_crop = input_ids[:, -raw.config.max_seq_len:]
        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature
        logits[:, real_vocab_size:] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == enc.eot_token:
            break
    return enc.decode(input_ids[0].tolist())


def train(config: TrainConfig):
    # ---- DDP setup (first thing) ----
    info = ddp_utils.ddp_setup()
    rank = info["rank"]
    local_rank = info["local_rank"]
    world_size = info["world_size"]
    device = info["device"]
    is_main = info["is_main"]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    dtype = torch.bfloat16
    enc = tiktoken.get_encoding("gpt2")

    # ---- Model: create -> (compile) -> wrap (in that order) ----
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

    if is_main:
        print("=" * 60)
        print("V3 ARCHITECTURE CONFIG")
        print("=" * 60)
        print(f"Parameters:      {sum(p.numel() for p in model.parameters()):,}")
        print(f"d_model:         {model_config.d_model}")
        print(f"n_layers:        {model_config.n_layers}")
        print(f"n_heads:         {model_config.n_heads}")
        print(f"n_kv_heads:      {model_config.n_kv_heads} "
              f"{'(GQA)' if model_config.n_kv_heads < model_config.n_heads else '(no GQA)'}")
        print(f"max_seq_len:     {model_config.max_seq_len}")
        print(f"use_qk_norm:     {model_config.use_qk_norm}")
        print(f"use_diff_attn:   {model_config.use_diff_attn}")
        print(f"use_mhc:         {model_config.use_mhc}")
        if model_config.use_mhc:
            print(f"n_streams:       {model_config.n_streams}")
            print(f"mhc_every_n:     {model_config.mhc_every_n_layers}")
        print(f"world_size:      {world_size}")
        print(f"micro_batch:     {config.micro_batch_size}")
        print(f"grad_accum:      {config.grad_accum_steps}")
        eff = config.micro_batch_size * config.grad_accum_steps * config.seq_len * world_size
        print(f"batch (tokens):  {eff:,}  (incl. world_size factor)")
        print("=" * 60)

    if config.compile_model:
        model = torch.compile(model)
        if is_main:
            print("✓ Model compiled")

    # Wrap in DDP LAST (after create + compile)
    model = ddp_utils.wrap_ddp(model, local_rank, world_size)

    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

    # ---- Data ----
    dataset = ManifestDataset(config.manifest_path, split="train", seq_len=config.seq_len)
    val_dataset = ManifestDataset(config.manifest_path, split="val", seq_len=config.seq_len)
    if is_main:
        print(f"✓ Data: {len(dataset):,} train chunks, {len(val_dataset):,} val chunks")

    train_loader, sampler = ddp_utils.make_distributed_loader(
        dataset, config.micro_batch_size, rank, world_size
    )
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    train_iter = iter(train_loader)

    # ---- Optimizers ----
    if config.use_muon:
        muon_optimizer, adamw_optimizer = configure_optimizers(
            model, lr=config.max_lr, muon_lr=config.muon_lr, weight_decay=config.weight_decay
        )
    else:
        adamw_optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay,
            betas=(0.9, 0.95), fused=True
        )
        muon_optimizer = None

    # ---- Resume ----
    start_step = 0
    latest_ckpt = find_latest_checkpoint(config.checkpoint_dir)
    if latest_ckpt:
        checkpoint = torch.load(latest_ckpt, weights_only=False, map_location=device)
        raw = model.module if hasattr(model, "module") else model
        raw.load_state_dict(checkpoint["model_weights"])
        if muon_optimizer and "muon_state" in checkpoint:
            muon_optimizer.load_state_dict(checkpoint["muon_state"])
        adamw_optimizer.load_state_dict(checkpoint["adamw_state"])
        if ema and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        start_step = checkpoint["step"]
        torch.random.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        if is_main:
            print(f"✓ Resumed from {latest_ckpt} (step {start_step})")
    elif is_main:
        print("✓ Starting fresh training")

    if HAS_WANDB and is_main:
        wandb.init(project="llm-1.5b", name=config.run_name, config={
            "max_steps": config.max_steps,
            "micro_batch_size": config.micro_batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "world_size": world_size,
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

    # ---- Training loop ----
    model.train()
    muon_lr = config.muon_lr
    for step in range(start_step, config.max_steps):
        t_start = time.time()

        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for pg in adamw_optimizer.param_groups:
            pg["lr"] = lr
        if muon_optimizer:
            muon_lr = lr * (config.muon_lr / config.max_lr)
            for pg in muon_optimizer.param_groups:
                pg["lr"] = muon_lr

        if muon_optimizer:
            muon_optimizer.zero_grad()
        adamw_optimizer.zero_grad()

        running_loss = 0.0
        for micro_step in range(config.grad_accum_steps):
            try:
                batch_inputs, batch_targets = next(train_iter)
            except StopIteration:
                # epoch boundary: bump epoch + reshuffle, then restart iterator
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch_inputs, batch_targets = next(train_iter)

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(batch_inputs, batch_targets)

            loss = loss / config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if muon_optimizer:
            muon_optimizer.step()
        adamw_optimizer.step()

        if ema:
            ema.update(model)

        step_time = time.time() - t_start
        # tok/s includes world_size: all GPUs process tokens in parallel
        tokens_per_sec = (config.seq_len * config.micro_batch_size *
                          config.grad_accum_steps * world_size) / step_time

        completed_step = step + 1

        # ---- Logging (rank 0 only) ----
        if completed_step % config.log_interval == 0 and is_main:
            lambdas = get_differential_lambdas(model)
            print(
                f"Step {step:>6d} | Loss {running_loss:.4f} | LR {lr:.2e} | "
                f"Grad Norm {grad_norm:.2f} | tok/s {tokens_per_sec:,.0f} | "
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

        # ---- Checkpoint (save_checkpoint self-guards to rank 0) ----
        if completed_step % config.save_interval == 0:
            save_checkpoint(model, muon_optimizer, adamw_optimizer, completed_step, config, ema=ema)

        # ---- Eval (rank 0 only) ----
        if completed_step % config.eval_interval == 0 and is_main:
            eval_model = ema.model if (config.eval_use_ema and ema is not None) else model
            eval_name = "EMA" if (config.eval_use_ema and ema is not None) else "raw"
            print(f"  >>> evaluating {eval_name} model")
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
            model.train()

    # ---- Final save + cleanup ----
    save_checkpoint(model, muon_optimizer, adamw_optimizer, config.max_steps, config, ema=ema)
    if is_main:
        print("✓ Final checkpoint saved")
    if HAS_WANDB and is_main:
        wandb.finish()

    ddp_utils.ddp_cleanup(world_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="debug")
    parser.add_argument("--manifest-path", type=str, default="data/v3/manifest.jsonl")
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--use-qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-diff-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mhc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-streams", type=int, default=2)
    parser.add_argument("--mhc-every-n-layers", type=int, default=1)
    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints_v3")
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)