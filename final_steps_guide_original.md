# Final Steps Guide — From train.py to Launch

*Everything remaining before you can start training, plus post-launch improvements.*
*Format: barebone code with hints first → full solution right after.*

---

## Status Check

Done:
- ✅ Model: RMSNorm, SwiGLU, RoPE, Attention, TransformerBlock, GPT
- ✅ Tokenizer: BPE (learning) + tiktoken (production)
- ✅ Data: PretrainDataset, prepare_data.py
- ✅ Training: train.py with config, LR schedule, grad accum, checkpointing

Remaining:
1. **Evaluation loop** — val loss + text generation (Part 1)
2. **torch.compile** — one-liner, ~2x speedup (Part 2)
3. **mHC Hyper-Connections** — DeepSeek ablation (Part 3)
4. **Muon optimizer** — optimizer split (Part 4)
5. **Wandb logging** — remote monitoring (Part 5)
6. **Launch checklist** — smoke tests before GPU hours (Part 6)

---

## Part 1: Evaluation Loop

### 1A: Validation Loss

**Concepts:**
- `@torch.no_grad()` disables gradient tracking — no computation graph, less memory.
- `model.eval()` turns off dropout. Call `model.train()` after to re-enable it.
- Run ~20 batches, not the full val set — fast estimate, averages out over multiple evals.

**Data split** — hold out last 2 shards as validation:
```bash
mkdir -p data/fineweb-edu/train data/fineweb-edu/val
mv data/fineweb-edu/shard_009[89].bin data/fineweb-edu/val/
mv data/fineweb-edu/shard_*.bin data/fineweb-edu/train/
```

#### Try It

```python
@torch.no_grad()
def evaluate(model, val_dataset, config, device, dtype, num_batches=20):
    """Compute average validation loss."""
    # TODO:
    # 1. Set model to eval mode
    #    Hint: model.eval()
    #
    # 2. Create a DataLoader from val_dataset
    #    Hint: shuffle=False, fewer num_workers, drop_last=True
    #
    # 3. Loop over batches (up to num_batches):
    #    a. Move input_ids and targets to device
    #    b. Forward pass with autocast
    #    c. Accumulate loss.item()
    #    Hint: your training micro-step loop minus backward()
    #
    # 4. Return average loss
    #    Hint: total_loss / min(num_batches, len(val_loader))
    #
    # NOTE: don't call model.train() here — let the caller do it
    pass
```

#### Full Solution

```python
@torch.no_grad()
def evaluate(model, val_dataset, config, device, dtype, num_batches=20):
    """Compute average validation loss."""
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    total_loss = 0.0
    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(input_ids, targets)
        total_loss += loss.item()

    return total_loss / min(num_batches, len(val_loader))
```

---

### 1B: Text Generation

**Concepts:**
- `forward()` returns logits `[batch, seq_len, vocab_size]`. For generation, only the **last position** matters.
- Crop input to `max_seq_len` if it gets too long.
- `logits / temperature` before softmax — lower = more confident, higher = more random.
- `torch.multinomial` samples from the distribution (unlike argmax, gives diverse outputs).
- Stop early at EOT token.

#### Try It

```python
@torch.no_grad()
def generate_sample(model, enc, device, prompt="The", max_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    # TODO:
    # 1. model.eval()
    #
    # 2. Encode prompt to tensor
    #    Hint: token_ids = enc.encode(prompt)
    #    Hint: input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    #
    # 3. Loop max_tokens times:
    #    a. Crop to context window
    #       Hint: input_crop = input_ids[:, -model.config.max_seq_len:]
    #
    #    b. Forward pass
    #       Hint: logits, _ = model(input_crop)
    #
    #    c. Last position logits + temperature
    #       Hint: logits = logits[:, -1, :] / temperature
    #
    #    d. Sample
    #       Hint: probs = F.softmax(logits, dim=-1)
    #       Hint: next_token = torch.multinomial(probs, num_samples=1)
    #
    #    e. Append
    #       Hint: input_ids = torch.cat([input_ids, next_token], dim=1)
    #
    #    f. Stop at EOT
    #       Hint: if next_token.item() == enc.eot_token: break
    #
    # 4. return enc.decode(input_ids[0].tolist())
    pass
```

#### Full Solution

```python
@torch.no_grad()
def generate_sample(model, enc, device, prompt="The", max_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    model.eval()

    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        input_crop = input_ids[:, -model.config.max_seq_len:]
        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == enc.eot_token:
            break

    return enc.decode(input_ids[0].tolist())
```

---

### 1C: Integration

Add to `TrainConfig`:
```python
    eval_interval: int = 500
    val_dir: str = "data/fineweb-edu/val"
```

Fixed eval prompts (same every time to track progress):
```python
EVAL_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The president announced that",
    "def fibonacci(n):",
]
```

In `train()` setup:
```python
    enc = tiktoken.get_encoding("gpt2")
    val_dataset = PretrainDataset(config.val_dir, config.seq_len)
```

In the loop, after checkpointing:
```python
        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, config, device, dtype)
            print(f"  >>> val_loss: {val_loss:.4f}")
            for prompt in EVAL_PROMPTS:
                sample = generate_sample(model, enc, device, prompt=prompt)
                print(f"  >>> [{prompt}] {sample[:150]}")
            model.train()  # back to training mode
```

---

## Part 2: torch.compile

```python
    model = GPT(model_config).to(device)
    model = torch.compile(model)  # ← one line
```

- First step slow (~30-60s, compiling). Every step after ~1.5-2x faster.
- Requires PyTorch 2.0+.
- Place after `.to(device)`, before optimizer creation.
- If issues, just remove it — model works either way.

---

## Part 3: mHC Hyper-Connections

**Standard residual:** `x = x + F(x)` — one stream.
**mHC:** `n` parallel streams mixed through a doubly-stochastic matrix (Sinkhorn). Convex combinations can't explode.

### Try It

```python
# src/model/hyper_connection.py

import torch
import torch.nn as nn


def sinkhorn(log_A: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project matrix to doubly stochastic."""
    # TODO:
    # 1. A = log_A.exp()
    # 2. Loop `iters` times:
    #    a. A = A / A.sum(dim=-1, keepdim=True)   # normalize rows
    #    b. A = A / A.sum(dim=-2, keepdim=True)   # normalize columns
    # 3. return A
    pass


class HyperConnection(nn.Module):
    """Wraps a sublayer with mHC residual streams."""

    def __init__(self, d_model, n_streams=4, alpha_init=0.01, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # TODO: Define learnable parameters:
        #
        # self.log_A — mixing matrix logits [n_streams, n_streams]
        #   Hint: nn.Parameter(torch.eye(n_streams).log())
        #   Why eye? After Sinkhorn → identity = standard residual at init
        #
        # self.agg_weights — stream aggregation [n_streams]
        #   Hint: nn.Parameter(torch.ones(n_streams) / n_streams)
        #
        # self.alpha — sublayer output scale
        #   Hint: nn.Parameter(torch.tensor(alpha_init))
        #   Why small? Sublayer output is noisy at init

    def forward(self, streams, sublayer):
        """
        streams: [n_streams, batch, seq, dim]
        sublayer: attention or MLP module
        Returns: [n_streams, batch, seq, dim]
        """
        # TODO:
        # 1. Aggregate streams → single input for sublayer
        #    Hint: weights = torch.softmax(self.agg_weights, dim=0)
        #    Hint: aggregated = torch.einsum("n,nbsd->bsd", weights, streams)
        #
        # 2. sublayer_out = sublayer(aggregated)
        #
        # 3. Doubly-stochastic mixing
        #    Hint: A = sinkhorn(self.log_A, self.sinkhorn_iters)
        #    Hint: mixed = torch.einsum("mn,nbsd->mbsd", A, streams)
        #
        # 4. Add scaled sublayer output
        #    Hint: mixed = mixed + self.alpha * sublayer_out.unsqueeze(0)
        #
        # 5. return mixed
        pass
```

### Full Solution

```python
import torch
import torch.nn as nn


def sinkhorn(log_A: torch.Tensor, iters: int = 20) -> torch.Tensor:
    A = log_A.exp()
    for _ in range(iters):
        A = A / A.sum(dim=-1, keepdim=True)
        A = A / A.sum(dim=-2, keepdim=True)
    return A


class HyperConnection(nn.Module):
    def __init__(self, d_model, n_streams=4, alpha_init=0.01, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.log_A = nn.Parameter(torch.eye(n_streams).log())
        self.agg_weights = nn.Parameter(torch.ones(n_streams) / n_streams)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, streams, sublayer):
        weights = torch.softmax(self.agg_weights, dim=0)
        aggregated = torch.einsum("n,nbsd->bsd", weights, streams)
        sublayer_out = sublayer(aggregated)
        A = sinkhorn(self.log_A, self.sinkhorn_iters)
        mixed = torch.einsum("mn,nbsd->mbsd", A, streams)
        mixed = mixed + self.alpha * sublayer_out.unsqueeze(0)
        return mixed
```

### Model Integration

Add to `GPTConfig`:
```python
    use_mhc: bool = False
    n_streams: int = 4
```

In `TransformerBlock.__init__`:
```python
    if config.use_mhc:
        from src.model.hyper_connection import HyperConnection
        self.attn_hc = HyperConnection(config.d_model, config.n_streams)
        self.mlp_hc = HyperConnection(config.d_model, config.n_streams)
```

In `TransformerBlock.forward`:
```python
    if self.config.use_mhc:
        x = self.attn_hc(x, lambda h: self.attn(self.norm1(h)))
        x = self.mlp_hc(x, lambda h: self.mlp(self.norm2(h)))
    else:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
```

In `GPT.forward`, expand/collapse streams:
```python
    x = self.token_embeddings(input_ids)
    if self.config.use_mhc:
        x = x.unsqueeze(0).expand(self.config.n_streams, -1, -1, -1).clone()
    for block in self.blocks:
        x = block(x)
    if self.config.use_mhc:
        x = x.mean(dim=0)  # collapse streams
    x = self.final_norm(x)
    logits = self.lm_head(x)
```

**Ablation:** Train with `use_mhc=False` (baseline) and `use_mhc=True`. Compare loss curves and grad norms.

---

## Part 4: Muon Optimizer

### Try It

```python
def configure_optimizers(model, lr, muon_lr, weight_decay):
    """Split params: Muon for 2D weights, AdamW for rest."""
    # TODO:
    # 1. Two lists: muon_params, adamw_params
    #
    # 2. for name, param in model.named_parameters():
    #    - Skip if not requires_grad
    #    - "embedding", "lm_head", "norm", "bias" in name → adamw
    #    - param.ndim == 2 → muon
    #    - else → adamw
    #
    # 3. Print param counts for verification
    #
    # 4. Return AdamW with two groups at different LRs
    #    Hint: torch.optim.AdamW([
    #        {"params": muon_params, "lr": muon_lr, ...},
    #        {"params": adamw_params, "lr": lr, ...},
    #    ], betas=(0.9, 0.95))
    pass
```

### Full Solution

```python
def configure_optimizers(model, lr, muon_lr, weight_decay):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["embedding", "lm_head", "norm", "bias"]):
            adamw_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": muon_params, "lr": muon_lr, "weight_decay": weight_decay},
        {"params": adamw_params, "lr": lr, "weight_decay": weight_decay},
    ], betas=(0.9, 0.95))

    return optimizer
```

Replace in train.py:
```python
    optimizer = configure_optimizers(
        model, lr=config.max_lr, muon_lr=config.max_lr * 0.5, weight_decay=config.weight_decay
    )
```

---

## Part 5: Wandb Logging

### Try It

```python
# TODO at top of train.py:
# Import wandb optionally (try/except)

# TODO in train() setup:
# wandb.init(project="llm-350m")

# TODO in logging block:
# wandb.log({"train/loss": ..., "train/lr": ..., ...}, step=step)

# TODO in eval block:
# wandb.log({"val/loss": val_loss}, step=step)

# TODO at end of train():
# wandb.finish()
```

### Full Solution

Top of file:
```python
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
```

Add `use_wandb: bool = True` to TrainConfig.

In train() setup:
```python
    if config.use_wandb and HAS_WANDB:
        wandb.init(project="llm-350m")
```

In logging block:
```python
            if config.use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": running_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/tok_per_sec": tokens_per_sec,
                }, step=step)
```

In eval block:
```python
            if config.use_wandb and HAS_WANDB:
                wandb.log({"val/loss": val_loss}, step=step)
```

End of train():
```python
    if config.use_wandb and HAS_WANDB:
        wandb.finish()
```

**Watch:** train/loss ↓, val/loss tracking train/loss, grad_norm stable, tok/s consistent.

---

## Part 6: Launch Checklist

### Smoke Test (A40, ~$0.40/hr)

```bash
python prepare_data.py dummy --output_dir data/test
# Run with: data_dir="data/test", max_steps=100, micro_batch_size=4, grad_accum_steps=1
```

- [ ] No crashes
- [ ] Initial loss ≈ 10.4
- [ ] Loss decreases
- [ ] Checkpoint saves/loads

### Overfit Test

```python
single_batch = next(iter(train_loader))
for step in range(200):
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(single_batch[0].to(device), single_batch[1].to(device))
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step}: loss={loss.item():.4f}")
# Must approach 0 — if not, there's a bug
```

### Gradient Flow

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"  {name}: grad_norm={p.grad.norm().item():.6f}")
    else:
        print(f"  {name}: NO GRADIENT ← BUG")
```

### Full Launch

```bash
python prepare_data.py fineweb --output_dir data/fineweb-edu --num_tokens 10000000000
mkdir -p data/fineweb-edu/train data/fineweb-edu/val
mv data/fineweb-edu/shard_009[89].bin data/fineweb-edu/val/
mv data/fineweb-edu/shard_*.bin data/fineweb-edu/train/
python train.py
```

### Expected Timeline (H100)

| Phase | Steps | Tokens | Time | Loss |
|-------|-------|--------|------|------|
| Warmup | 0–1000 | ~500M | ~1h | 10.4 → ~5.0 |
| Early | 1000–5000 | ~2.5B | ~4h | 5.0 → ~3.5 |
| Mid | 5000–12000 | ~6B | ~7h | 3.5 → ~3.1 |
| Late | 12000–20000 | ~10B | ~8h | 3.1 → ~2.9 |
| **Total** | **20000** | **~10B** | **~20h** | **~2.9** |

Cost: ~$54 on H100. Budget $80-100 for reruns.

---

## Part 7: Complete train.py

### Try It

Take your current train.py and add in this order:
1. `eval_interval`, `val_dir`, `compile`, `use_wandb` to TrainConfig
2. `evaluate()` and `generate_sample()` functions
3. `EVAL_PROMPTS` list
4. In train(): tiktoken enc, val_dataset, torch.compile, wandb init
5. In the loop: eval block after checkpointing
6. wandb.finish() at end

### Full Solution

```python
# train.py
"""Training loop for 350M parameter language model."""

import os
import time
import math
import torch
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import PretrainDataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class TrainConfig:
    train_dir: str = "data/fineweb-edu/train"
    val_dir: str = "data/fineweb-edu/val"
    seq_len: int = 1024

    micro_batch_size: int = 16
    grad_accum_steps: int = 32

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    warmup_steps: int = 1000
    max_steps: int = 20000

    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    device: str = "cuda"
    compile: bool = True
    use_wandb: bool = True


EVAL_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The president announced that",
    "def fibonacci(n):",
]


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
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    torch.save(checkpoint, path)
    print(f"  >>> Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_weights"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.random.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    print(f"  >>> Resumed from step {checkpoint['step']}")
    return checkpoint["step"]


@torch.no_grad()
def evaluate(model, val_dataset, config, device, dtype, num_batches=20):
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )
    total_loss = 0.0
    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(input_ids, targets)
        total_loss += loss.item()
    return total_loss / min(num_batches, len(val_loader))


@torch.no_grad()
def generate_sample(model, enc, device, prompt="The", max_tokens=100, temperature=0.8):
    model.eval()
    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        input_crop = input_ids[:, -model.config.max_seq_len:]
        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == enc.eot_token:
            break

    return enc.decode(input_ids[0].tolist())


def train(config: TrainConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = config.device
    dtype = torch.bfloat16
    enc = tiktoken.get_encoding("gpt2")

    if config.use_wandb and HAS_WANDB:
        wandb.init(project="llm-350m")

    model_config = GPTConfig()
    model = GPT(model_config).to(device)
    if config.compile:
        model = torch.compile(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = PretrainDataset(config.train_dir, config.seq_len)
    val_dataset = PretrainDataset(config.val_dir, config.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    model.train()
    for step in range(config.max_steps):
        t_start = time.time()

        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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

            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(batch_inputs, batch_targets)

            loss = loss / config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        step_time = time.time() - t_start
        tokens_per_sec = (config.seq_len * config.micro_batch_size * config.grad_accum_steps) / step_time

        if step % config.log_interval == 0:
            print(
                f"Step {step:>6d} | "
                f"Loss {running_loss:.4f} | "
                f"LR {lr:.2e} | "
                f"Grad Norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"dt {step_time*1000:.0f}ms"
            )
            if config.use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": running_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/tok_per_sec": tokens_per_sec,
                }, step=step)

        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(model, optimizer, step, config)

        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, config, device, dtype)
            print(f"  >>> val_loss: {val_loss:.4f}")
            for prompt in EVAL_PROMPTS:
                sample = generate_sample(model, enc, device, prompt=prompt)
                print(f"  >>> [{prompt}] {sample[:150]}")
            if config.use_wandb and HAS_WANDB:
                wandb.log({"val/loss": val_loss}, step=step)
            model.train()

    if config.use_wandb and HAS_WANDB:
        wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
```

---

## What Comes After Training

1. **SFT** — fine-tune on instruction data
2. **GRPO** — RL phase: generate K completions, score with reward, update toward better ones. Connects to your Connections Best-of-K experience.
3. **Blog post** — same format as Connections: technical deep-dive, ablation tables, casual tone
