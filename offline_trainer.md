# Training Infrastructure Guide — 350M LLM from Scratch

*Offline reference for building the complete training pipeline*

---

## Where You Are

You've built the full model stack: RMSNorm → SwiGLU → RoPE → MultiHeadAttention → TransformerBlock → GPT, plus a BPE tokenizer for learning. The model config:

```python
GPTConfig(
    vocab_size=32768,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    dropout=0.0,
    max_seq_len=1024,
    use_flash=True,
    tie_weights=True
)
```

Now you need three things: **data pipeline**, **training loop**, and **optimizer setup**.

---

## Part 1: Dataset Class (PreTokenizedDataset)

### Concepts

Your data lives on disk as `.bin` shard files. Each shard is a flat array of `uint16` token IDs — millions of tokens concatenated end-to-end with EOS tokens between documents. The Dataset class serves random 1024-token chunks as `(input, target)` pairs.

**Key idea — memory mapping:** `np.memmap` lets you access a file as if it's an array in memory, but the OS only loads pages you actually read. This means you can work with 50GB+ of data without needing 50GB of RAM.

**Key idea — input/target shift:** For a 1024-token context, `input = tokens[0:1024]` and `target = tokens[1:1025]`. You need **1025 contiguous tokens** to make one pair.

**Key idea — shard indexing:** If shard 0 has 500 chunks and shard 1 has 300, you build a cumulative array `[0, 500, 800]`. For `idx=600`, it falls between 500 and 800, so it's in shard 1 at local offset 100.

### Questions to Answer

**Q1:** The `__init__` needs to build cumulative chunk counts. If a shard has `L` tokens, how many chunks does it produce? (Answer: `(L - 1) // seq_len` — the `-1` accounts for the extra token needed for the target shift.)

**Q2:** In `__getitem__`, once you know the shard and local offset, what's the actual byte position to start reading? (Answer: `start = local_offset * seq_len`. Then read `tokens[start : start + seq_len + 1]`.)

**Q3:** What dtype should the returned tensors be? (Answer: `torch.long` — PyTorch's `nn.Embedding` and `F.cross_entropy` expect `int64` indices.)

### Code to Implement

```python
# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PreTokenizedDataset(Dataset):
    """Dataset that serves chunks from pre-tokenized .bin shard files."""

    def __init__(self, data_dir: str, seq_len: int = 1024):
        self.seq_len = seq_len

        # Find all shards, sorted for deterministic ordering
        shard_paths = sorted(Path(data_dir).glob("*.bin"))
        assert len(shard_paths) > 0, f"No .bin files found in {data_dir}"

        # Memory-map each shard
        self.shards = []
        for path in shard_paths:
            shard = np.memmap(path, dtype=np.uint16, mode='r')
            self.shards.append(shard)

        # Build cumulative chunk counts
        # Each shard of length L produces (L - 1) // seq_len chunks
        self.cumulative_chunks = [0]
        for shard in self.shards:
            chunks_in_shard = (len(shard) - 1) // seq_len
            self.cumulative_chunks.append(
                self.cumulative_chunks[-1] + chunks_in_shard
            )

    def __len__(self) -> int:
        return self.cumulative_chunks[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Find which shard this index belongs to
        # TODO: figure out shard_idx and local_offset
        # Hint: walk through cumulative_chunks to find the right interval
        # Or use np.searchsorted(self.cumulative_chunks, idx, side='right') - 1

        shard_idx = np.searchsorted(self.cumulative_chunks, idx, side='right') - 1
        local_offset = idx - self.cumulative_chunks[shard_idx]

        # Read seq_len + 1 tokens
        start = local_offset * self.seq_len
        chunk = self.shards[shard_idx][start : start + self.seq_len + 1]

        # Convert to torch tensors
        chunk = torch.from_numpy(chunk.astype(np.int64))
        input_ids = chunk[:-1]    # tokens[0:1024]
        targets = chunk[1:]       # tokens[1:1025]

        return input_ids, targets
```

### Understanding `np.searchsorted`

This is a binary search. If `cumulative_chunks = [0, 500, 800, 1200]`:
- `idx=300` → `searchsorted(..., 300, 'right')` returns 1, minus 1 = shard 0
- `idx=600` → returns 2, minus 1 = shard 1, local_offset = 600 - 500 = 100
- `idx=900` → returns 3, minus 1 = shard 2, local_offset = 900 - 800 = 100

### Test It

```python
if __name__ == "__main__":
    import tempfile, os

    # Create dummy shard data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Two shards, each with 10250 tokens (should give 10 chunks each at seq_len=1024)
        for i in range(2):
            tokens = np.random.randint(0, 32768, size=10250, dtype=np.uint16)
            tokens.tofile(os.path.join(tmpdir, f"shard_{i:04d}.bin"))

        dataset = PreTokenizedDataset(tmpdir, seq_len=1024)
        print(f"Total chunks: {len(dataset)}")

        x, y = dataset[0]
        print(f"Input shape: {x.shape}, Target shape: {y.shape}")
        print(f"Input dtype: {x.dtype}")
        assert x.shape == (1024,)
        assert y.shape == (1024,)
        assert (x[1:] == y[:-1]).all()  # shifted by 1
        print("✓ All tests passed")
```

---

## Part 2: Data Preprocessing Script

This script streams FineWeb-Edu, tokenizes it, and saves `.bin` shards.

### Concepts

**Why pre-tokenize?** Tokenization is CPU-bound. If you tokenize on-the-fly during training, your GPU sits idle waiting for data. Pre-tokenize once, then training just reads integers from disk.

**Document concatenation:** You concatenate documents with an EOS token between them, then slice into clean `seq_len`-sized chunks. Attention will bleed across document boundaries — this is fine at 350M scale.

**Shard sizing:** ~100M tokens per shard is a good default. At `uint16` (2 bytes/token), that's ~200MB per shard file — manageable to read and shuffle.

### Code to Implement

```python
# src/data/prepare_data.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset


def prepare_fineweb_edu(
    output_dir: str,
    num_tokens: int = 10_000_000_000,  # 10B tokens target
    shard_size: int = 100_000_000,     # 100M tokens per shard
    seq_len: int = 1024,
):
    """Stream FineWeb-Edu, tokenize, save as .bin shards."""
    os.makedirs(output_dir, exist_ok=True)

    # Use GPT-2 tokenizer via tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token  # end-of-text token ID

    # Stream the dataset (no full download needed)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Accumulate tokens into buffer, write shards when full
    token_buffer = []
    shard_idx = 0
    total_tokens = 0

    for example in dataset:
        # Tokenize the document text
        text = example["text"]
        token_ids = enc.encode(text, allowed_special=set())

        # Append tokens + EOS
        token_buffer.extend(token_ids)
        token_buffer.append(eot_token)

        # Write shard when buffer is full
        while len(token_buffer) >= shard_size:
            # Take exactly shard_size tokens
            shard_tokens = token_buffer[:shard_size]
            token_buffer = token_buffer[shard_size:]

            # Save as uint16 numpy array
            shard_array = np.array(shard_tokens, dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
            shard_array.tofile(shard_path)

            total_tokens += shard_size
            shard_idx += 1
            print(f"Saved shard {shard_idx}: {total_tokens:,} tokens total")

        # Stop if we have enough tokens
        if total_tokens >= num_tokens:
            break

    # Save any remaining tokens as final shard
    if token_buffer:
        shard_array = np.array(token_buffer, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
        shard_array.tofile(shard_path)
        total_tokens += len(token_buffer)
        print(f"Saved final shard {shard_idx}: {total_tokens:,} tokens total")

    print(f"\nDone! {total_tokens:,} tokens across {shard_idx + 1} shards")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/fineweb-edu")
    parser.add_argument("--num_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    args = parser.parse_args()

    prepare_fineweb_edu(
        output_dir=args.output_dir,
        num_tokens=args.num_tokens,
        shard_size=args.shard_size,
    )
```

### Important Detail: Vocab Size vs uint16

`uint16` holds values 0–65535. GPT-2's tiktoken has ~50,257 tokens — fits fine. But if you ever use a tokenizer with vocab > 65535, you'd need `uint32` (and double your storage). Always check: `assert max(token_ids) < 65536`.

### Validation Questions

Before moving on, make sure you can answer:
- Why do we use `streaming=True`? (FineWeb-Edu is terabytes — you don't want to download it all)
- Why `uint16` and not `int32`? (Half the storage, tokens fit in 16 bits)
- What happens at document boundaries? (EOS token separates them, attention can cross — acceptable at this scale)

---

## Part 3: Training Loop

This is where everything comes together. The training loop ties together model, data, optimizer, scheduler, mixed precision, gradient accumulation, logging, and checkpointing.

### Concepts

**Mixed precision (bf16):** You keep a master copy of weights in float32, but run the forward pass and backward pass in bfloat16. The `torch.autocast` context manager handles this automatically. Gradients are computed in bf16 but accumulated/applied in fp32. This halves memory for activations and roughly doubles throughput.

**Gradient accumulation:** Your GPU can only fit `micro_batch_size` sequences at once (say 16). But you want an effective batch of 512 sequences. So you do 32 forward+backward passes, accumulating gradients, then do one optimizer step. The loss must be divided by the accumulation steps (or equivalently, use `mean` reduction which handles it per micro-batch).

**Learning rate schedule:** Warmup + cosine decay. Start from 0, linearly ramp to peak LR over `warmup_steps`, then cosine-decay to `min_lr` (typically 10% of peak). The warmup prevents early instability when the model hasn't learned anything yet and gradients are noisy.

**Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Prevents exploding gradients from destabilizing training. Applied after accumulation, before the optimizer step.

### Optimizer Setup: Muon + AdamW

Split parameters into two groups:

```python
def configure_optimizers(model, lr, weight_decay, muon_lr):
    """Set up Muon for 2D params, AdamW for everything else."""
    muon_params = []   # 2D weight matrices
    adamw_params = []  # embeddings, norms, biases, 1D params

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embeddings and LM head -> AdamW
        if "token_embeddings" in name or "lm_head" in name:
            adamw_params.append(param)
        # RMSNorm scales (1D) -> AdamW
        elif "norm" in name:
            adamw_params.append(param)
        # Biases (1D) -> AdamW
        elif "bias" in name:
            adamw_params.append(param)
        # 2D weight matrices (attention projections, MLP) -> Muon
        elif param.ndim == 2:
            muon_params.append(param)
        # Anything else -> AdamW
        else:
            adamw_params.append(param)

    # Create optimizer groups
    # You'll need to install Muon: pip install muon-optimizer (or implement it)
    # For now, you can use AdamW for everything as baseline
    optimizer = torch.optim.AdamW([
        {"params": muon_params, "lr": muon_lr, "weight_decay": weight_decay},
        {"params": adamw_params, "lr": lr, "weight_decay": weight_decay},
    ], betas=(0.9, 0.95))

    return optimizer
```

**When you add Muon later**, replace the first group's optimizer. The parameter split stays the same.

### Learning Rate Scheduler

```python
import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Warmup + cosine decay schedule."""
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    # Cosine decay
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### Training Config

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/fineweb-edu"
    seq_len: int = 1024

    # Batch size
    micro_batch_size: int = 16        # sequences per GPU per step
    grad_accum_steps: int = 32        # accumulation steps
    # effective_batch = 16 * 32 = 512 sequences = 524,288 tokens

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5              # 10% of max
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 19000            # ~10B tokens / 524K tokens_per_step

    # Logging & Checkpointing
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    checkpoint_dir: str = "checkpoints"

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
```

### The Training Loop

```python
# train.py
import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import PreTokenizedDataset


def train(config: TrainConfig):
    """Main training loop."""

    # ---- Setup ----
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device(config.device)
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    # Model
    model_config = GPTConfig()
    model = GPT(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compile for speed (PyTorch 2.0+)
    model = torch.compile(model)

    # Data
    train_dataset = PreTokenizedDataset(config.data_dir, seq_len=config.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # avoid partial batches
    )
    train_iter = iter(train_loader)

    # Optimizer
    optimizer = configure_optimizers(
        model,
        lr=config.max_lr,
        weight_decay=config.weight_decay,
        muon_lr=config.max_lr,  # tune separately later
    )

    # ---- Training ----
    model.train()

    for step in range(config.max_steps):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(step, config.warmup_steps, config.max_steps,
                    config.max_lr, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(config.grad_accum_steps):
            # Get batch (restart iterator if exhausted)
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass in mixed precision
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

            # Scale loss by accumulation steps
            loss = loss / config.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        )

        # Optimizer step
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (
            config.micro_batch_size * config.grad_accum_steps
            * config.seq_len / dt
        )

        # Logging
        if step % config.log_interval == 0:
            print(
                f"step {step:>6d} | "
                f"loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.2f} | "
                f"dt {dt*1000:.0f}ms | "
                f"tok/s {tokens_per_sec:,.0f}"
            )

        # Checkpointing
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(model, optimizer, step, config)

        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, config, device, dtype)
            print(f"  >>> val_loss: {val_loss:.4f}")
            model.train()  # back to training mode
```

### Checkpointing

```python
def save_checkpoint(model, optimizer, step, config):
    """Save everything needed to resume training."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        # Add these for exact reproducibility:
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    path = os.path.join(config.checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(checkpoint, path)
    print(f"  >>> Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer):
    """Resume training from checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.random.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    return checkpoint["step"]
```

### Evaluation

```python
@torch.no_grad()
def evaluate(model, config, device, dtype, num_batches=20):
    """Run evaluation on held-out data."""
    model.eval()

    # Load validation data (separate directory or held-out shards)
    val_dataset = PreTokenizedDataset(
        config.data_dir.replace("train", "val"),
        seq_len=config.seq_len
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=2,
    )

    total_loss = 0.0
    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= num_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        total_loss += loss.item()

    return total_loss / min(num_batches, len(val_loader))
```

---

## Part 4: Putting It All Together

### File Structure

```
your-project/
├── src/
│   ├── model/
│   │   ├── rmsnorm.py
│   │   ├── swiglu.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── block.py
│   │   └── gpt.py
│   ├── tokenizer/
│   │   └── bpe.py
│   └── data/
│       ├── dataset.py         ← PreTokenizedDataset
│       └── prepare_data.py    ← FineWeb-Edu preprocessing
├── train.py                   ← Main training script
├── configs/
│   └── base_350m.yaml         ← Hyperparameters (optional)
└── checkpoints/
```

### Workflow

```bash
# Step 1: Pre-tokenize data (run once, takes hours for 10B tokens)
python src/data/prepare_data.py --output_dir data/fineweb-edu --num_tokens 10000000000

# Step 2: Smoke test (quick, validates everything works)
python train.py --max_steps 100 --micro_batch_size 4 --grad_accum_steps 1

# Step 3: Full training on H100
python train.py
```

### Debugging Checklist

Run these checks before committing to a long training run:

**1. Loss sanity check:** At initialization, loss should be approximately `-ln(1/vocab_size) = -ln(1/32768) ≈ 10.4`. If it's much higher or lower, something is wrong with your model or data.

**2. Overfitting test:** Train on a single batch for 100 steps. Loss should drop to near zero. If it can't memorize one batch, there's a bug.

```python
# Quick overfit test
single_batch = next(iter(train_loader))
for step in range(100):
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(single_batch[0].to(device))
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            single_batch[1].to(device).view(-1)
        )
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step}: loss={loss.item():.4f}")
# Loss should approach 0
```

**3. Gradient flow check:** After one step, verify gradients exist and aren't zero/NaN:

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name}: grad_norm={p.grad.norm().item():.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

**4. Generation test:** Generate text at step 0 (should be gibberish) and every 1000 steps (should get progressively more coherent):

```python
def generate_sample(model, enc, prompt="The", max_tokens=100):
    model.eval()
    tokens = enc.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.8)
    return enc.decode(output[0].tolist())
```

### What to Watch During Training

**Healthy signs:**
- Loss decreasing smoothly (some noise is fine)
- Gradient norms roughly stable (not growing or collapsing)
- `tok/s` consistent (no I/O bottlenecks)
- Generated text quality improving at checkpoints

**Red flags:**
- Loss spikes that don't recover → lower learning rate
- Loss = NaN → gradient explosion, lower LR or check for bugs
- Loss plateaus early → LR too low, or data issue
- Gradient norm consistently hitting clip value → LR may be too high
- `tok/s` drops over time → memory leak or dataloader issue

### Expected Timeline

| Phase | Steps | Tokens | Time (H100) | Expected Loss |
|-------|-------|--------|-------------|---------------|
| Warmup | 0–2000 | ~1B | ~2h | 10.4 → ~4.0 |
| Mid training | 2000–10000 | ~5B | ~10h | 4.0 → ~3.2 |
| Late training | 10000–19000 | ~10B | ~8h | 3.2 → ~2.9 |

---

## Part 5: Exercises

Once you've implemented the above, try these to deepen understanding:

**Exercise 1:** Add wandb logging. Log loss, learning rate, gradient norm, throughput, and generated samples. This is essential for debugging remote runs.

**Exercise 2:** Implement Muon properly. Replace the first optimizer group with actual Muon (Newton-Schulz orthogonalization on the momentum buffer). Benchmark against pure AdamW.

**Exercise 3:** Add the mHC (multi-head Hyper-Connections) behind a config flag. Train two runs — one baseline, one with mHC — and compare loss curves. This is your ablation study.

**Exercise 4:** Implement data loader state saving/loading so you can resume without re-reading data. Track which shard and position you're at.

**Exercise 5:** Add multi-GPU support with `torch.distributed` and `DistributedDataParallel`. Not needed for 350M on a single H100, but great learning exercise.

---

## Quick Reference Card

```
# Effective batch size
effective_batch = micro_batch_size × grad_accum_steps × num_gpus
effective_tokens = effective_batch × seq_len

# Total training steps
total_steps = total_tokens / effective_tokens_per_step

# Initial loss sanity check
expected_init_loss = -ln(1 / vocab_size) ≈ 10.4  (for vocab=32768)

# Memory rule of thumb (bf16)
model_memory ≈ 2 × num_params bytes (weights)
optimizer_memory ≈ 8 × num_params bytes (AdamW: fp32 copy + m + v)
activation_memory ≈ depends on batch size and seq_len

# 350M model estimate:
# Weights: ~700MB (bf16)
# Optimizer: ~2.8GB (fp32)
# Activations: ~2-8GB (depends on batch size)
# Total: ~5-12GB → fits on any modern GPU
```