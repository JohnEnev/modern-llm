# eval_checkpoint.py
"""Evaluate a raw or EMA checkpoint on FineWeb-Edu validation.

Example:

python eval_checkpoint.py \
  --checkpoint /workspace/checkpoints_v2_mhc2/step_000500.pt \
  --val-dir data/fineweb-edu/val \
  --seq-len 1024 \
  --micro-batch-size 16 \
  --num-batches 50 \
  --n-kv-heads 4 \
  --use-qk-norm \
  --use-diff-attn \
  --use-mhc \
  --n-streams 2

For raw model eval, do not pass --use-ema.
For EMA eval, pass --use-ema.
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import PretrainDataset
from src.model.attention import DifferentialAttention
from src.model.mhc import MHCResidual


def clean_state_dict(state_dict):
    """Remove torch.compile _orig_mod. prefixes if present."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


@torch.no_grad()
def evaluate(model, val_dataset, device, dtype, micro_batch_size, num_batches):
    model.eval()

    loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i, (input_ids, targets) in enumerate(loader):
        if i >= num_batches:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(input_ids, targets)

        total_loss += loss.item()
        total_tokens += input_ids.numel()

    dt = time.time() - t0
    avg_loss = total_loss / num_batches
    tok_s = total_tokens / dt

    return avg_loss, tok_s, dt


@torch.no_grad()
def get_differential_lambdas(model):
    values = []

    for module in model.modules():
        if isinstance(module, DifferentialAttention):
            lam = module.compute_lambda().detach().float().cpu().item()
            values.append(lam)

    if not values:
        return None

    return torch.tensor(values)


@torch.no_grad()
def get_mhc_stats(model):
    row_errors = []
    col_errors = []
    write_mins = []
    write_maxs = []
    read_entropies = []

    for module in model.modules():
        if isinstance(module, MHCResidual):
            A = module.mixing_matrix().detach().float()

            row_errors.append((A.sum(dim=-1) - 1).abs().max().item())
            col_errors.append((A.sum(dim=-2) - 1).abs().max().item())

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
def generate_sample(model, enc, device, prompt="The", max_tokens=80, temperature=0.8):
    model.eval()

    token_ids = enc.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    real_vocab_size = enc.n_vocab

    for _ in range(max_tokens):
        input_crop = input_ids[:, -model.config.max_seq_len:]

        logits, _ = model(input_crop)
        logits = logits[:, -1, :] / temperature

        # Mask padded vocab IDs.
        logits[:, real_vocab_size:] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == enc.eot_token:
            break

    return enc.decode(input_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()

    # Checkpoint / data
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-dir", type=str, default="data/fineweb-edu/val")
    parser.add_argument("--use-ema", action="store_true")

    # Model config
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--use-diff-attn", action="store_true")
    parser.add_argument("--use-mhc", action="store_true")
    parser.add_argument("--n-streams", type=int, default=2)

    # Eval config
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile-model", action="store_true")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    device = args.device
    dtype = torch.bfloat16
    enc = tiktoken.get_encoding("gpt2")

    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        dropout=0.0,
        max_seq_len=args.seq_len,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=args.use_qk_norm,
        use_diff_attn=args.use_diff_attn,
        use_mhc=args.use_mhc,
        n_streams=args.n_streams,
    )

    print("=" * 80)
    print("EVAL CONFIG")
    print("=" * 80)
    print(f"checkpoint:       {args.checkpoint}")
    print(f"use_ema:          {args.use_ema}")
    print(f"val_dir:          {args.val_dir}")
    print(f"num_batches:      {args.num_batches}")
    print(f"micro_batch_size: {args.micro_batch_size}")
    print(f"seq_len:          {args.seq_len}")
    print(f"n_kv_heads:       {args.n_kv_heads}")
    print(f"use_qk_norm:      {args.use_qk_norm}")
    print(f"use_diff_attn:    {args.use_diff_attn}")
    print(f"use_mhc:          {args.use_mhc}")
    print(f"n_streams:        {args.n_streams if args.use_mhc else 'n/a'}")
    print("=" * 80)

    model = GPT(model_config).to(device)

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.use_ema:
        if "ema" not in ckpt:
            raise ValueError("Checkpoint does not contain EMA weights.")
        state_dict = clean_state_dict(ckpt["ema"])
        print("Loaded EMA weights.")
    else:
        state_dict = clean_state_dict(ckpt["model_weights"])
        print("Loaded raw model weights.")

    model.load_state_dict(state_dict, strict=True)

    if args.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    val_dataset = PretrainDataset(args.val_dir, args.seq_len)
    print(f"Validation chunks: {len(val_dataset):,}")

    val_loss, tok_s, dt = evaluate(
        model=model,
        val_dataset=val_dataset,
        device=device,
        dtype=dtype,
        micro_batch_size=args.micro_batch_size,
        num_batches=args.num_batches,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"val/loss:          {val_loss:.4f}")
    print(f"eval tok/s:        {tok_s:,.0f}")
    print(f"eval time:         {dt:.2f}s")
    print("=" * 80)

    lambdas = get_differential_lambdas(model)
    if lambdas is not None:
        print("\nDifferential attention lambdas:")
        print(f"  min:  {lambdas.min().item():.4f}")
        print(f"  max:  {lambdas.max().item():.4f}")
        print(f"  mean: {lambdas.mean().item():.4f}")

    mhc_stats = get_mhc_stats(model)
    if mhc_stats is not None:
        print("\nmHC stats:")
        for k, v in mhc_stats.items():
            print(f"  {k}: {v:.6f}")

    print("\nSamples:")
    prompts = [
        "The meaning of life is",
        "In a distant galaxy,",
        "The president announced that",
        "def fibonacci(n):",
    ]

    for prompt in prompts:
        sample = generate_sample(model, enc, device, prompt=prompt)
        print("-" * 80)
        print(f"[{prompt}]")
        print(sample[:500])


if __name__ == "__main__":
    main()