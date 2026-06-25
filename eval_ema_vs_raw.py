#!/usr/bin/env python3
"""Compare EMA vs raw final weights on the validation set.

Loads a V3 checkpoint, builds two models (one from 'model_weights', one from
'ema'), and runs the SAME validation pass on each so the comparison is apples-
to-apples. Prints both val losses + perplexities side by side.

The lower val_loss is your best base model -> serve/eval from that one.

Run:
  python eval_ema_vs_raw.py \
    --checkpoint /workspace/checkpoints_v3_700m_xsa_30b/step_028610.pt \
    --manifest /workspace/data/v3/manifest.jsonl \
    --num-batches 100
"""
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.gpt import GPT, GPTConfig
from src.data.manifest_dataset import ManifestDataset


def build_model(state, device):
    """Build a V3 GPT and load a (compile-prefixed) state dict into it."""
    config = GPTConfig(
        vocab_size=50304,
        d_model=1536, n_layers=24, n_heads=12, n_kv_heads=3,
        dropout=0.0, max_seq_len=1024,
        use_flash=True, tie_weights=True,
        use_qk_norm=True, use_diff_attn=False, use_xsa=True, use_mhc=False,
    )
    model = GPT(config)
    # weights were saved from a torch.compile'd module -> strip _orig_mod.
    clean = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, val_dataset, device, dtype, micro_batch_size, num_batches):
    """Average val loss over a FIXED number of batches.

    shuffle=False + drop_last=True => both models see the EXACT same batches,
    so the comparison is clean (no sampling noise between the two runs).
    """
    model.eval()
    loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )
    total_loss, n = 0.0, 0
    for i, (inputs, targets) in enumerate(loader):
        if i >= num_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(inputs, targets)
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="/workspace/data/v3/manifest.jsonl")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=64)
    ap.add_argument("--num-batches", type=int, default=100,
                    help="more batches = tighter estimate (100 is plenty)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = torch.bfloat16

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    print(f"  checkpoint keys: {list(ckpt.keys())}")
    has_ema = "ema" in ckpt and ckpt["ema"] is not None
    print(f"  has EMA: {has_ema}")
    if not has_ema:
        print("  !! No EMA in checkpoint — can only eval raw. Exiting comparison.")

    # Build val dataset ONCE; reuse for both models so batches are identical.
    val_dataset = ManifestDataset(args.manifest, split="val", seq_len=args.seq_len)
    print(f"  val chunks: {len(val_dataset):,}\n")

    results = {}

    # ---- RAW ----
    print("Evaluating RAW (model_weights)...")
    raw_model = build_model(ckpt["model_weights"], args.device)
    raw_loss, nb = evaluate(raw_model, val_dataset, args.device, dtype,
                            args.micro_batch_size, args.num_batches)
    results["raw"] = raw_loss
    print(f"  raw   val_loss = {raw_loss:.4f}  (ppl {math.exp(raw_loss):.2f})  over {nb} batches")
    del raw_model
    torch.cuda.empty_cache()

    # ---- EMA ----
    if has_ema:
        print("\nEvaluating EMA...")
        ema_model = build_model(ckpt["ema"], args.device)
        ema_loss, nb = evaluate(ema_model, val_dataset, args.device, dtype,
                                args.micro_batch_size, args.num_batches)
        results["ema"] = ema_loss
        print(f"  ema   val_loss = {ema_loss:.4f}  (ppl {math.exp(ema_loss):.2f})  over {nb} batches")
        del ema_model
        torch.cuda.empty_cache()

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    for name, loss in results.items():
        print(f"  {name:>4s}: val_loss {loss:.4f}  ppl {math.exp(loss):.2f}")
    if "ema" in results:
        delta = results["raw"] - results["ema"]   # positive => EMA better
        winner = "EMA" if delta > 0 else "RAW"
        print(f"\n  EMA - RAW = {-delta:+.4f}  ->  {winner} is better")
        if abs(delta) < 0.005:
            print("  (difference is within noise — either is fine; pick raw for simplicity)")
        print(f"\n  >>> Serve / eval from: {winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()