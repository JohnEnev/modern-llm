"""
Staged smoke test for v2 architecture.

Runs a tiny model through ~50 steps on a single repeated batch (overfit-style)
at each stage, checking for:
  - no NaN/Inf in loss or gradients
  - loss decreases (model can learn)
  - reasonable grad norm (no explosion)

Run this LOCALLY (CPU is fine for this tiny size) before any GPU pod.
Place at repo root and run: python3 smoke_test.py
"""

import torch
import torch.nn.functional as F
from src.model.gpt import GPT, GPTConfig
from src.optim.muon import configure_optimizers

torch.manual_seed(0)

# Tiny config for fast CPU iteration
TINY = dict(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=4,   # standard MHA unless overridden
    dropout=0.0,
    max_seq_len=64,
    use_flash=False,  # flash often unavailable/slow on CPU; manual path is fine for smoke test
    tie_weights=True,
)

BATCH, SEQ = 4, 32
N_STEPS = 50


def run_stage(name: str, config_overrides: dict, use_muon: bool):
    print(f"\n{'='*60}")
    print(f"STAGE: {name}")
    print(f"{'='*60}")

    cfg = dict(TINY)
    cfg.update(config_overrides)
    config = GPTConfig(**cfg)
    model = GPT(config)

    # Fixed random batch, reused every step (overfit check)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ))
    targets = torch.randint(0, config.vocab_size, (BATCH, SEQ))

    if use_muon:
        muon_opt, adamw_opt = configure_optimizers(
            model, lr=3e-3, muon_lr=1.5e-3, weight_decay=0.0
        )
        optimizers = [muon_opt, adamw_opt]
    else:
        adamw_opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        optimizers = [adamw_opt]

    losses = []
    for step in range(N_STEPS):
        for opt in optimizers:
            opt.zero_grad()

        logits, loss = model(input_ids, targets)

        if not torch.isfinite(loss):
            print(f"  ✗ FAILED at step {step}: loss is {loss.item()}")
            return False

        loss.backward()

        # Check for NaN/Inf gradients
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            print(f"  ✗ FAILED at step {step}: non-finite gradient detected")
            return False

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        for opt in optimizers:
            opt.step()

        losses.append(loss.item())

        if step % 10 == 0 or step == N_STEPS - 1:
            print(f"  step {step:3d} | loss {loss.item():.4f} | grad_norm {grad_norm:.3f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    improved = final_loss < initial_loss * 0.9  # expect at least 10% reduction in 50 steps

    print(f"  Initial loss: {initial_loss:.4f}  ->  Final loss: {final_loss:.4f}")
    if improved:
        print(f"  ✓ PASSED ({name})")
    else:
        print(f"  ⚠ WARNING: loss did not improve much — investigate before proceeding")

    return improved


if __name__ == "__main__":
    results = {}

    # Stage 1: pure baseline — no new v2 features at all
    results["1_baseline"] = run_stage(
        "1. Baseline (no Muon, no GQA, no QK-Norm, no DiffAttn, no mHC)",
        dict(use_qk_norm=False, use_diff_attn=False, use_mhc=False, n_kv_heads=4),
        use_muon=False,
    )

    # Stage 2: baseline + Muon only
    results["2_muon"] = run_stage(
        "2. Baseline + Muon optimizer",
        dict(use_qk_norm=False, use_diff_attn=False, use_mhc=False, n_kv_heads=4),
        use_muon=True,
    )

    # Stage 3: full attention stack (GQA + QK-Norm + DiffAttn), no mHC, with Muon
    results["3_attention_stack"] = run_stage(
        "3. Muon + GQA + QK-Norm + Differential Attention (no mHC)",
        dict(use_qk_norm=True, use_diff_attn=True, use_mhc=False, n_kv_heads=2),
        use_muon=True,
    )

    # Stage 4: full v2 — everything including mHC
    results["4_full_v2"] = run_stage(
        "4. Full v2 (Muon + GQA + QK-Norm + DiffAttn + mHC, n_streams=2)",
        dict(use_qk_norm=True, use_diff_attn=True, use_mhc=True, n_kv_heads=2, n_streams=2),
        use_muon=True,
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for stage, passed in results.items():
        status = "✓ PASS" if passed else "⚠ CHECK"
        print(f"  {stage}: {status}")