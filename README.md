# Modern LLM

Custom decoder-only language model experiments in PyTorch. The repository contains the model code, data preparation utilities, pretraining loops, SFT scripts, GRPO experiments, and evaluation adapters used while training three model generations:

- **V1**: ~350M parameters
- **V2**: ~615M parameters
- **V3**: ~672M parameters

The codebase is research-oriented: most scripts are direct experiment entry points with architecture and data paths exposed as command-line flags or top-level config values.

## Article series

This repository accompanies a four-part article series about coding, training, post-training, and serving small language models from scratch.

1. [Building a 350M Transformer From Scratch](https://john463212.substack.com/p/building-a-350m-transformer-from)  
   V1: a 353M baseline transformer with attention, RoPE, RMSNorm, SwiGLU, tied embeddings, training loops, bugs, and loss curves.

2. [Modernizing the Architecture](https://john463212.substack.com/p/modernizing-the-architecture)  
   V2: GQA, QK-Norm, Muon, EMA, Differential Attention, mHC, and what did or did not earn its place.

3. Post-training: SFT and GRPO  
   Coming soon.

4. Scaling and serving V3  
   Coming soon.

## What Is In Here

```text
src/model/          Transformer model components
src/data/           Binary shard, manifest, and SFT datasets
src/optim/          Muon + AdamW optimizer setup
src/eval/           lm-eval-harness adapter
src/grpo/           GRPO training, rewards, and curriculum utilities
train.py            Single-process pretraining loop for V1/V2-style runs
train_v3.py         DDP pretraining loop for manifest-based V3 runs
sft.py              Supervised fine-tuning for earlier checkpoints
sft_v3.py           Supervised fine-tuning for the V3 architecture
eval_checkpoint.py  Validation loss + sample generation for one checkpoint
eval_all.py         Benchmark sweep across configured checkpoints
make_manifest.py    Build the V3 mixed-source JSONL shard manifest
```

Core model features include RoPE, RMSNorm, SwiGLU blocks, optional grouped-query attention, optional QK norm, differential attention, experimental XSA support, optional mHC residual streams, EMA checkpoints, and optional Muon optimization.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable package metadata and dev tools:

```bash
pip install -e ".[dev]"
```

GPU training expects CUDA-capable PyTorch. The training scripts use bf16 autocast and are written primarily for Linux training machines or pods with `/workspace/...` mounted for data and checkpoints.

## Data

Pretraining expects tokenized `uint16` `.bin` shards containing GPT-2 token IDs. `src/data/prepare_data.py` can stream Hugging Face datasets and write shards.

Examples:

```bash
python -m src.data.prepare_data fineweb \
  --output_dir /workspace/data/fineweb-edu/train \
  --num_tokens 10000000000 \
  --shard_size 100000000

python -m src.data.prepare_data code_python \
  --output_dir /workspace/data/code_python \
  --num_tokens 5000000000 \
  --hf_token
```

For V3-style mixed-source training, edit the source paths and shard counts in `make_manifest.py`, then run:

```bash
python make_manifest.py
```

That writes a JSONL manifest with train/val shard entries and source labels.

## Pretraining

Single-process V1/V2-style training:

```bash
python train.py \
  --run-name v2_pretrain \
  --max-steps 43000 \
  --seq-len 1024 \
  --checkpoint-dir /workspace/checkpoints_v2 \
  --use-qk-norm \
  --use-diff-attn \
  --n-kv-heads 4
```

DDP manifest-based V3-style training:

```bash
torchrun --standalone --nproc_per_node=8 train_v3.py \
  --run-name v3_pretrain \
  --manifest-path /workspace/data/v3/manifest.jsonl \
  --target-train-tokens 43000000000 \
  --seq-len 2048 \
  --checkpoint-dir /workspace/checkpoints_v3 \
  --d-model 1536 \
  --n-layers 24 \
  --n-heads 12 \
  --n-kv-heads 3 \
  --use-xsa
```

Adjust batch size, gradient accumulation, and architecture flags to match the exact checkpoint you are resuming or reproducing. Architecture flags must match when loading a checkpoint.

## Supervised Fine-Tuning

Earlier checkpoints:

```bash
python sft.py
```

V3 checkpoint:

```bash
python sft_v3.py
```

Both scripts have config dataclasses near the top of the file. Update `base_checkpoint`, `checkpoint_dir`, sequence length, and dataset mix before running.

## Evaluation

Evaluate one checkpoint on a validation shard directory:

```bash
python eval_checkpoint.py \
  --checkpoint /workspace/checkpoints_v2/step_043000.pt \
  --val-dir /workspace/data/fineweb-edu/val \
  --seq-len 1024 \
  --micro-batch-size 16 \
  --num-batches 50
```

Run a configured `lm-eval-harness` benchmark sweep:

```bash
python -u eval_all.py
```

`eval_all.py` contains hard-coded checkpoint paths and architecture overrides, so update the `CONFIGS` list before running.

## Cleanup Recommendations

Safe to remove from the working tree if present:

```bash
rm -rf .venv __pycache__ .pytest_cache .ruff_cache
find . -name __pycache__ -type d -prune -exec rm -rf {} +
find . -name .DS_Store -type f -delete
rm -rf src/modern_llm.egg-info
```

Also consider deleting or consolidating these project files after checking whether you still need the history:

- `scripts/train.py`, `scripts/sample.py`, `scripts/eval.py`: tracked but currently zero-byte placeholders.
- `final_steps_guide_original.md`: appears to duplicate `final_steps_guide.md`.
- `v2_and_sft_guide.md` and `v2_and_sft_guide_v2.md`: keep only the current version if one supersedes the other.
- `# Complete LLM Training Guide: Theory + `: odd filename and likely a scratch guide export.
- `src/data/prepare_data_old.py`: keep only if it documents behavior missing from `src/data/prepare_data.py`.
- Empty placeholder directories such as `configs/`, `tests/`, and `src/training/` if you do not plan to fill them soon.

Do not commit local checkpoints, tokenized production shards, wandb runs, logs, or virtual environments. They are intentionally ignored.

## Notes

- Checkpoints are saved as full training states with model weights, optimizer state, RNG state, and optionally EMA weights.
- `torch.compile` prefixes can appear in state dict keys; eval/SFT helpers strip `_orig_mod.` where needed.
- `tiktoken` GPT-2 encoding is used throughout, with the vocabulary padded to `50304` for efficient tensor shapes.
