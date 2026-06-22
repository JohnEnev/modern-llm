"""DDP (DistributedDataParallel) helpers for V3 multi-GPU training.

Goal: data-parallel training across N GPUs on one node. Each GPU runs its
own process with a full model replica; each processes a different slice of
every batch; gradients are all-reduced so replicas stay identical.

Launch with:
    torchrun --standalone --nproc_per_node=2 -m src.train_v3 [args...]

torchrun spawns nproc_per_node processes and sets env vars (RANK,
LOCAL_RANK, WORLD_SIZE) that this module reads.

The intent is to integrate these helpers into train_v3.py rather than
rewrite it: wrap the model in DDP, swap the sampler, and guard
logging/saving to rank 0.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Init / teardown
# ---------------------------------------------------------------------------

def ddp_setup() -> dict:
    """Initialize the process group and return rank info.

    Returns a dict with keys: rank, local_rank, world_size, device, is_main.

    Falls back to single-GPU defaults (rank=0, world_size=1) if torchrun
    env vars are absent, so the same script runs with or without torchrun.
    """
    # torchrun sets these three env vars before spawning each process.
    # .get(..., default) means single-GPU runs (no torchrun) get sane values.
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # NCCL is the standard GPU-to-GPU collective backend.
        # This call blocks until ALL ranks have called it (it's a collective).
        dist.init_process_group(backend="nccl")

    # Tell PyTorch which physical GPU this process owns.
    # local_rank is "which GPU on THIS node" (0 or 1 for a 2-GPU machine).
    # Returns None — it's a side effect, not a value.
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    return {
        "rank": rank,               # global rank across all nodes
        "local_rank": local_rank,   # which GPU on this node
        "world_size": world_size,   # total number of processes
        "device": device,           # e.g. "cuda:0" or "cuda:1"
        "is_main": (rank == 0),     # only rank 0 logs/saves
    }


def ddp_cleanup(world_size: int):
    """Destroy the process group at the end of training.

    Must be called after the training loop so NCCL resources are released
    cleanly. Safe to call even if init failed — the world_size guard
    ensures we only tear down what we set up.
    """
    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Wrap the model
# ---------------------------------------------------------------------------

def wrap_ddp(model, local_rank: int, world_size: int):
    """Wrap a model in DDP (if multi-GPU), else return it unchanged.

    DDP installs gradient-sync hooks so that during loss.backward(),
    gradients are all-reduced across all ranks automatically — you don't
    call anything manually. Every replica ends up with the same averaged
    gradient, so optimizer.step() keeps them identical.

    IMPORTANT: after wrapping, the raw model is at model.module.
    Always save model.module.state_dict() — not model.state_dict() —
    so the checkpoint has no 'module.' key prefix and is loadable
    in any non-DDP context (eval harness, inference, single-GPU runs).
    """
    # Move to this rank's GPU before wrapping —
    # DDP requires the model to already be on the target device.
    model.to(f"cuda:{local_rank}")

    if world_size > 1:
        # device_ids tells DDP which GPU this process owns.
        return DDP(model, device_ids=[local_rank])

    # Single-GPU: return unchanged (no DDP overhead, same code path).
    return model


# ---------------------------------------------------------------------------
# Distributed sampler for the dataloader
# ---------------------------------------------------------------------------

def make_distributed_loader(
    dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
):
    """Build a DataLoader that gives each rank a DIFFERENT slice of the data.

    Returns (loader, sampler). Call sampler.set_epoch(epoch) at the start
    of each epoch so the shuffle seed changes (otherwise every epoch sees
    data in the same order).

    In single-GPU mode, sampler=None is returned — the caller should guard:
        if sampler is not None:
            sampler.set_epoch(epoch)
    """
    if world_size > 1:
        # DistributedSampler partitions the dataset across ranks.
        # Each rank sees a disjoint 1/world_size slice of the data —
        # the fix to "both GPUs see the same data" (your B1 answer).
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        # CRITICAL: do NOT also pass shuffle=True to DataLoader when using a
        # sampler — they conflict and PyTorch raises an error. The sampler
        # owns shuffling; DataLoader just reads in sampler order.
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            # drop_last=True keeps all ranks in sync: without it, the last
            # batch may be smaller on some ranks, causing a size mismatch
            # in the all-reduce and hanging the run.
            drop_last=True,
        )
    else:
        # Single GPU: plain DataLoader with direct shuffle.
        sampler = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return loader, sampler


# ---------------------------------------------------------------------------
# Rank-guarded helpers (so only main process logs/saves)
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    """True if this is rank 0, or if no process group is initialized.

    Use this to guard wandb.log(), print(), and torch.save() so they only
    fire once (from rank 0), not N times (once per GPU).

    Do NOT use this to guard loss.backward() — backward contains the
    all-reduce collective that syncs gradients. If any rank skips it,
    the others hang forever waiting for the missing participant.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Single-GPU or process group not yet set up: always main.
    return True


def save_checkpoint_main(model, optimizer, step, path, world_size):
    """Save a checkpoint ONLY from the main process (rank 0).

    Since all replicas are identical after all-reduce, saving just one is
    sufficient. Non-main ranks return immediately without doing any I/O.

    Unwraps the DDP wrapper before saving so checkpoint keys have no
    'module.' prefix — the checkpoint is then identical in format to a
    single-GPU checkpoint and loadable without DDP.
    """
    # All non-main ranks bail out immediately — no file I/O, no duplicate saves.
    if not is_main_process():
        return

    # model.module is the raw GPT if DDP-wrapped; otherwise model itself.
    # Saving raw.state_dict() gives keys like "token_embeddings.weight",
    # not "module.token_embeddings.weight" — matches your existing loader code.
    raw = model.module if hasattr(model, "module") else model

    torch.save({
        "model_weights": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
    }, path)


# ---------------------------------------------------------------------------
# Integration notes — how to wire this into train_v3.py
# ---------------------------------------------------------------------------
#
#   info = ddp_setup()
#   model = build_model(config)
#   model = wrap_ddp(model, info["local_rank"], info["world_size"])
#   loader, sampler = make_distributed_loader(
#       dataset, micro_batch, info["rank"], info["world_size"]
#   )
#
#   for epoch in range(num_epochs):
#       if sampler is not None:
#           sampler.set_epoch(epoch)       # different shuffle each epoch
#       for batch in loader:
#           loss = forward(model, batch)
#           loss.backward()                # DDP all-reduces grads — NO guard
#           optimizer.step()
#           if info["is_main"] and step % log_interval == 0:
#               wandb.log(...)             # rank-0 only
#       if info["is_main"]:
#           save_checkpoint_main(model, optimizer, step, path,
#                                info["world_size"])
#
#   ddp_cleanup(info["world_size"])
#
# Effective batch tokens = micro_batch * grad_accum * seq_len * world_size
# Account for world_size when targeting ~1M tokens/step for V3.
