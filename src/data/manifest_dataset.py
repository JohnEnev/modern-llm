"""Manifest-based dataset for V3 multi-source pretraining.

A manifest is a JSONL file, one entry per shard:
    {"source": "fineweb", "split": "train", "path": ".../shard_0000.bin", "tokens": 100000000}
    {"source": "python",  "split": "train", "path": ".../shard_0000.bin", "tokens": 100000000}
    {"source": "math",    "split": "val",   "path": ".../shard_0049.bin", "tokens": 100000000}

This decouples "what shards exist on disk" from "what this run trains on",
and lets you control the train/val split and the source mix explicitly.

Mixing strategy: simple concatenate-and-shuffle. The mix ratio is determined
by HOW MANY shards of each source are listed in the train split (e.g. 320
fineweb + 50 code + 30 math ~= 80/12/8). No weighted sampling.

Chunking convention: stride = seq_len (matches the existing PretrainDataset),
reading seq_len+1 tokens per chunk. Consecutive chunks share one boundary
token (the last target of one chunk == the first input of the next), which
tiles a continuous token stream cleanly. This is the standard convention and
keeps V3 consistent with how V1/V2 were trained.
"""

import os
import json
import glob
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str, split: str) -> list[dict]:
    """Read a JSONL manifest and return only entries matching `split`."""
    # Each line is a JSON object; parse them all into dicts.
    with open(manifest_path, "r") as json_file:
        entries = [json.loads(line) for line in json_file]

    # Keep only the requested split ("train" or "val").
    matching = [e for e in entries if e["split"] == split]

    # Print a per-source shard count so you can eyeball the mix at load time.
    counts = Counter(e["source"] for e in matching)
    print(f"[{split}] shards per source: {dict(counts)}")

    return matching


def write_manifest(manifest_path: str, entries: list[dict]):
    """Write a list of entry dicts as JSONL (one JSON object per line)."""
    with open(manifest_path, "w") as jsonl_file:
        for entry in entries:
            json.dump(entry, jsonl_file)
            jsonl_file.write("\n")


def build_manifest_entries(
    source_dirs: dict[str, str],
    train_shards_per_source: dict[str, int],
    val_shards_per_source: dict[str, int],
    tokens_per_shard: int = 100_000_000,
) -> list[dict]:
    """Construct manifest entries from per-source shard directories.

    Run this ONCE to generate the manifest, then commit the manifest file.

    Args:
        source_dirs:             {"fineweb": "/path/to/fineweb/train", ...}
        train_shards_per_source: {"fineweb": 320, "python": 50, "math": 30}
        val_shards_per_source:   {"fineweb": 3,   "python": 1,  "math": 1}
        tokens_per_shard:        recorded in each entry (informational)
    """
    entries = []

    for source, dir_path in source_dirs.items():
        # List this source's shards, sorted for deterministic assignment.
        shards = sorted(glob.glob(os.path.join(dir_path, "*.bin")))

        n_val = val_shards_per_source[source]
        n_train = train_shards_per_source[source]

        # First n_val shards -> val; next n_train -> train. Disjoint slices,
        # so no shard is ever in both splits.
        val_shards = shards[:n_val]
        train_shards = shards[n_val : n_val + n_train]

        for path in val_shards:
            entries.append({
                "source": source,
                "split": "val",
                "path": path,
                "tokens": tokens_per_shard,
            })
        for path in train_shards:
            entries.append({
                "source": source,
                "split": "train",
                "path": path,
                "tokens": tokens_per_shard,
            })

    return entries


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    """Concatenate-and-shuffle dataset over manifest-listed shards.

    TRAIN: load all train-split shards, treat as one big token pool, chunk
    into seq_len+1 windows. Shuffling is handled by the DataLoader
    (shuffle=True) at the sequence-index level.

    Per-source VAL: build ONE ManifestDataset per source (source_filter=...)
    so each source's val loss can be reported separately.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        seq_len: int = 1024,
        source_filter: str | None = None,
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.seq_len = seq_len
        self.source_filter = source_filter

        # Load the entries for this split, optionally filtered to one source.
        entries = load_manifest(manifest_path, split)
        if source_filter is not None:
            entries = [e for e in entries if e["source"] == source_filter]

        if len(entries) == 0:
            raise ValueError(
                f"No shards for split={split!r} source_filter={source_filter!r}"
            )

        # Memory-map every shard (lazy: maps the file, doesn't load it into RAM)
        # and build a cumulative chunk-count table so __getitem__ can map a
        # global chunk index -> (shard, offset) without concatenating anything.
        self.shards = []          # one memmap per shard
        cumulative = []           # cumulative chunk counts (shard boundaries)
        total_chunks = 0

        for entry in entries:
            shard = np.memmap(entry["path"], dtype=np.uint16, mode="r")
            self.shards.append(shard)

            T = shard.shape[0]
            # Stride = seq_len, window = seq_len+1 tokens. The last valid chunk
            # needs start + seq_len + 1 <= T, i.e. offset*seq_len + seq_len+1 <= T,
            # so the number of chunks is (T - 1) // seq_len.
            shard_chunks = (T - 1) // self.seq_len
            total_chunks += shard_chunks
            cumulative.append(total_chunks)

        self.cumulative = np.array(cumulative)   # for fast searchsorted
        self.n_chunks = int(total_chunks)        # Python int for __len__

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, targets), each shape [seq_len], dtype int64."""
        # Which shard does this global chunk index fall in? side="right" so
        # that idx == a boundary value maps to the NEXT shard (correct: the
        # chunk after a shard's last chunk is the first chunk of the next).
        shard_idx = int(np.searchsorted(self.cumulative, idx, side="right"))

        # Offset of this chunk WITHIN its shard (subtract prior shards' chunks).
        prev = self.cumulative[shard_idx - 1] if shard_idx > 0 else 0
        offset = idx - prev

        # Stride = seq_len (matches PretrainDataset). Read seq_len+1 tokens so
        # we get seq_len inputs and seq_len targets (shifted by one).
        start = offset * self.seq_len
        shard = self.shards[shard_idx]
        tokens = shard[start : start + self.seq_len + 1]

        # uint16 memmap slice -> int64 torch tensor. .astype(int64) also copies,
        # so torch isn't handed a read-only memmap view.
        input_ids = torch.from_numpy(tokens[:-1].astype(np.int64))   # [seq_len]
        targets   = torch.from_numpy(tokens[1:].astype(np.int64))    # [seq_len]
        return input_ids, targets