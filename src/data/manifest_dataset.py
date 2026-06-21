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
"""

import os
import json
import numpy as np
import glob
from collections import Counter
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str, split: str) -> list[dict]:
    """Read a JSONL manifest and return entries matching `split`."""
    with open(manifest_path, 'r') as json_file:
        entries = [json.loads(line) for line in f]

    matching = [e for e in entries if e["split"] == split]

    # count per source generically
    counts = Counter(e["source"] for e in matching)
    print(f"[{split}] shards per source: {dict(counts)}")

    return matching



def write_manifest(
    manifest_path: str,
    entries: list[dict],
):
    """Write a list of entry dicts as JSONL (one JSON object per line)."""

    with open(manifest_path, 'w') as jsonl_file:
        for entry in entries:
            json.dump(entry, jsonl_file)
            jsonl_file.write('\n')

def build_manifest_entries(
    source_dirs: dict[str, str],
    train_shards_per_source: dict[str, int],
    val_shards_per_source: dict[str, int],
    tokens_per_shard: int = 100_000_000,
) -> list[dict]:
    """Construct manifest entries from per-source shard directories.

    Args:
        source_dirs: {"fineweb": "/path/to/fineweb/train", "python": ...}
        train_shards_per_source: {"fineweb": 320, "python": 50, "math": 30}
        val_shards_per_source:   {"fineweb": 3, "python": 1, "math": 1}

    Returns a list of entry dicts (source/split/path/tokens) ready to write.

    TODOs:
        - for each source:
            * list the .bin shards in its dir, sorted
            * assign the FIRST val_shards_per_source[source] shards to "val"
            * assign the NEXT train_shards_per_source[source] shards to "train"
              (val and train must NOT overlap — slice carefully)
            * build an entry dict for each with source/split/path/tokens
        - return all entries
        - (this is a helper you run ONCE to generate the manifest, then commit it)
    """
    entries = []

    for source, dir_path in source_dirs.items():
        # list all .bin shards in this source's dir, sorted for determinism
        shards = sorted(glob.glob(os.path.join(dir_path, "*.bin")))

        n_val = val_shards_per_source[source]
        n_train = train_shards_per_source[source]

        # first n_val shards -> val ; next n_train shards -> train (no overlap)
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

    For TRAIN: load all shards for the split, treat as one big token pool,
    chunk into seq_len+1 windows. Shuffling is handled by the DataLoader
    (shuffle=True) at the sequence-index level.

    For per-source VAL: construct ONE ManifestDataset per source (filter the
    manifest entries to that source) so each source's val loss is separate.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        seq_len: int = 1024,
        source_filter: str | None = None,
    ):
        """
        Args:
            manifest_path: path to the JSONL manifest
            split: "train" or "val"
            seq_len: sequence length (chunk size)
            source_filter: if set, keep only entries from this source
                           (used to build per-source val datasets)

        TODOs:
            - load entries via load_manifest(manifest_path, split)
            - if source_filter is not None, keep only entries whose
              source == source_filter
            - store the list of shard paths
            - memory-map each shard (np.memmap, dtype=uint16, mode="r")
              so you don't load 40B tokens into RAM at once
            - figure out how many seq_len-chunks each shard yields, and build
              an index mapping global_chunk_idx -> (shard_idx, offset_within_shard)
              (this is what lets __getitem__ find the right chunk without
               concatenating everything in memory)
            - store total number of chunks as self.n_chunks
        """
        # TODO: implement
        raise NotImplementedError

    def __len__(self) -> int:
        """Return total number of seq_len chunks across all shards.

        TODOs:
            - return self.n_chunks
        """
        # TODO: implement
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, targets) for chunk idx, each shape [seq_len].

        TODOs:
            - map idx -> (shard_idx, offset) using the index built in __init__
            - read seq_len+1 tokens from that shard's memmap starting at offset
            - input_ids  = tokens[:-1]  (seq_len)
            - targets    = tokens[1:]   (seq_len, shifted by one)
            - convert to torch LongTensors and return

        Note: matches the existing PretrainDataset interface so the trainer
        doesn't need to change how it consumes batches.
        """
        # TODO: implement
        raise NotImplementedError