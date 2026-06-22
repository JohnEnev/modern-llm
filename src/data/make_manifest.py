# make_manifest.py
"""Generate the V3 training manifest (run ONCE on the pod, then commit it).

The manifest is a JSONL file listing which shards belong to train vs val,
tagged by source. The training MIX RATIO is controlled here by how many
shards of each source you include in the train split.

Mix target (~43B train tokens at ~80/12/8 fineweb/code/math):
    fineweb : 340 shards x 100M = 34.0B   (~80%)
    python  :  49 shards x 100M =  5.0B   (~12.5%)
    math    :  40 shards x 100M =  4.0B   (~7.5%)
                                  -------
                                  43.0B total

Val: a small fixed held-out slice per source (per-source val loss).

Adjust the dir paths + counts to match what's actually on disk. Run:
    python make_manifest.py
then sanity-check the printed per-source counts before training.
"""

import os
import sys

# Allow "python make_manifest.py" from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.manifest_dataset import build_manifest_entries, write_manifest, load_manifest

SOURCE_DIRS = {
    "fineweb": "/workspace/data/v3/fineweb-edu/train",   # 800 fineweb shards
    "python":  "/workspace/data/v3/code_python_5b",      # 50 code shards
    "math":    "/workspace/data/v3/math_openwebmath_5b",             # 50 math shards
}

# How many TRAIN shards from each source (this sets the mix ratio)
TRAIN_SHARDS = {
    "fineweb": 340,   # 34B  (~79%)
    "python":  49,    # 5B   (~11%)
    "math":    40,    # 4B   (~9%)
}

# How many VAL shards from each source (held out, fixed forever)
VAL_SHARDS = {
    "fineweb": 3,     # ~300M
    "python":  1,     # ~100M
    "math":    1,     # ~100M
}

MANIFEST_OUT = "/workspace/data/v3/manifest.jsonl"
TOKENS_PER_SHARD = 100_000_000


def main():
    # Sanity-check the dirs exist and report how many shards each has,
    # so you catch a wrong path or a source with too few shards BEFORE
    # writing a manifest that references files that don't exist.
    import glob
    print("=" * 70)
    print("SHARD AVAILABILITY CHECK")
    print("=" * 70)
    ok = True
    for source, d in SOURCE_DIRS.items():
        n_avail = len(glob.glob(os.path.join(d, "*.bin")))
        n_need = TRAIN_SHARDS[source] + VAL_SHARDS[source]
        status = "OK" if n_avail >= n_need else "!! NOT ENOUGH"
        if n_avail < n_need:
            ok = False
        print(f"  {source:<8} dir={d}")
        print(f"           available={n_avail:<4} need(train+val)={n_need:<4} [{status}]")
    print("=" * 70)

    if not ok:
        print("ABORTING: at least one source has fewer shards than requested.")
        print("Fix the dir paths or lower the TRAIN_SHARDS/VAL_SHARDS counts.")
        sys.exit(1)

    # Build entries (val shards first, then disjoint train shards, per source)
    entries = build_manifest_entries(
        source_dirs=SOURCE_DIRS,
        train_shards_per_source=TRAIN_SHARDS,
        val_shards_per_source=VAL_SHARDS,
        tokens_per_shard=TOKENS_PER_SHARD,
    )

    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
    write_manifest(MANIFEST_OUT, entries)
    print(f"\nWrote {len(entries)} entries -> {MANIFEST_OUT}")

    # Read it back and print per-source/per-split counts as a final confirm.
    print("\nVerification (reading manifest back):")
    train_entries = load_manifest(MANIFEST_OUT, "train")
    val_entries = load_manifest(MANIFEST_OUT, "val")

    train_tokens = sum(e["tokens"] for e in train_entries)
    val_tokens = sum(e["tokens"] for e in val_entries)
    print(f"\n  train: {len(train_entries)} shards  (~{train_tokens/1e9:.1f}B tokens)")
    print(f"  val:   {len(val_entries)} shards  (~{val_tokens/1e9:.1f}B tokens)")

    # Confirm no shard path appears in both splits (train/val leakage check).
    train_paths = {e["path"] for e in train_entries}
    val_paths = {e["path"] for e in val_entries}
    overlap = train_paths & val_paths
    if overlap:
        print(f"\n  !! WARNING: {len(overlap)} shard(s) in BOTH train and val:")
        for p in list(overlap)[:5]:
            print(f"     {p}")
    else:
        print("\n  ✓ No train/val shard overlap")


if __name__ == "__main__":
    main()