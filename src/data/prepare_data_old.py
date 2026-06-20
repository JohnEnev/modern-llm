"""Pre-tokenize FineWeb-Edu and save as .bin shards for training."""

import os
import numpy as np
import tiktoken


def prepare_fineweb_edu(
    output_dir: str,
    num_tokens: int = 10_000_000_000,  # 10B tokens
    shard_size: int = 100_000_000,     # 100M tokens per shard (~200MB at uint16)
    sample: str = "sample-100BT", 
):
    """Stream FineWeb-Edu, tokenize with tiktoken, save as .bin shards.

    Each shard is a flat numpy array of uint16 token IDs. Documents are
    separated by EOT tokens and concatenated end-to-end.

    Args:
        output_dir: Directory to write shard files.
        num_tokens: Stop after accumulating this many tokens.
        shard_size: Number of tokens per shard file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # GPT-2 tokenizer (~50k vocab, fits in uint16)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    # Stream dataset — no full download needed
    from datasets import load_dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample,
        split="train",
        streaming=True,
    )

    buffer = []
    shard_idx = 0
    total_tokens = 0

    for doc in dataset:
        # Tokenize document and append end-of-text separator
        tokenised_doc = enc.encode(doc["text"], disallowed_special=()) # make sure special tokens are encoded as plain text
        buffer.extend(tokenised_doc)
        buffer.append(eot)
        total_tokens += len(tokenised_doc) + 1  # +1 for eot

        # Write full shards to disk as they fill up
        while len(buffer) >= shard_size:
            shard_tokens = buffer[:shard_size]
            buffer = buffer[shard_size:]

            shard_array = np.array(shard_tokens, dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
            shard_array.tofile(shard_path)

            shard_idx += 1
            print(f"Saved shard {shard_idx:>4d} | {total_tokens:>13,} tokens total")

        if total_tokens >= num_tokens:
            break

    # Save any remaining tokens as a final shard
    if buffer:
        shard_array = np.array(buffer, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
        shard_array.tofile(shard_path)
        shard_idx += 1
        print(f"Saved shard {shard_idx:>4d} | {total_tokens:>13,} tokens total (final)")

    print(f"\nDone: {total_tokens:,} tokens across {shard_idx} shards in {output_dir}/")


def prepare_dummy(output_dir: str, num_shards: int = 3, tokens_per_shard: int = 50_000):
    """Create dummy shards with random token IDs for local testing.

    Args:
        output_dir: Directory to write shard files.
        num_shards: Number of shards to create.
        tokens_per_shard: Tokens per shard.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_shards):
        tokens = np.random.randint(0, 50257, size=tokens_per_shard, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"shard_{i:04d}.bin")
        tokens.tofile(shard_path)
        print(f"Created {shard_path} ({tokens_per_shard:,} tokens)")

    print(f"\nDummy data ready: {num_shards} shards in {output_dir}/")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_prepare_and_load():
    """End-to-end test: create dummy shards, load with PretrainDataset."""
    import tempfile

    # Lazy import to avoid circular dependency
    from dataset import PretrainDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy shards
        tokens_per_shard = 10_250  # gives 10 chunks at seq_len=1024
        prepare_dummy(tmpdir, num_shards=2, tokens_per_shard=tokens_per_shard)

        # Verify shard files exist and have correct size
        shard_files = sorted(os.listdir(tmpdir))
        assert len(shard_files) == 2
        for f in shard_files:
            path = os.path.join(tmpdir, f)
            file_size = os.path.getsize(path)
            expected_size = tokens_per_shard * 2  # uint16 = 2 bytes
            assert file_size == expected_size, f"Expected {expected_size}B, got {file_size}B"
        print("✓ Shard files have correct size")

        # Verify token values are in range
        shard = np.memmap(os.path.join(tmpdir, shard_files[0]), dtype=np.uint16, mode="r")
        assert shard.max() < 65536, "Token IDs must fit in uint16"
        print("✓ Token IDs fit in uint16")

        # Load with PretrainDataset and verify shapes
        dataset = PretrainDataset(tmpdir, seq_len=1024)
        input_ids, targets = dataset[0]
        assert input_ids.shape == (1024,)
        assert targets.shape == (1024,)
        print("✓ Dataset loads shards correctly")

        # Verify we can iterate through all chunks
        for i in range(len(dataset)):
            x, y = dataset[i]
        print(f"✓ All {len(dataset)} chunks accessible")

        print("\nAll tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize data for LLM training")
    parser.add_argument(
        "mode",
        choices=["fineweb", "dummy", "test"],
        help="'fineweb' to tokenize FineWeb-Edu, 'dummy' for test data, 'test' to run tests",
    )
    parser.add_argument("--output_dir", type=str, default="data/fineweb-edu")
    parser.add_argument("--num_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--sample", type=str, default="sample-100BT")
    args = parser.parse_args()

    if args.mode == "fineweb":
        prepare_fineweb_edu(
            output_dir=args.output_dir,
            num_tokens=args.num_tokens,
            shard_size=args.shard_size,
            sample=args.sample,
        )
    elif args.mode == "dummy":
        prepare_dummy(output_dir=args.output_dir)
    elif args.mode == "test":
        test_prepare_and_load()