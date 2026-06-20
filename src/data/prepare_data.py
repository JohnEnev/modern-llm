"""Pre-tokenize datasets and save as .bin shards for training.

Supports:
  - FineWeb-Edu
  - BigCode The Stack Python subset
  - Generic HuggingFace streaming datasets

Each shard is a flat numpy array of uint16 token IDs.
Documents are separated by GPT-2 EOT tokens.
"""

import os
import time
import argparse
from typing import Optional

import numpy as np
import tiktoken
from datasets import load_dataset


class TokenShardWriter:
    """Efficient uint16 shard writer.

    Avoids keeping a giant Python list of token IDs in memory.
    """

    def __init__(self, output_dir: str, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        os.makedirs(output_dir, exist_ok=True)

        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.pos = 0
        self.shard_idx = 0
        self.total_tokens = 0

    def add_tokens(self, tokens: list[int], max_total_tokens: Optional[int] = None):
        """Add tokens, flushing full shards as needed.

        If max_total_tokens is provided, tokens are truncated so total_tokens
        never exceeds it.
        """
        if max_total_tokens is not None:
            remaining_total = max_total_tokens - self.total_tokens
            if remaining_total <= 0:
                return
            if len(tokens) > remaining_total:
                tokens = tokens[:remaining_total]

        offset = 0
        n = len(tokens)

        while offset < n:
            remaining_shard = self.shard_size - self.pos
            take = min(remaining_shard, n - offset)

            chunk = np.asarray(tokens[offset:offset + take], dtype=np.uint16)
            self.buffer[self.pos:self.pos + take] = chunk

            self.pos += take
            self.total_tokens += take
            offset += take

            if self.pos == self.shard_size:
                self.flush(full=True)

    def flush(self, full: bool = False):
        """Write current buffer to disk."""
        if self.pos == 0:
            return

        arr = self.buffer if full else self.buffer[:self.pos].copy()
        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_idx:04d}.bin")
        arr.tofile(shard_path)

        print(
            f"Saved shard {self.shard_idx:04d} | "
            f"{arr.size:>12,} tokens | "
            f"{self.total_tokens:>14,} total | "
            f"{shard_path}",
            flush=True,
        )

        self.shard_idx += 1
        self.pos = 0


def load_streaming_hf_dataset(
    dataset_name: str,
    split: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    hf_token: bool = False,
):
    """Load a HuggingFace dataset in streaming mode.

    Args:
        dataset_name: HF dataset path, e.g. "HuggingFaceFW/fineweb-edu".
        split: Dataset split.
        name: Dataset config/name, e.g. "sample-100BT" for FineWeb-Edu.
        data_dir: Optional data_dir, e.g. "data/python" for The Stack.
        hf_token: Pass token=True for gated datasets / accepted terms.
    """
    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }

    if name is not None:
        kwargs["name"] = name

    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    if hf_token:
        kwargs["token"] = True

    return load_dataset(**kwargs)


def inspect_dataset(
    dataset_name: str,
    split: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    hf_token: bool = False,
    n: int = 3,
):
    """Print a few examples and keys from a streaming dataset."""
    dataset = load_streaming_hf_dataset(
        dataset_name=dataset_name,
        split=split,
        name=name,
        data_dir=data_dir,
        hf_token=hf_token,
    )

    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)
    print(f"dataset_name: {dataset_name}")
    print(f"name:         {name}")
    print(f"data_dir:     {data_dir}")
    print(f"split:        {split}")
    print("=" * 80)

    for i, doc in zip(range(n), dataset):
        print(f"\n--- Example {i} ---")
        print("keys:", list(doc.keys()))

        for key, value in doc.items():
            if isinstance(value, str):
                preview = value[:500].replace("\n", "\\n")
                print(f"{key}: {preview!r}")
            else:
                print(f"{key}: {type(value).__name__} = {value}")


def prepare_streaming_dataset(
    output_dir: str,
    dataset_name: str,
    text_field: str,
    num_tokens: int,
    shard_size: int = 100_000_000,
    split: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    hf_token: bool = False,
    min_chars: int = 0,
    log_every_docs: int = 10_000,
):
    """Stream a text/code dataset, tokenize, and save uint16 shards.

    Args:
        output_dir: Directory to write shard files.
        dataset_name: HF dataset path.
        text_field: Field containing the text/code, e.g. "text" or "content".
        num_tokens: Stop after writing this many tokens.
        shard_size: Tokens per shard file.
        split: HF split.
        name: Optional HF config/name.
        data_dir: Optional HF data_dir.
        hf_token: Whether to pass HF token.
        min_chars: Skip documents shorter than this many characters.
        log_every_docs: Progress print frequency.
    """
    os.makedirs(output_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print("=" * 80)
    print("PREPARE STREAMING DATASET")
    print("=" * 80)
    print(f"dataset_name: {dataset_name}")
    print(f"name:         {name}")
    print(f"data_dir:     {data_dir}")
    print(f"split:        {split}")
    print(f"text_field:   {text_field}")
    print(f"output_dir:   {output_dir}")
    print(f"num_tokens:   {num_tokens:,}")
    print(f"shard_size:   {shard_size:,}")
    print(f"min_chars:    {min_chars:,}")
    print("=" * 80)

    dataset = load_streaming_hf_dataset(
        dataset_name=dataset_name,
        split=split,
        name=name,
        data_dir=data_dir,
        hf_token=hf_token,
    )

    writer = TokenShardWriter(output_dir=output_dir, shard_size=shard_size)

    docs_seen = 0
    docs_used = 0
    docs_skipped = 0
    t0 = time.time()

    for doc in dataset:
        docs_seen += 1

        if text_field not in doc:
            raise KeyError(
                f"text_field={text_field!r} not found. "
                f"Available keys: {list(doc.keys())}"
            )

        text = doc[text_field]

        if text is None:
            docs_skipped += 1
            continue

        if not isinstance(text, str):
            text = str(text)

        if len(text) < min_chars:
            docs_skipped += 1
            continue

        # Encode special-token-looking substrings as plain text.
        tokens = enc.encode(text, disallowed_special=())
        tokens.append(eot)

        writer.add_tokens(tokens, max_total_tokens=num_tokens)
        docs_used += 1

        if docs_seen % log_every_docs == 0:
            elapsed = time.time() - t0
            tok_per_s = writer.total_tokens / max(elapsed, 1e-6)
            print(
                f"docs_seen={docs_seen:,} | docs_used={docs_used:,} | "
                f"skipped={docs_skipped:,} | tokens={writer.total_tokens:,} | "
                f"tok/s={tok_per_s:,.0f}",
                flush=True,
            )

        if writer.total_tokens >= num_tokens:
            break

    writer.flush(full=False)

    elapsed = time.time() - t0
    tok_per_s = writer.total_tokens / max(elapsed, 1e-6)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"tokens:       {writer.total_tokens:,}")
    print(f"shards:       {writer.shard_idx:,}")
    print(f"docs_seen:    {docs_seen:,}")
    print(f"docs_used:    {docs_used:,}")
    print(f"docs_skipped: {docs_skipped:,}")
    print(f"elapsed:      {elapsed / 3600:.2f} hours")
    print(f"tok/s:        {tok_per_s:,.0f}")
    print(f"output_dir:   {output_dir}")
    print("=" * 80)


def prepare_fineweb_edu(
    output_dir: str,
    num_tokens: int = 10_000_000_000,
    shard_size: int = 100_000_000,
    sample: str = "sample-100BT",
):
    prepare_streaming_dataset(
        output_dir=output_dir,
        dataset_name="HuggingFaceFW/fineweb-edu",
        name=sample,
        split="train",
        text_field="text",
        num_tokens=num_tokens,
        shard_size=shard_size,
    )


def prepare_python_code(
    output_dir: str,
    num_tokens: int = 5_000_000_000,
    shard_size: int = 100_000_000,
    hf_token: bool = False,
):
    """Prepare Python code from BigCode The Stack dedup.

    Uses the Python subset via data_dir="data/python".
    The main text/code field is expected to be "content".
    """
    prepare_streaming_dataset(
        output_dir=output_dir,
        dataset_name="bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train",
        text_field="content",
        num_tokens=num_tokens,
        shard_size=shard_size,
        hf_token=hf_token,
        min_chars=50,
    )


def prepare_dummy(output_dir: str, num_shards: int = 3, tokens_per_shard: int = 50_000):
    """Create dummy shards with random token IDs for local testing."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_shards):
        tokens = np.random.randint(0, 50257, size=tokens_per_shard, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"shard_{i:04d}.bin")
        tokens.tofile(shard_path)
        print(f"Created {shard_path} ({tokens_per_shard:,} tokens)")

    print(f"\nDummy data ready: {num_shards} shards in {output_dir}/")


def test_prepare_and_load():
    """End-to-end test: create dummy shards, load with PretrainDataset."""
    import tempfile

    from dataset import PretrainDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_per_shard = 10_250
        prepare_dummy(tmpdir, num_shards=2, tokens_per_shard=tokens_per_shard)

        shard_files = sorted(os.listdir(tmpdir))
        assert len(shard_files) == 2

        for f in shard_files:
            path = os.path.join(tmpdir, f)
            file_size = os.path.getsize(path)
            expected_size = tokens_per_shard * 2
            assert file_size == expected_size, f"Expected {expected_size}B, got {file_size}B"

        print("✓ Shard files have correct size")

        shard = np.memmap(os.path.join(tmpdir, shard_files[0]), dtype=np.uint16, mode="r")
        assert shard.max() < 65536
        print("✓ Token IDs fit in uint16")

        dataset = PretrainDataset(tmpdir, seq_len=1024)
        input_ids, targets = dataset[0]
        assert input_ids.shape == (1024,)
        assert targets.shape == (1024,)
        print("✓ Dataset loads shards correctly")

        for i in range(len(dataset)):
            x, y = dataset[i]

        print(f"✓ All {len(dataset)} chunks accessible")
        print("\nAll tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize data for LLM training")

    parser.add_argument(
        "mode",
        choices=["fineweb", "code_python", "generic", "inspect", "dummy", "test"],
    )

    parser.add_argument("--output_dir", type=str, default="data/out")
    parser.add_argument("--num_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--shard_size", type=int, default=100_000_000)

    # FineWeb
    parser.add_argument("--sample", type=str, default="sample-100BT")

    # Generic HF dataset options
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--hf_token", action="store_true")
    parser.add_argument("--min_chars", type=int, default=0)
    parser.add_argument("--inspect_n", type=int, default=3)

    args = parser.parse_args()

    if args.mode == "fineweb":
        prepare_fineweb_edu(
            output_dir=args.output_dir,
            num_tokens=args.num_tokens,
            shard_size=args.shard_size,
            sample=args.sample,
        )

    elif args.mode == "code_python":
        prepare_python_code(
            output_dir=args.output_dir,
            num_tokens=args.num_tokens,
            shard_size=args.shard_size,
            hf_token=args.hf_token,
        )

    elif args.mode == "generic":
        if args.dataset is None:
            raise ValueError("--dataset is required for generic mode")

        prepare_streaming_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            name=args.name,
            data_dir=args.data_dir,
            split=args.split,
            text_field=args.text_field,
            num_tokens=args.num_tokens,
            shard_size=args.shard_size,
            hf_token=args.hf_token,
            min_chars=args.min_chars,
        )

    elif args.mode == "inspect":
        if args.dataset is None:
            raise ValueError("--dataset is required for inspect mode")

        inspect_dataset(
            dataset_name=args.dataset,
            name=args.name,
            data_dir=args.data_dir,
            split=args.split,
            hf_token=args.hf_token,
            n=args.inspect_n,
        )

    elif args.mode == "dummy":
        prepare_dummy(output_dir=args.output_dir)

    elif args.mode == "test":
        test_prepare_and_load()


if __name__ == "__main__":
    main()