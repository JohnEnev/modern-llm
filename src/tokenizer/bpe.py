"""Byte-Pair Encoding tokenizer trained from scratch."""

import json
from pathlib import Path


class BPETokenizer:
    """A minimal BPE tokenizer operating on raw UTF-8 bytes.
    
    Starts with a base vocabulary of 256 byte-level tokens and iteratively
    merges the most frequent adjacent pair until the target vocab size is reached.
    """

    # Reserved special token IDs (assigned after learned merges)
    EOS_TOKEN = "<|eos|>"

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # (pair) -> merged token ID
        self.vocab: dict[int, bytes] = {}              # token ID -> byte sequence
        self.eos_id: int | None = None

    def train(self, text: str, vocab_size: int) -> None:
        """Learn BPE merges from training text.
        
        Args:
            text: Raw training text.
            vocab_size: Target vocabulary size (must be > 256).
                        Final vocab will be vocab_size + 1 (for EOS token).
        """
        assert vocab_size > 256, "vocab_size must exceed the 256 base byte tokens"

        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = vocab_size - 256

        for merge_idx in range(num_merges):
            # Count adjacent pairs
            pair_counts: dict[tuple[int, int], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break  # sequence too short to merge further

            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = len(self.vocab)

            # Record merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to token sequence
            tokens = self._apply_merge(tokens, best_pair, new_id)

        # Add special tokens after learned vocab
        self.eos_id = len(self.vocab)
        self.vocab[self.eos_id] = self.EOS_TOKEN.encode("utf-8")

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode text into a list of token IDs.
        
        Args:
            text: Input text to tokenize.
            add_eos: If True, append EOS token at the end.
        """
        tokens = list(text.encode("utf-8"))

        for pair, new_id in self.merges.items():
            tokens = self._apply_merge(tokens, pair, new_id)

        if add_eos and self.eos_id is not None:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text.
        
        Args:
            ids: List of token IDs to decode.
        """
        # Filter out special tokens (EOS) before decoding
        byte_chunks = []
        for token_id in ids:
            if token_id == self.eos_id:
                continue
            byte_chunks.append(self.vocab[token_id])

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Save tokenizer to disk.
        
        Args:
            path: Directory path to save vocab and merges.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert tuple keys to strings for JSON serialization
        merges_serializable = {f"{p[0]},{p[1]}": v for p, v in self.merges.items()}
        vocab_serializable = {str(k): list(v) for k, v in self.vocab.items()}

        data = {
            "merges": merges_serializable,
            "vocab": vocab_serializable,
            "eos_id": self.eos_id,
        }

        with open(save_dir / "tokenizer.json", "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load tokenizer from disk.
        
        Args:
            path: Directory path containing saved tokenizer.
        """
        with open(Path(path) / "tokenizer.json", "r") as f:
            data = json.load(f)

        self.merges = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data["merges"].items()
        }
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self.eos_id = data["eos_id"]

    @staticmethod
    def _apply_merge(
        tokens: list[int],
        pair: tuple[int, int],
        new_id: int,
    ) -> list[int]:
        """Replace every occurrence of `pair` in `tokens` with `new_id`."""
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: Round-trip
    tok = BPETokenizer()
    tok.train("hello world hello world hello", vocab_size=260)

    encoded = tok.encode("hello world")
    decoded = tok.decode(encoded)
    assert decoded == "hello world", f"Round-trip failed: got '{decoded}'"
    print(f"Encoded: {encoded}  ->  Decoded: '{decoded}'")
    print("✓ Round-trip passed\n")

    # Test 2: Compression
    raw_len = len(list("hello world".encode("utf-8")))
    print(f"Compression: {raw_len} bytes -> {len(encoded)} tokens")
    assert len(encoded) < raw_len
    print("✓ Compression passed\n")

    # Test 3: Learned merges
    print("Learned merges:")
    for pair, new_id in tok.merges.items():
        label = tok.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  {pair} -> {new_id} ('{label}')")

    # Test 4: Unicode round-trip
    tok2 = BPETokenizer()
    tok2.train("café café café naïve naïve", vocab_size=262)
    assert tok2.decode(tok2.encode("café naïve")) == "café naïve"
    print("\n✓ Unicode round-trip passed")

    # Test 5: EOS token
    encoded_eos = tok.encode("hello", add_eos=True)
    assert encoded_eos[-1] == tok.eos_id
    assert tok.decode(encoded_eos) == "hello"
    print("✓ EOS token passed")

    # Test 6: Save/load round-trip
    tok.save("/tmp/test_tokenizer")
    tok_loaded = BPETokenizer()
    tok_loaded.load("/tmp/test_tokenizer")
    assert tok_loaded.decode(tok_loaded.encode("hello world")) == "hello world"
    print("✓ Save/load round-trip passed")