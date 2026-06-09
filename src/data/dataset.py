# src/data/dataset.py
"""Pre-tokenized dataset for causal language model pretraining."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PretrainDataset(Dataset):
    """Serves (input, target) chunks from pre-tokenized .bin shard files.
    
    Each shard is a flat array of uint16 token IDs. The dataset slices these
    into seq_len-sized chunks with a 1-token shift for next-token prediction.
    
    Args:
        data_dir: Directory containing .bin shard files.
        seq_len: Context window size (default 1024).
    """
    def __init__(self, data_dir: str, seq_len: int = 1024):
        self.seq_len = seq_len

        shard_paths = sorted(Path(data_dir).glob("*.bin"))
        assert len(shard_paths) > 0, f"No .bin files found in {data_dir}"

        # Memory-map each shard (no RAM cost until accessed)
        self.shards = []
        self.cumulative_chunks = [0]

        for path in shard_paths:
            shard = np.memmap(path, dtype=np.uint16, mode='r')
            self.shards.append(shard)
            # Each chunk needs seq_len + 1 tokens (input + 1 shifted target)
            num_chunks = (len(shard) - 1) // seq_len
            self.cumulative_chunks.append(self.cumulative_chunks[-1] + num_chunks)

    def __len__(self) -> int:
        return self.cumulative_chunks[-1]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        assert idx < len(self), f"Index {idx} out of range (size {len(self)})"

        # Find which shard this chunk lives in
        shard_idx = np.searchsorted(self.cumulative_chunks, idx, side='right') - 1

        # old method using for loop
        # for i, chunk in enumerate(self.total_chunks):
        #     if idx < chunk:
        #         shard_idx = i - 1
        #         break
        #     else:
        #         continue
        
        offset = idx - self.cumulative_chunks[shard_idx]

        # Read seq_len + 1 contiguous tokens
        start = offset * self.seq_len
        tokens = self.shards[shard_idx][start: start + self.seq_len + 1]

        # Convert to torch tensors
        tokens = torch.from_numpy(tokens.astype(np.int64))
        input_ids = tokens[:-1]    # tokens[0:seq_len]
        targets = tokens[1:]       # tokens[1:seq_len + 1]

        return (input_ids, targets)
    

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import os
 
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_len = 1024
 
        # Create two dummy shards (~10 chunks each)
        for i in range(2):
            tokens = np.random.randint(0, 32768, size=10 * seq_len + 1, dtype=np.uint16)
            tokens.tofile(os.path.join(tmpdir, f"shard_{i:04d}.bin"))
 
        dataset = PretrainDataset(tmpdir, seq_len=seq_len)
        print(f"Shards: {len(dataset.shards)}")
        print(f"Total chunks: {len(dataset)}")
 
        # Basic shape / dtype checks
        input_ids, targets = dataset[0]
        assert input_ids.shape == (seq_len,), f"Expected ({seq_len},), got {input_ids.shape}"
        assert targets.shape == (seq_len,), f"Expected ({seq_len},), got {targets.shape}"
        assert input_ids.dtype == torch.int64
        print("✓ Shape and dtype correct")
 
        # Verify the 1-token shift
        assert (input_ids[1:] == targets[:-1]).all(), "Shift relationship broken"
        print("✓ Input/target shift correct")
 
        # Verify cross-shard indexing
        first_shard_chunks = dataset.cumulative_chunks[1]
        x1, y1 = dataset[first_shard_chunks]  # first chunk of shard 1
        print(f"✓ Cross-shard access works (shard 1 starts at chunk {first_shard_chunks})")
 
        # DataLoader integration
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, seq_len)
        print(f"✓ DataLoader works: batch shape {batch_x.shape}")
 
        print("\nAll tests passed!")