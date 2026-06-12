# src/data/sft_dataset.py
"""SFT dataset: loads instruction-response pairs with loss masking."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning.
    
    Returns (input_ids, targets, loss_mask) where loss_mask=-100 on
    prompt tokens so the model only learns from response tokens.
    
    Format:
        <|endoftext|>User: {instruction}<|endoftext|>Assistant: {response}<|endoftext|>
    
    Args:
        data_sources: List of (dataset_name, split, weight, instruction_key, response_key)
        max_seq_len: Maximum sequence length (longer examples are truncated)
        enc: tiktoken encoder
    """
    
    def __init__(
        self,
        data_sources: list[dict],
        max_seq_len: int = 1024,
        enc=None,
    ):
        if enc is None:
            enc = tiktoken.get_encoding("gpt2")
        self.enc = enc
        self.max_seq_len = max_seq_len
        self.eot = enc.eot_token  # use as separator
        
        # Load and mix all data sources
        self.examples = []
        for source in data_sources:
            examples = self._load_source(source)
            self.examples.extend(examples)
        
        print(f"SFT dataset: {len(self.examples):,} examples")
    
    def _load_source(self, source: dict) -> list[dict]:
        """Load examples from a HuggingFace dataset."""
        dataset = load_dataset(
            source["name"],
            split=source.get("split", "train"),
            streaming=False,
        )
        
        instruction_key = source.get("instruction_key", "instruction")
        response_key = source.get("response_key", "output")
        max_examples = source.get("max_examples", None)
        
        examples = []
        for i, item in enumerate(dataset):
            if max_examples and i >= max_examples:
                break

            # OpenHermes has nested conversations format
            if "conversations" in item:
                turns = item["conversations"]
                # Skip examples that don't have exactly human + gpt turns
                if len(turns) < 2:
                    continue
                instruction = turns[0]["value"]
                response = turns[1]["value"]
            else:
                instruction = item[instruction_key]
                response = item[response_key]

            examples.append({
                "instruction": instruction,
                "response": response,
            })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        example = self.examples[idx]
        
        # Format: EOT + "User: " + instruction + EOT + "Assistant: " + response + EOT
        # Using EOT as separator since it's already in the tiktoken GPT-2 vocab
        user_text = f"User: {example['instruction']}"
        asst_text = f"Assistant: {example['response']}"
        
        # Tokenize separately so we know the boundary
        user_tokens  = [self.eot] + self.enc.encode(user_text, disallowed_special=())  + [self.eot]
        asst_tokens  = self.enc.encode(asst_text, disallowed_special=()) + [self.eot]
        
        # Full sequence: user + assistant
        full_tokens = user_tokens + asst_tokens
        
        # Truncate if too long (keep as much of response as possible)
        if len(full_tokens) > self.max_seq_len + 1:
            full_tokens = full_tokens[:self.max_seq_len + 1]
        
        # input_ids: all but last token
        # targets: all but first token (shifted by 1)
        input_ids = full_tokens[:-1]
        targets   = full_tokens[1:]
        
        # Loss mask: -100 on prompt (user) tokens, keep response tokens
        prompt_len = len(user_tokens) - 1  # -1 because of the shift
        loss_mask  = [-100] * min(prompt_len, len(targets)) + [1] * max(0, len(targets) - prompt_len)
        
        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids  = input_ids  + [self.eot] * pad_len
        targets    = targets    + [-100]      * pad_len  # -100 on padding
        loss_mask  = loss_mask  + [-100]      * pad_len
        
        return (
            torch.tensor(input_ids,  dtype=torch.long),
            torch.tensor(targets,    dtype=torch.long),
            torch.tensor(loss_mask,  dtype=torch.long),
        )