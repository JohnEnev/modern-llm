import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock
from .rmsnorm import RMSNorm

class GPTConfig:
    """Configuration for GPT model"""
    def __init__(
            self,
            vocab_size: int = 50304,
            d_model: int = 1024,
            n_layers: int = 24,
            n_heads: int = 16,
            n_kv_heads: int = 4,
            dropout: float = 0.0,
            max_seq_len: int = 1024,
            use_flash: bool = True,
            tie_weights: bool = True,
            use_qk_norm: bool = True,
            use_diff_attn: bool = True,
            use_mhc: bool = False,
            n_streams: int = 2,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.tie_weights = tie_weights
        self.use_qk_norm = use_qk_norm
        self.use_diff_attn = use_diff_attn
        self.use_mhc = use_mhc
        self.n_streams = n_streams
        
class GPT(nn.Module):
    """
    GPT decoder-only transformer with modern architecture.
    
    Components:
    - Token embeddings (no positional embeddings - RoPE handles this)
    - Stack of transformer blocks
    - Final RMSNorm
    - Language modeling head
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model) # [vocab_size, d_model]

        # Dropout after embeddings, if specified
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None # [batch, seq_len, d_model]

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                layer_idx=i,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_flash=config.use_flash,
                use_qk_norm=config.use_qk_norm,
                use_diff_attn=config.use_diff_attn, 
                use_mhc = config.use_mhc,
                n_streams = config.n_streams,
            )
            for i in range(config.n_layers)
        ]) # List of [TransformerBlock] of length n_layers, each block processes [batch, seq_len, d_model]

        # Final RMSNorm
        self.norm = RMSNorm(config.d_model) # [d_model]

        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # [d_model, vocab_size]

        # Optionally tie weights of token embeddings and language modeling head
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        if config.use_mhc:
            read_logits = torch.full((config.n_streams,), -2.0)
            read_logits[0] = 2.0
            self.final_read_logits = nn.Parameter(read_logits)
        else:
            self.final_read_logits = None


        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for better training.
        
        Standard practice for transformers:
        - Linear layers: normal distribution with std=0.02
        - Embeddings: normal distribution with std=0.02
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: torch.Tensor,
            targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            targets: Target token IDs [batch, seq_len] (optional, for training)
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided), else None
        """
        # Step 1 - Get token embeddings
        x = self.token_embeddings(input_ids) # [batch, seq_len, d_model]

        # Step 2 - Apply dropout to embeddings, if specified
        if self.emb_dropout is not None:
            x = self.emb_dropout(x) # [batch, seq_len, d_model]

        # Step 3 - Pass through transformer blocks - or mHC streams
        if self.config.use_mhc:
            # Initialize S streams as copies of x: [S, B, T, D]
            streams = x.unsqueeze(0).repeat(self.config.n_streams, 1, 1, 1)
            
            for block in self.blocks:
                streams = block(streams)
            
            # Learned final readout over streams
            read_weights = F.softmax(self.final_read_logits, dim=0)
            x = torch.einsum('s,sbtd->btd', read_weights, streams)

        else:
            for block in self.blocks:
                x = block(x) # [batch, seq_len, d_model]

        # Step 4 - Final RMSNorm
        x = self.norm(x) # [batch, seq_len, d_model]

        # Step 5 - Language modeling head to get logits
        logits = self.lm_head(x) # [batch, seq_len, vocab_size]

        # Step 6 - If targets provided, compute cross-entropy loss
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            # logits: [batch*seq_len, vocab_size], targets: [batch*seq_len]
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100 # Ignore padding tokens if any
            )

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated sequence [batch, seq_len + max_new_tokens]
        """
        self.eval() # Set to eval mode for generation (disables dropout)
        for _ in range(max_new_tokens):
            # Crop input_ids if longer than max_seq_len
            input_ids_crop = input_ids[:, -self.config.max_seq_len:]
            # Get logits for current input
            logits, _ = self(input_ids_crop) # [batch, seq_len_crop, vocab_size]
            # We only care about the last token's logits for sampling
            logits = logits[:, -1, :]  # [batch, vocab_size]
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1) # [batch, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1) # [batch, 1]
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1) # [batch, seq_len + 1]
        return input_ids
        

    
    def count_parameters(self) -> dict:
        """Count parameters in different components."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        counts = {
            'embeddings': count_params(self.token_embeddings),
            'blocks': sum(count_params(block) for block in self.blocks),
            'final_norm': count_params(self.norm),
            'lm_head': 0 if self.config.tie_weights else count_params(self.lm_head),
            'total': sum(p.numel() for p in self.parameters())
        }
        
        return counts


# ============================================================================
# TESTS
# ============================================================================

def test_gpt_model():
    """Test GPT model implementation."""
    print("="*60)
    print("Testing GPT Model")
    print("="*60)
    
    # Small config for testing
    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        max_seq_len=128,
        use_flash=False,
        tie_weights=True
    )
    
    print("\n1. Testing model initialization...")
    model = GPT(config)
    print("   ✓ Model created successfully")
    
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when targets not provided"
    print("   ✓ Forward pass works")
    print(f"   ✓ Output shape: {logits.shape}")
    
    print("\n3. Testing training mode (with loss)...")
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, targets)
    
    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.item() > 0, "Loss should be positive"
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    
    print("\n4. Testing backward pass...")
    loss.backward()
    
    # Check that gradients exist for key parameters
    assert model.token_embeddings.weight.grad is not None
    assert not torch.isnan(model.token_embeddings.weight.grad).any()
    print("   ✓ Gradients flow correctly")
    
    print("\n5. Testing weight tying...")
    if config.tie_weights:
        assert model.token_embeddings.weight is model.lm_head.weight, \
            "Weights should be tied (same tensor)"
        print("   ✓ Weight tying verified")
    
    print("\n6. Testing generation...")
    model.eval()
    start_tokens = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    
    assert generated.shape == (1, 15), f"Expected (1, 15), got {generated.shape}"
    print(f"   ✓ Generation works: {generated.shape}")
    
    print("\n7. Testing parameter count...")
    param_counts = model.count_parameters()
    print(f"   Total parameters: {param_counts['total']:,}")
    print(f"   - Embeddings: {param_counts['embeddings']:,}")
    print(f"   - Transformer blocks: {param_counts['blocks']:,}")
    print(f"   - Final norm: {param_counts['final_norm']:,}")
    print(f"   - LM head: {param_counts['lm_head']:,} (tied: {config.tie_weights})")
    
    print("\n8. Testing different sequence lengths...")
    for test_seq_len in [5, 20, 50]:
        input_test = torch.randint(0, config.vocab_size, (1, test_seq_len))
        logits_test, _ = model(input_test)
        assert logits_test.shape == (1, test_seq_len, config.vocab_size)
    print("   ✓ Works with variable sequence lengths")
    
    print("\n" + "="*60)
    print("GPT Model Tests Passed! ✓")
    print("="*60)


def test_full_350m_model():
    """Test creating a full 350M parameter model."""
    print("\n" + "="*60)
    print("Testing Full 350M Model Configuration")
    print("="*60)
    
    # Full 350M config
    config = GPTConfig(
    vocab_size=50304,
    d_model=1024, n_layers=24, n_heads=16,
    dropout=0.0, max_seq_len=1024, use_flash=True, tie_weights=True,
    )   
    
    print("\n1. Creating 350M model...")
    model = GPT(config)
    print("   ✓ Model created")
    
    print("\n2. Counting parameters...")
    param_counts = model.count_parameters()
    total_params = param_counts['total']
    print(f"   Total parameters: {total_params:,}")
    print(f"   Target: ~315,000,000")
    
    # Check if we're in the right ballpark (340M - 360M)
    assert 310_000_000 < total_params < 325_000_000, \
        f"Expected ~350M params, got {total_params:,}"
    print("   ✓ Parameter count in target range")
    
    print("\n3. Testing forward pass on small batch...")
    # Small batch to test (don't want to OOM)
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    logits, _ = model(input_ids)
    
    assert logits.shape == (1, 128, config.vocab_size)
    print("   ✓ Forward pass works")
    
    print("\n4. Model breakdown:")
    print(f"   - Embeddings: {param_counts['embeddings']:,} " +
          f"({param_counts['embeddings']/total_params*100:.1f}%)")
    print(f"   - Blocks: {param_counts['blocks']:,} " +
          f"({param_counts['blocks']/total_params*100:.1f}%)")
    print(f"   - Other: {param_counts['final_norm']:,}")
    
    print("\n" + "="*60)
    print("350M Model Tests Passed! ✓")
    print("="*60)

def test_gpt_mhc_model():
    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        dropout=0.0,
        max_seq_len=128,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=True,
        use_diff_attn=True,
        use_mhc=True,
        n_streams=2,
    )

    model = GPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    print("✓ GPT with mHC works")


def run_all_tests():
    """Run all GPT tests."""
    print("\n" + "="*70)
    print(" "*20 + "GPT MODEL TESTS")
    print("="*70 + "\n")
    
    test_gpt_model()
    test_full_350m_model()
    test_gpt_mhc_model()
    
    print("\n" + "="*70)
    print(" "*15 + "ALL GPT MODEL TESTS PASSED! 🎉")
    print("="*70)
    print("\nYou now have a complete modern transformer implementation!")
    print("Ready to move on to training infrastructure!")


if __name__ == "__main__":
    run_all_tests()