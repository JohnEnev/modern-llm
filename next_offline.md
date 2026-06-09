# Offline Learning Guide: Transformer Block → Embeddings → GPT Model

Work through each section in order. Try to answer questions and implement code before checking answers at the bottom.

---

# PART 1: Transformer Block (Pre-LN Architecture)

## Conceptual Questions

Try to answer these before looking at the code or answers:

**Q1:** What is the difference between Pre-LN and Post-LN architecture?

Pre-LN (modern):
```
x = x + SubLayer(Norm(x))
```

Post-LN (original Transformer):
```
x = Norm(x + SubLayer(x))
```

Which one has better gradient flow? Why is Pre-LN preferred for deep models?

**Q2:** A transformer block has two residual connections. Draw the computational graph showing:
- Input x
- First residual: attention path
- Second residual: MLP path
- Where normalization is applied
- Output

**Q3:** For a model with d_model=1024:
- Attention has 4 projection matrices (Q, K, V, O): How many parameters?
- SwiGLU has 3 projection matrices (gate, up, down with 8/3 expansion): How many parameters?
- RMSNorm has 2 gamma vectors (one before attention, one before MLP): How many parameters?
- What percentage of the block's parameters are in the MLP vs attention?

**Q4:** Why do we apply RMSNorm BEFORE the sub-layer (attention or MLP) in Pre-LN?
What would happen if we applied it AFTER?

**Q5:** Residual connections allow gradients to "skip" layers. How does this help with:
- Vanishing gradients?
- Training very deep networks (e.g., 50+ layers)?
- Gradient magnitude stability?

---

## Coding Challenge: Implement Transformer Block

Create `src/model/block.py`:

```python
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .swiglu import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Transformer block with Pre-LN architecture.
    
    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_flash: bool = False
    ):
        super().__init__()
        
        # TODO: Initialize RMSNorm before attention
        # self.norm1 = ...
        
        # TODO: Initialize multi-head attention
        # self.attention = ...
        
        # TODO: Initialize RMSNorm before MLP
        # self.norm2 = ...
        
        # TODO: Initialize SwiGLU MLP
        # self.mlp = ...
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # TODO: First residual path - attention
        # Step 1: Apply normalization
        # Step 2: Pass through attention
        # Step 3: Add residual connection
        # x = x + self.attention(self.norm1(x))
        
        # TODO: Second residual path - MLP
        # Step 1: Apply normalization  
        # Step 2: Pass through MLP
        # Step 3: Add residual connection
        # x = x + self.mlp(self.norm2(x))
        
        # return x
        pass


# ============================================================================
# TESTS
# ============================================================================

def test_transformer_block():
    """Test transformer block implementation."""
    print("="*60)
    print("Testing Transformer Block")
    print("="*60)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    print("\n1. Testing initialization...")
    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        use_flash=False
    )
    print("   ✓ Block created successfully")
    
    print("\n2. Testing forward pass...")
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("   ✓ Output shape correct")
    
    print("\n3. Testing residual connections (gradient flow)...")
    x_grad = torch.randn(1, seq_len, d_model, requires_grad=True)
    output_grad = block(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "Gradients should flow through residuals"
    assert not torch.isnan(x_grad.grad).any(), "Gradients should not be NaN"
    print("   ✓ Gradients flow correctly through residual connections")
    
    print("\n4. Testing with different sequence lengths...")
    for test_seq_len in [5, 20, 50, 100]:
        x_test = torch.randn(1, test_seq_len, d_model)
        output_test = block(x_test)
        assert output_test.shape == (1, test_seq_len, d_model), \
            f"Failed for seq_len={test_seq_len}"
    print("   ✓ Works with variable sequence lengths")
    
    print("\n5. Testing parameter count...")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Expected breakdown:
    # - Attention: 4 * d_model^2 = 4 * 512^2 = 1,048,576
    # - SwiGLU: ~8 * d_model^2 = ~8 * 512^2 = ~2,097,152
    # - RMSNorm (2x): 2 * d_model = 2 * 512 = 1,024
    # Total: ~3,146,752
    
    expected_params = 12 * d_model**2 + 2 * d_model
    tolerance = d_model * 200  # Allow some rounding in SwiGLU
    
    assert abs(total_params - expected_params) < tolerance, \
        f"Expected ~{expected_params:,}, got {total_params:,}"
    print(f"   ✓ Parameter count reasonable (expected ~{expected_params:,})")
    
    print("\n6. Testing train vs eval mode...")
    block.train()
    with torch.no_grad():
        output_train = block(x)
    
    block.eval()
    with torch.no_grad():
        output_eval = block(x)
    
    # Outputs should potentially differ due to dropout
    print("   ✓ Train/eval mode working")
    
    print("\n7. Testing with Flash Attention...")
    block_flash = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        use_flash=True
    )
    output_flash = block_flash(x)
    assert output_flash.shape == (batch_size, seq_len, d_model)
    print("   ✓ Flash Attention mode works")
    
    print("\n" + "="*60)
    print("Transformer Block Tests Passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_transformer_block()
```

---

# PART 2: Token Embeddings and Output Head

## Conceptual Questions

**Q6:** Why don't we use positional embeddings in addition to token embeddings?
What component handles positional information in our model?

**Q7:** Weight tying is when the embedding matrix and output projection share weights:
```
token_embeddings.weight = lm_head.weight
```

Why is this beneficial?
- Parameter efficiency?
- Model performance?
- Theoretical motivation?

**Q8:** For a vocabulary of 32,768 tokens and d_model=1024:
- Token embedding matrix size?
- If we use weight tying, how many parameters do we save?
- What percentage of a 350M parameter model is the embedding?

**Q9:** During generation, we:
1. Get embeddings for input tokens
2. Pass through transformer blocks
3. Apply final norm
4. Project to vocabulary with lm_head
5. Sample from the logits

At step 4, what is the shape transformation?
[batch, seq_len, d_model] → ?

**Q10:** Why do we apply a final RMSNorm before the output projection?
What would happen if we skipped it?

---

## Coding Challenge: Embeddings and GPT Model

Create `src/model/gpt.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock
from .rmsnorm import RMSNorm


class GPTConfig:
    """Configuration for GPT model."""
    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        use_flash: bool = False,
        tie_weights: bool = True
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.tie_weights = tie_weights


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
        
        # TODO: Token embeddings
        # Shape: [vocab_size, d_model]
        # self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # TODO: Dropout after embeddings (if dropout > 0)
        # self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # TODO: Stack of transformer blocks
        # Use nn.ModuleList to create n_layers blocks
        # self.blocks = nn.ModuleList([
        #     TransformerBlock(
        #         d_model=config.d_model,
        #         n_heads=config.n_heads,
        #         dropout=config.dropout,
        #         max_seq_len=config.max_seq_len,
        #         use_flash=config.use_flash
        #     )
        #     for _ in range(config.n_layers)
        # ])
        
        # TODO: Final RMSNorm
        # self.final_norm = RMSNorm(config.d_model)
        
        # TODO: Language modeling head (projects to vocabulary)
        # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # TODO: Weight tying (if enabled)
        # if config.tie_weights:
        #     self.lm_head.weight = self.token_embeddings.weight
        
        # TODO: Initialize weights
        # self.apply(self._init_weights)
        
        pass
    
    def _init_weights(self, module):
        """
        Initialize weights for better training.
        
        Standard practice for transformers:
        - Linear layers: normal distribution with std=0.02
        - Embeddings: normal distribution with std=0.02
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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
        # TODO: Step 1 - Get token embeddings
        # x = self.token_embeddings(input_ids)  # [batch, seq_len, d_model]
        
        # TODO: Step 2 - Apply dropout to embeddings (if training)
        # if self.emb_dropout is not None:
        #     x = self.emb_dropout(x)
        
        # TODO: Step 3 - Pass through transformer blocks
        # for block in self.blocks:
        #     x = block(x)
        
        # TODO: Step 4 - Apply final normalization
        # x = self.final_norm(x)
        
        # TODO: Step 5 - Project to vocabulary
        # logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # TODO: Step 6 - Compute loss if targets provided
        # loss = None
        # if targets is not None:
        #     # Flatten for cross-entropy
        #     # logits: [batch * seq_len, vocab_size]
        #     # targets: [batch * seq_len]
        #     loss = F.cross_entropy(
        #         logits.view(-1, logits.size(-1)),
        #         targets.view(-1),
        #         ignore_index=-1  # Ignore padding tokens if any
        #     )
        
        # return logits, loss
        pass
    
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
        self.eval()  # Set to eval mode
        
        for _ in range(max_new_tokens):
            # TODO: Crop input_ids if longer than max_seq_len
            # input_ids_crop = input_ids[:, -self.config.max_seq_len:]
            
            # TODO: Get logits for current sequence
            # logits, _ = self(input_ids_crop)
            
            # TODO: Get logits for last position only
            # logits = logits[:, -1, :]  # [batch, vocab_size]
            
            # TODO: Apply temperature
            # logits = logits / temperature
            
            # TODO: Optionally apply top-k filtering
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('inf')
            
            # TODO: Sample next token
            # probs = F.softmax(logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # TODO: Append to sequence
            # input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # return input_ids
        pass
    
    def count_parameters(self) -> dict:
        """Count parameters in different components."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        counts = {
            'embeddings': count_params(self.token_embeddings),
            'blocks': sum(count_params(block) for block in self.blocks),
            'final_norm': count_params(self.final_norm),
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
        vocab_size=32768,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        dropout=0.0,
        max_seq_len=1024,
        use_flash=True,
        tie_weights=True
    )
    
    print("\n1. Creating 350M model...")
    model = GPT(config)
    print("   ✓ Model created")
    
    print("\n2. Counting parameters...")
    param_counts = model.count_parameters()
    total_params = param_counts['total']
    print(f"   Total parameters: {total_params:,}")
    print(f"   Target: ~350,000,000")
    
    # Check if we're in the right ballpark (340M - 360M)
    assert 340_000_000 < total_params < 360_000_000, \
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


def run_all_tests():
    """Run all GPT tests."""
    print("\n" + "="*70)
    print(" "*20 + "GPT MODEL TESTS")
    print("="*70 + "\n")
    
    test_gpt_model()
    test_full_350m_model()
    
    print("\n" + "="*70)
    print(" "*15 + "ALL GPT MODEL TESTS PASSED! 🎉")
    print("="*70)
    print("\nYou now have a complete modern transformer implementation!")
    print("Ready to move on to training infrastructure!")


if __name__ == "__main__":
    run_all_tests()
```

---

# PART 3: Understanding the Complete Architecture

## Final Conceptual Questions

**Q11:** Trace the flow of a batch of tokens through the entire model:
- Input: [batch=2, seq_len=10] token IDs
- After embedding: shape?
- After first transformer block: shape?
- After all 24 blocks: shape?
- After final norm: shape?
- After lm_head: shape?

**Q12:** During training, we compute loss on all positions. During generation, we only use the last position.
Why? What's different about the two scenarios?

**Q13:** In the generate function, why do we crop to `max_seq_len`?
```python
input_ids_crop = input_ids[:, -self.config.max_seq_len:]
```
What happens if we don't do this?

**Q14:** Compare parameter count with and without weight tying for vocab_size=32768, d_model=1024:
- With tying: ? parameters
- Without tying: ? parameters
- Difference: ?

**Q15:** For a 350M model with d_model=1024, n_layers=24:
- Approximately how many parameters are in one transformer block?
- What percentage of total parameters are in the embedding layer?
- Why is the MLP portion of each block larger than the attention portion?

---
---
---

# ANSWERS SECTION
# (Try everything above before looking here!)

---

## Part 1 Answers: Transformer Block

**A1: Pre-LN vs Post-LN**

**Pre-LN** (modern):
```
x = x + SubLayer(Norm(x))
```
- Normalization BEFORE sub-layer
- Gradients flow directly through residual path
- More stable training
- Can train deeper without warmup

**Post-LN** (original):
```
x = Norm(x + SubLayer(x))
```
- Normalization AFTER residual addition
- Gradients must flow through normalization
- Can cause gradient issues in very deep models
- Required careful warmup

**Pre-LN has better gradient flow** because the residual connection provides a direct path for gradients that doesn't go through the normalization layer. In Post-LN, gradients must flow through the normalization after every residual addition, which can cause instability in deep networks.

**A2: Computational Graph**

```
Input: x [batch, seq_len, d_model]
    ↓
    ├─→ RMSNorm → Attention → (+) → x₁
    │                          ↑
    └──────────────────────────┘
                               ↓
    ├─→ RMSNorm → SwiGLU → (+) → x₂
    │                        ↑
    └────────────────────────┘
                             ↓
Output: x₂ [batch, seq_len, d_model]
```

Two residual paths, each with:
1. Normalization (RMSNorm)
2. Transformation (Attention or SwiGLU)
3. Residual addition

**A3: Parameter Breakdown**

For d_model = 1024:

**Attention:**
- Q, K, V, O projections: 4 × (1024 × 1024) = 4,194,304 params

**SwiGLU:**
- hidden = int(8 * 1024 / 3) ≈ 2731
- gate: 1024 × 2731 = 2,796,544
- up: 1024 × 2731 = 2,796,544
- down: 2731 × 1024 = 2,796,544
- Total: 8,389,632 params

**RMSNorm (2x):**
- 2 × 1024 = 2,048 params

**Total per block:** ~12,585,984 params

**Percentage:**
- Attention: 4,194,304 / 12,585,984 = **33.3%**
- MLP: 8,389,632 / 12,585,984 = **66.7%**

MLP has roughly 2x the parameters of attention!

**A4: Why Normalize BEFORE Sub-Layer**

**Pre-LN (normalize before):**
- Input to sub-layer has controlled distribution (normalized)
- Sub-layer receives stable inputs throughout training
- Gradients flow cleanly through residual
- Better for very deep networks

**If we normalized AFTER (Post-LN):**
- Sub-layer receives unnormalized inputs (can grow/shrink)
- Harder to keep activations stable
- Gradient path interrupted by normalization
- Need learning rate warmup to avoid early instability

**A5: Residual Connections and Gradients**

**Vanishing gradients:**
Gradient of residual connection:
```
∂x_out/∂x_in = ∂(x + F(x))/∂x = 1 + ∂F(x)/∂x
```
The "+1" means gradients can't vanish - there's always a direct path.

**Deep networks:**
With L layers, without residuals: gradient scales as (∂F)^L → 0 as L grows
With residuals: gradient always has a component of 1^L = 1

**Gradient magnitude:**
Residuals act as "gradient highways" - gradients can flow backward without being multiplied through many layers of transformations.

---

## Part 1 Code Implementation

```python
# src/model/block.py
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .swiglu import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Transformer block with Pre-LN architecture."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_flash: bool = False
    ):
        super().__init__()
        
        # Pre-attention normalization
        self.norm1 = RMSNorm(d_model)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_flash=use_flash
        )
        
        # Pre-MLP normalization
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU MLP
        self.mlp = SwiGLU(d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # First residual: attention
        x = x + self.attention(self.norm1(x))
        
        # Second residual: MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


def test_transformer_block():
    """Test transformer block implementation."""
    print("="*60)
    print("Testing Transformer Block")
    print("="*60)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    print("\n1. Testing initialization...")
    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        use_flash=False
    )
    print("   ✓ Block created successfully")
    
    print("\n2. Testing forward pass...")
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    print("   ✓ Output shape correct")
    
    print("\n3. Testing residual connections (gradient flow)...")
    x_grad = torch.randn(1, seq_len, d_model, requires_grad=True)
    output_grad = block(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print("   ✓ Gradients flow correctly through residual connections")
    
    print("\n4. Testing with different sequence lengths...")
    for test_seq_len in [5, 20, 50, 100]:
        x_test = torch.randn(1, test_seq_len, d_model)
        output_test = block(x_test)
        assert output_test.shape == (1, test_seq_len, d_model)
    print("   ✓ Works with variable sequence lengths")
    
    print("\n5. Testing parameter count...")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    expected_params = 12 * d_model**2 + 2 * d_model
    tolerance = d_model * 200
    
    assert abs(total_params - expected_params) < tolerance
    print(f"   ✓ Parameter count reasonable (expected ~{expected_params:,})")
    
    print("\n6. Testing train vs eval mode...")
    block.train()
    with torch.no_grad():
        output_train = block(x)
    
    block.eval()
    with torch.no_grad():
        output_eval = block(x)
    print("   ✓ Train/eval mode working")
    
    print("\n7. Testing with Flash Attention...")
    block_flash = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        use_flash=True
    )
    output_flash = block_flash(x)
    assert output_flash.shape == (batch_size, seq_len, d_model)
    print("   ✓ Flash Attention mode works")
    
    print("\n" + "="*60)
    print("Transformer Block Tests Passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_transformer_block()
```

---

## Part 2 Answers: Token Embeddings and GPT

**A6: No Positional Embeddings**

We don't use separate positional embeddings because **RoPE handles all positional information**.

- RoPE is applied to Q and K in the attention mechanism
- It encodes position through rotation
- No need for additive position embeddings
- Cleaner separation: embeddings = semantic, RoPE = positional

**A7: Weight Tying Benefits**

**Parameter efficiency:**
- Save vocab_size × d_model parameters
- For vocab=32768, d_model=1024: save ~33M parameters

**Performance:**
- Embedding learns: "token X → semantic vector"
- Output learns: "semantic vector → token X"
- These are inverse operations - sharing weights makes sense
- Forces consistency between input and output representations

**Theoretical:**
- Input and output spaces are the same (token space)
- Mathematically elegant: E and E^T as encoder/decoder

**A8: Embedding Size**

Vocab = 32,768, d_model = 1024:

**Token embedding matrix:** 32,768 × 1,024 = 33,554,432 params

**With weight tying:** These are the only params for embeddings
**Without weight tying:** Would need separate lm_head: another 33,554,432 params

**Savings:** 33,554,432 params (cut embeddings in half)

**Percentage of 350M model:**
33.5M / 350M = **~9.6%** of total parameters

**A9: Output Shape Transformation**

```
Input to lm_head: [batch, seq_len, d_model]
                  [2, 10, 1024]

lm_head weight: [vocab_size, d_model]
                [32768, 1024]

Matrix multiply: [batch, seq_len, d_model] @ [d_model, vocab_size]
              → [batch, seq_len, vocab_size]
              → [2, 10, 32768]
```

Each position gets a distribution over all vocab tokens.

**A10: Final RMSNorm**

**Why we need it:**
- After 24 transformer blocks, activation magnitudes can drift
- RMSNorm standardizes before final projection
- Keeps logit scales reasonable
- Stabilizes training

**Without it:**
- Logits could have very different scales
- Softmax could saturate
- Training could be unstable
- Final layer would need to adapt to varying input scales

It's cheap (just 1024 parameters) and helps a lot.

---

## Part 2 Code Implementation

```python
# src/model/gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock
from .rmsnorm import RMSNorm


class GPTConfig:
    """Configuration for GPT model."""
    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        use_flash: bool = False,
        tie_weights: bool = True
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.tie_weights = tie_weights


class GPT(nn.Module):
    """GPT decoder-only transformer with modern architecture."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_flash=config.use_flash
            )
            for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass."""
        # Get embeddings
        x = self.token_embeddings(input_ids)
        
        # Apply dropout
        if self.emb_dropout is not None:
            x = self.emb_dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
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
        """Generate new tokens."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_ids_crop = input_ids[:, -self.config.max_seq_len:]
            
            # Get logits
            logits, _ = self(input_ids_crop)
            
            # Get last position
            logits = logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """Count parameters."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        counts = {
            'embeddings': count_params(self.token_embeddings),
            'blocks': sum(count_params(block) for block in self.blocks),
            'final_norm': count_params(self.final_norm),
            'lm_head': 0 if self.config.tie_weights else count_params(self.lm_head),
            'total': sum(p.numel() for p in self.parameters())
        }
        
        return counts


# Tests included in template above
def test_gpt_model():
    """Test GPT model."""
    print("="*60)
    print("Testing GPT Model")
    print("="*60)
    
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
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None
    print("   ✓ Forward pass works")
    
    print("\n3. Testing training mode...")
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, targets)
    
    assert loss is not None
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    
    print("\n4. Testing backward pass...")
    loss.backward()
    assert model.token_embeddings.weight.grad is not None
    print("   ✓ Gradients flow correctly")
    
    print("\n5. Testing weight tying...")
    if config.tie_weights:
        assert model.token_embeddings.weight is model.lm_head.weight
        print("   ✓ Weight tying verified")
    
    print("\n6. Testing generation...")
    model.eval()
    start_tokens = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=10)
    
    assert generated.shape == (1, 15)
    print(f"   ✓ Generation works")
    
    print("\n7. Testing parameter count...")
    param_counts = model.count_parameters()
    print(f"   Total: {param_counts['total']:,}")
    
    print("\n" + "="*60)
    print("GPT Model Tests Passed! ✓")
    print("="*60)


def test_full_350m_model():
    """Test 350M model."""
    print("\n" + "="*60)
    print("Testing Full 350M Model")
    print("="*60)
    
    config = GPTConfig(
        vocab_size=32768,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        dropout=0.0,
        max_seq_len=1024,
        use_flash=True,
        tie_weights=True
    )
    
    print("\n1. Creating model...")
    model = GPT(config)
    print("   ✓ Model created")
    
    print("\n2. Counting parameters...")
    param_counts = model.count_parameters()
    total = param_counts['total']
    print(f"   Total: {total:,}")
    
    assert 340_000_000 < total < 360_000_000
    print("   ✓ In target range")
    
    print("\n3. Testing forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    logits, _ = model(input_ids)
    assert logits.shape == (1, 128, config.vocab_size)
    print("   ✓ Works")
    
    print("\n" + "="*60)
    print("350M Model Tests Passed! ✓")
    print("="*60)


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "GPT MODEL TESTS")
    print("="*70 + "\n")
    
    test_gpt_model()
    test_full_350m_model()
    
    print("\n" + "="*70)
    print(" "*15 + "ALL TESTS PASSED! 🎉")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
```

---

## Part 3 Answers: Complete Architecture

**A11: Data Flow Through Model**

```
Input token IDs: [2, 10] (batch=2, seq_len=10)
    ↓
Token Embeddings: [2, 10, 1024]
    ↓
Dropout (optional): [2, 10, 1024]
    ↓
Block 1: [2, 10, 1024]
Block 2: [2, 10, 1024]
...
Block 24: [2, 10, 1024]
    ↓
Final RMSNorm: [2, 10, 1024]
    ↓
LM Head: [2, 10, 32768]
```

Shape stays [batch, seq_len, d_model] through all blocks, then expands to vocab_size at the end.

**A12: Training vs Generation**

**Training:**
- Compute loss on ALL positions
- Teacher forcing: know correct next token
- Parallel: process entire sequence at once
- Goal: learn to predict next token at every position

**Generation:**
- Only use LAST position for prediction
- Autoregressive: feed prediction back as input
- Sequential: one token at a time
- Goal: generate new content

During training, position i tries to predict token i+1. During generation, we already have tokens 0..i and want to predict i+1.

**A13: Cropping to max_seq_len**

```python
input_ids_crop = input_ids[:, -self.config.max_seq_len:]
```

**Why crop:**
- RoPE cache is precomputed up to max_seq_len
- Attention is optimized for this length
- Memory constraints

**Without cropping:**
- Could exceed RoPE cache
- Would need to recompute frequencies
- Might run out of memory
- Attention complexity grows O(N²)

We keep the LAST max_seq_len tokens because recent context is most relevant.

**A14: Weight Tying Parameters**

For vocab_size=32768, d_model=1024:

**With tying:**
- Embedding: 32,768 × 1,024 = 33,554,432 params
- LM head: 0 (shared with embedding)
- Total: 33,554,432 params

**Without tying:**
- Embedding: 33,554,432 params
- LM head: 33,554,432 params
- Total: 67,108,864 params

**Difference: 33,554,432 params saved** (~10% of 350M model)

**A15: 350M Model Breakdown**

**One transformer block (d_model=1024):**
- Attention: 4 × 1024² ≈ 4.2M params
- SwiGLU: 8 × 1024² ≈ 8.4M params
- RMSNorm (2x): 2 × 1024 ≈ 2K params
- **Total per block: ~12.6M params**

**For 24 blocks:**
- 24 × 12.6M ≈ 302M params

**Embeddings:**
- 32,768 × 1,024 ≈ 33.6M params
- **Percentage: 33.6M / 350M ≈ 9.6%**

**Why MLP is larger:**
- MLP: 3 projections with 8/3 expansion = 8 × d²
- Attention: 4 projections = 4 × d²
- MLP has 2× the parameters of attention
- This is intentional - gives more capacity for "thinking"

---

# Summary

You now have a complete modern transformer implementation:

✅ **RMSNorm** - Efficient normalization
✅ **SwiGLU** - Gated MLP with superior performance  
✅ **RoPE** - Rotary position embeddings
✅ **Multi-Head Attention** - With manual and Flash modes
✅ **Transformer Block** - Pre-LN architecture with residuals
✅ **GPT Model** - Complete 350M parameter model

**Next steps:**
1. Tokenizer (BPE implementation)
2. Data loading (FineWeb-Edu streaming)
3. Training loop
4. Optimization and scheduling
5. Checkpointing and logging

Ready to train! 🚀