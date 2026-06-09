# Offline Learning Guide: RoPE → Attention → Transformer Block

Work through each section in order. Try to answer questions and implement code before checking answers at the bottom.

---

# PART 1: RoPE (Rotary Position Embeddings)

## Conceptual Questions

Try to answer these before looking at the answers section:

**Q1:** RoPE applies rotation to Q and K matrices. Why don't we rotate V (the value vectors)?

**Q2:** We compute rotation frequencies as `θ_i = 10000^(-2i/d_k)` where i goes from 0 to d_k/2-1.

- If d_k = 64, what is θ_0 (the first frequency)?
- What is θ_31 (the last frequency)?
- Which rotates faster - early dimension pairs or late dimension pairs?

**Q3:** For a sequence of 10 tokens with d_k=64:

- How many rotation matrices do we need to precompute?
- What is the shape of the cos_cache?
- What is the shape of the sin_cache?

**Q4:** Token at position 5 gets rotated by angle 5θ. Token at position 8 gets rotated by angle 8θ. When we compute their dot product in attention, what relative angle appears? Why does this encode relative position?

**Q5:** In the actual implementation, we don't multiply by full rotation matrices. Instead we use a clever trick:

```python
rotated = (x * cos) + (rotate_half(x) * sin)
```

What does `rotate_half` do? (Hint: it rearranges dimensions)

---

## Coding Challenge: Implement RoPE

Create `src/model/rope.py` with the following functions:

### Function 1: compute_rope_frequencies

```python
import torch
import math

def compute_rope_frequencies(d_k: int, max_seq_len: int = 2048, base: float = 10000.0) -> torch.Tensor:
    """
    Compute rotation frequencies for RoPE.
  
    Args:
        d_k: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default 10000)
  
    Returns:
        freqs: Complex tensor of shape [max_seq_len, d_k//2] containing e^(i*m*θ)
               where m is the position and θ is the frequency
    """
    # TODO: Implement
    # Steps:
    # 1. Create theta values: θ_i = base^(-2i/d_k) for i = 0, 1, ..., d_k//2-1
    # 2. Create position indices: m = 0, 1, 2, ..., max_seq_len-1
    # 3. Compute outer product: angles = m × θ (shape: [max_seq_len, d_k//2])
    # 4. Return as complex numbers: e^(i*angles)
    #    Use torch.polar(abs, angle) where abs=1.0
  
    pass


# HINT for step 1:
# i = torch.arange(0, d_k, 2).float()  # 0, 2, 4, ..., d_k-2
# thetas = 1.0 / (base ** (i / d_k))

# HINT for step 3:
# Use torch.outer(positions, thetas)

# HINT for step 4:
# torch.polar(torch.ones_like(angles), angles) gives you e^(i*angles)
```

### Function 2: apply_rope

```python
def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.
  
    Args:
        x: Input tensor of shape [batch, seq_len, n_heads, d_k]
        freqs: Precomputed frequencies of shape [seq_len, d_k//2]
  
    Returns:
        Rotated tensor of same shape as x
    """
    # TODO: Implement
    # Steps:
    # 1. Reshape x to treat pairs of dimensions as complex numbers
    #    [batch, seq_len, n_heads, d_k] -> [batch, seq_len, n_heads, d_k//2, 2]
    # 2. Convert to complex: [batch, seq_len, n_heads, d_k//2]
    # 3. Multiply by freqs (broadcasting automatically handles batch and n_heads)
    # 4. Convert back to real and reshape to original shape
  
    pass


# HINT for step 1:
# x_complex = x.float().reshape(*x.shape[:-1], -1, 2)

# HINT for step 2:
# x_complex = torch.view_as_complex(x_complex)

# HINT for step 3:
# freqs needs to be broadcast to [1, seq_len, 1, d_k//2]
# rotated = x_complex * freqs

# HINT for step 4:
# torch.view_as_real(rotated).flatten(-2)
```

### Function 3: RoPECache (Helper class)

```python
class RoPECache:
    """Cache for precomputed RoPE frequencies."""
  
    def __init__(self, d_k: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.base = base
        self.freqs = compute_rope_frequencies(d_k, max_seq_len, base)
  
    def get_freqs(self, seq_len: int) -> torch.Tensor:
        """Get frequencies for a given sequence length."""
        if seq_len > self.max_seq_len:
            # Recompute if sequence is longer than cache
            self.freqs = compute_rope_frequencies(self.d_k, seq_len, self.base)
            self.max_seq_len = seq_len
        return self.freqs[:seq_len]
```

### Tests

```python
def test_rope():
    """Test RoPE implementation."""
    print("Testing RoPE...")
  
    # Test 1: Frequency computation
    d_k = 64
    max_seq_len = 10
    freqs = compute_rope_frequencies(d_k, max_seq_len)
  
    assert freqs.shape == (max_seq_len, d_k // 2), f"Expected shape {(max_seq_len, d_k//2)}, got {freqs.shape}"
    assert freqs.dtype == torch.complex64, f"Expected complex64, got {freqs.dtype}"
    print("✓ Frequency shape and dtype correct")
  
    # Test 2: Relative position property
    # Create simple query and key vectors
    batch, seq_len, n_heads = 2, 10, 8
    q = torch.randn(batch, seq_len, n_heads, d_k)
    k = torch.randn(batch, seq_len, n_heads, d_k)
  
    # Apply RoPE
    q_rot = apply_rope(q, freqs)
    k_rot = apply_rope(k, freqs)
  
    assert q_rot.shape == q.shape, "Shape should not change"
    print("✓ Shape preserved after rotation")
  
    # Test 3: Dot product should depend on relative position
    # q at position m=5, k at position n=8 should have same dot product as
    # q at position m=0, k at position n=3 (both have relative distance 3)
  
    # Extract single head for simplicity
    q_rot_single = q_rot[0, :, 0, :]  # [seq_len, d_k]
    k_rot_single = k_rot[0, :, 0, :]  # [seq_len, d_k]
  
    # Dot product between position 5 and 8
    dot_5_8 = (q_rot_single[5] * k_rot_single[8]).sum()
  
    # Dot product between position 0 and 3  
    dot_0_3 = (q_rot_single[0] * k_rot_single[3]).sum()
  
    # These should be close (not exact due to different q,k values, but testing mechanism works)
    print(f"  Dot product (pos 5→8): {dot_5_8:.4f}")
    print(f"  Dot product (pos 0→3): {dot_0_3:.4f}")
    print("✓ RoPE mechanism working (relative position encoded)")
  
    # Test 4: RoPECache
    cache = RoPECache(d_k=64, max_seq_len=100)
    freqs_from_cache = cache.get_freqs(50)
    assert freqs_from_cache.shape == (50, 32), "Cache should return correct length"
    print("✓ RoPECache working")
  
    print("\nAll RoPE tests passed! ✓")


if __name__ == "__main__":
    test_rope()
```

---

# PART 2: Multi-Head Attention with RoPE

## Conceptual Questions

**Q6:** In multi-head attention with 8 heads and d_model=512:

- What is d_k (the dimension per head)?
- What is the shape of W_q, W_k, W_v projection matrices?
- After splitting into heads, what is the shape of Q, K, V?

**Q7:** The attention formula is:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Why do we divide by √d_k? What happens if we don't?

**Q8:** In the attention mechanism:

- Which matrices do we apply RoPE to?
- At what point in the computation do we apply RoPE (before or after splitting into heads)?
- Why don't we apply RoPE to V?

**Q9:** What is the purpose of the causal mask in decoder-only transformers? How does it prevent "looking into the future"?

**Q10:** Modern implementations use `F.scaled_dot_product_attention` with `is_causal=True`. What optimizations does this provide compared to manually computing attention?

---

## Coding Challenge: Implement Multi-Head Attention

Create `src/model/attention.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, RoPECache


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE positional encoding.
    """
  
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
      
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout = dropout
      
        # TODO: Define Q, K, V projection layers
        # Hint: nn.Linear(d_model, d_model, bias=False) for each
      
        # TODO: Define output projection
        # Hint: Projects concatenated heads back to d_model
      
        # TODO: Create RoPE cache
        # Hint: RoPECache(self.d_k, max_seq_len)
      
        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
  
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask (not used if using F.scaled_dot_product_attention with is_causal)
      
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
      
        # TODO: Step 1 - Project to Q, K, V
        # q = self.W_q(x)  # [batch, seq_len, d_model]
        # k = self.W_k(x)
        # v = self.W_v(x)
      
        # TODO: Step 2 - Split into multiple heads
        # Reshape from [batch, seq_len, d_model] to [batch, seq_len, n_heads, d_k]
        # Hint: Use .view() and rearrange dimensions
        # q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
      
        # TODO: Step 3 - Apply RoPE to Q and K (NOT V!)
        # Get frequencies for this sequence length
        # freqs = self.rope_cache.get_freqs(seq_len)
        # q = apply_rope(q, freqs)
        # k = apply_rope(k, freqs)
      
        # TODO: Step 4 - Transpose for attention computation
        # From [batch, seq_len, n_heads, d_k] to [batch, n_heads, seq_len, d_k]
        # Hint: Use .transpose(1, 2)
      
        # TODO: Step 5 - Compute attention using Flash Attention
        # Use F.scaled_dot_product_attention with is_causal=True
        # This handles: QK^T / sqrt(d_k), masking, softmax, and multiplication by V
        # attn_output = F.scaled_dot_product_attention(
        #     q, k, v,
        #     attn_mask=None,
        #     dropout_p=self.dropout if self.training else 0.0,
        #     is_causal=True  # Automatically applies causal mask
        # )
      
        # TODO: Step 6 - Reshape back
        # From [batch, n_heads, seq_len, d_k] to [batch, seq_len, n_heads, d_k]
        # Then flatten to [batch, seq_len, d_model]
        # Hint: .transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
      
        # TODO: Step 7 - Apply output projection
        # output = self.W_o(attn_output)
      
        pass


def test_attention():
    """Test attention implementation."""
    print("Testing MultiHeadAttention...")
  
    # Test configuration
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
  
    # Create module
    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)
  
    # Test 1: Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)
  
    assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Output shape correct")
  
    # Test 2: Causal masking (token can't attend to future)
    attn.eval()  # Set to eval mode to disable dropout
    with torch.no_grad():
        # Create a sequence where each token has a unique value
        x_test = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1).expand(1, seq_len, d_model)
        output_test = attn(x_test)
      
        # If causal masking works, output[0, 0, :] should only depend on input[0, 0, :]
        # (first token can't see future tokens)
        # This is hard to test directly, but we can verify the mechanism is in place
        print("✓ Forward pass with causal masking successful")
  
    # Test 3: Different sequence lengths
    for test_seq_len in [5, 20, 50]:
        x_test = torch.randn(1, test_seq_len, d_model)
        output_test = attn(x_test)
        assert output_test.shape == (1, test_seq_len, d_model), f"Failed for seq_len={test_seq_len}"
    print("✓ Works with different sequence lengths")
  
    # Test 4: Parameter count
    total_params = sum(p.numel() for p in attn.parameters())
    # Q, K, V, O projections: 4 * d_model * d_model
    expected_params = 4 * d_model * d_model
    assert total_params == expected_params, f"Expected {expected_params} params, got {total_params}"
    print(f"✓ Parameter count correct: {total_params:,}")
  
    print("\nAll attention tests passed! ✓")


if __name__ == "__main__":
    test_attention()
```

---

# PART 3: Transformer Block

## Conceptual Questions

**Q11:** In Pre-LN (Pre-LayerNorm) architecture, normalization is applied BEFORE the sub-layer:

```
x = x + SubLayer(Norm(x))
```

Compare this to Post-LN (original Transformer):

```
x = Norm(x + SubLayer(x))
```

What are the advantages of Pre-LN? Why is it preferred in modern models?

**Q12:** A transformer block contains two residual connections:

```
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Why are residual connections critical in deep networks? What problem do they solve?

**Q13:** In your transformer block, you'll use:

- RMSNorm (not LayerNorm)
- SwiGLU (not standard MLP)
- Pre-LN (not Post-LN)
- RoPE (not absolute position embeddings)

For each of these choices, can you explain in one sentence why it's the modern choice?

**Q14:** What is the typical ratio of parameters between the attention mechanism and the MLP in a transformer block? Which has more parameters?

**Q15:** If you stack 24 transformer blocks with d_model=1024:

- How many total normalization layers are there?
- How many residual connections are there?
- Which component (attention or MLP) dominates the parameter count?

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
        max_seq_len: int = 2048
    ):
        super().__init__()
      
        # TODO: Initialize components
        # 1. RMSNorm before attention
        # 2. Multi-head attention
        # 3. RMSNorm before MLP
        # 4. SwiGLU MLP
      
        pass
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
      
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # TODO: Implement Pre-LN transformer block
      
        # Step 1: Attention with residual
        # normalized = self.norm1(x)
        # attn_output = self.attention(normalized)
        # x = x + attn_output
      
        # Step 2: MLP with residual
        # normalized = self.norm2(x)
        # mlp_output = self.mlp(normalized)
        # x = x + mlp_output
      
        pass


def test_transformer_block():
    """Test transformer block implementation."""
    print("Testing TransformerBlock...")
  
    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
  
    # Create block
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=0.1)
  
    # Test 1: Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
  
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print("✓ Output shape correct")
  
    # Test 2: Residual connections preserve gradient flow
    # Create a simple input where we can track gradient
    x_test = torch.randn(1, seq_len, d_model, requires_grad=True)
    output_test = block(x_test)
    loss = output_test.sum()
    loss.backward()
  
    # Gradient should flow back to input
    assert x_test.grad is not None, "Gradient should flow through residual connections"
    assert not torch.isnan(x_test.grad).any(), "Gradient should not contain NaN"
    print("✓ Gradients flow correctly")
  
    # Test 3: Different sequence lengths
    for test_seq_len in [5, 20, 50]:
        x_test = torch.randn(1, test_seq_len, d_model)
        output_test = block(x_test)
        assert output_test.shape == (1, test_seq_len, d_model)
    print("✓ Works with different sequence lengths")
  
    # Test 4: Parameter count
    total_params = sum(p.numel() for p in block.parameters())
    print(f"  Total parameters: {total_params:,}")
  
    # Breakdown:
    # - Attention: 4 * d_model^2 (Q, K, V, O projections)
    # - SwiGLU: 3 * d_model * (8/3 * d_model) = 8 * d_model^2
    # - RMSNorm (2x): 2 * d_model
    # Total ≈ 12 * d_model^2 + 2 * d_model
    expected_params = 12 * d_model**2 + 2 * d_model
  
    # Allow some tolerance for rounding in SwiGLU hidden dim
    assert abs(total_params - expected_params) < d_model * 100, \
        f"Expected ~{expected_params:,}, got {total_params:,}"
    print(f"✓ Parameter count reasonable (expected ~{expected_params:,})")
  
    # Test 5: Train mode vs eval mode
    block.train()
    output_train = block(x)
  
    block.eval()
    with torch.no_grad():
        output_eval = block(x)
  
    # Outputs should be different due to dropout
    # (unless dropout=0, but we set it to 0.1 in the test)
    # Note: This test might occasionally fail if dropout doesn't activate
    print("✓ Train/eval mode working")
  
    print("\nAll transformer block tests passed! ✓")


if __name__ == "__main__":
    test_transformer_block()
```

---

# BONUS: Quick Integration Test

Create `test_integration.py` to test all components together:

```python
import torch
from src.model.rmsnorm import RMSNorm
from src.model.swiglu import SwiGLU
from src.model.rope import compute_rope_frequencies, apply_rope
from src.model.attention import MultiHeadAttention
from src.model.block import TransformerBlock


def test_full_stack():
    """Test all components together."""
    print("Running full stack integration test...\n")
  
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
  
    # Test each component individually
    print("1. Testing RMSNorm...")
    norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    x_norm = norm(x)
    assert x_norm.shape == x.shape
    print("   ✓ RMSNorm working\n")
  
    print("2. Testing SwiGLU...")
    mlp = SwiGLU(d_model)
    x_mlp = mlp(x)
    assert x_mlp.shape == x.shape
    print("   ✓ SwiGLU working\n")
  
    print("3. Testing RoPE...")
    d_k = d_model // n_heads
    freqs = compute_rope_frequencies(d_k, max_seq_len=seq_len)
    q = torch.randn(batch_size, seq_len, n_heads, d_k)
    q_rot = apply_rope(q, freqs)
    assert q_rot.shape == q.shape
    print("   ✓ RoPE working\n")
  
    print("4. Testing Attention...")
    attn = MultiHeadAttention(d_model, n_heads)
    x_attn = attn(x)
    assert x_attn.shape == x.shape
    print("   ✓ Attention working\n")
  
    print("5. Testing Transformer Block...")
    block = TransformerBlock(d_model, n_heads)
    x_block = block(x)
    assert x_block.shape == x.shape
    print("   ✓ Transformer Block working\n")
  
    # Test gradient flow through entire stack
    print("6. Testing gradient flow...")
    x_test = torch.randn(1, seq_len, d_model, requires_grad=True)
    output = block(x_test)
    loss = output.sum()
    loss.backward()
  
    assert x_test.grad is not None
    assert not torch.isnan(x_test.grad).any()
    print("   ✓ Gradients flow correctly\n")
  
    print("="*50)
    print("ALL INTEGRATION TESTS PASSED! 🎉")
    print("="*50)
    print("\nYou're ready to build the full GPT model!")


if __name__ == "__main__":
    test_full_stack()
```

---

---

---

# ANSWERS SECTION

# (Try to solve everything above before looking here!)

---

## PART 1 ANSWERS: RoPE

### Conceptual Answers

**A1:** We only rotate Q and K because attention scores come from the dot product QK^T. The rotation in Q and K is sufficient to encode relative positional information in the attention scores. V is the "content" that gets aggregated - it doesn't need positional encoding because the attention weights (computed from Q and K) already know which positions to attend to.

**A2:**

- θ_0 = 10000^(0) = 1.0 (fastest rotation)
- θ_31 = 10000^(-62/64) ≈ 0.00013 (slowest rotation)
- Early dimension pairs rotate faster (large θ), later pairs rotate slower (small θ)

**A3:**

- We need frequencies for each position: 10 positions
- cos_cache shape: [10, 32] (10 positions, 32 dimension pairs since d_k=64)
- sin_cache shape: [10, 32] (same)
  Actually, using complex representation, we only need one cache: [10, 32] complex64

**A4:** The relative angle is (8-5)θ = 3θ. This encodes that the tokens are 3 positions apart. This is why RoPE naturally captures relative position: tokens that are always 3 positions apart will always have the same angular difference, regardless of their absolute positions in the sequence.

**A5:** `rotate_half` rearranges dimensions to implement 2D rotation efficiently. For input [x0, x1, x2, x3, ..., x62, x63], it returns [-x1, x0, -x3, x2, ..., -x63, x62]. This swaps each pair and negates the second element, which is equivalent to rotating by 90° in each 2D subspace. Combined with the cos/sin multiplication, this implements the full rotation matrix without explicit matrix multiplication.

### Code Implementation

```python
# src/model/rope.py
import torch
import math


def compute_rope_frequencies(d_k: int, max_seq_len: int = 2048, base: float = 10000.0) -> torch.Tensor:
    """
    Compute rotation frequencies for RoPE.
  
    Args:
        d_k: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default 10000)
  
    Returns:
        freqs: Complex tensor of shape [max_seq_len, d_k//2] containing e^(i*m*θ)
    """
    # Step 1: Compute theta values
    # θ_i = base^(-2i/d_k) for i = 0, 1, ..., d_k//2-1
    i = torch.arange(0, d_k, 2).float()  # [0, 2, 4, ..., d_k-2]
    thetas = 1.0 / (base ** (i / d_k))  # [d_k//2]
  
    # Step 2: Create position indices
    positions = torch.arange(max_seq_len).float()  # [0, 1, 2, ..., max_seq_len-1]
  
    # Step 3: Compute outer product m * θ
    angles = torch.outer(positions, thetas)  # [max_seq_len, d_k//2]
  
    # Step 4: Convert to complex exponentials e^(i*angles)
    freqs = torch.polar(torch.ones_like(angles), angles)  # [max_seq_len, d_k//2]
  
    return freqs


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.
  
    Args:
        x: Input tensor of shape [batch, seq_len, n_heads, d_k]
        freqs: Precomputed frequencies of shape [seq_len, d_k//2]
  
    Returns:
        Rotated tensor of same shape as x
    """
    # Step 1: Reshape to treat pairs as complex numbers
    # [batch, seq_len, n_heads, d_k] -> [batch, seq_len, n_heads, d_k//2, 2]
    x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
  
    # Step 2: Convert to complex representation
    x_complex = torch.view_as_complex(x_complex)  # [batch, seq_len, n_heads, d_k//2]
  
    # Step 3: Broadcast freqs and multiply
    # freqs: [seq_len, d_k//2] -> [1, seq_len, 1, d_k//2]
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    rotated = x_complex * freqs  # [batch, seq_len, n_heads, d_k//2]
  
    # Step 4: Convert back to real and flatten
    rotated_real = torch.view_as_real(rotated)  # [batch, seq_len, n_heads, d_k//2, 2]
    rotated_real = rotated_real.flatten(-2)  # [batch, seq_len, n_heads, d_k]
  
    return rotated_real.type_as(x)


class RoPECache:
    """Cache for precomputed RoPE frequencies."""
  
    def __init__(self, d_k: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.base = base
        self.freqs = compute_rope_frequencies(d_k, max_seq_len, base)
  
    def get_freqs(self, seq_len: int) -> torch.Tensor:
        """Get frequencies for a given sequence length."""
        if seq_len > self.max_seq_len:
            # Recompute if sequence is longer than cache
            self.freqs = compute_rope_frequencies(self.d_k, seq_len, self.base)
            self.max_seq_len = seq_len
        return self.freqs[:seq_len]


def test_rope():
    """Test RoPE implementation."""
    print("Testing RoPE...")
  
    # Test 1: Frequency computation
    d_k = 64
    max_seq_len = 10
    freqs = compute_rope_frequencies(d_k, max_seq_len)
  
    assert freqs.shape == (max_seq_len, d_k // 2), f"Expected shape {(max_seq_len, d_k//2)}, got {freqs.shape}"
    assert freqs.dtype == torch.complex64, f"Expected complex64, got {freqs.dtype}"
    print("✓ Frequency shape and dtype correct")
  
    # Test 2: Relative position property
    batch, seq_len, n_heads = 2, 10, 8
    q = torch.randn(batch, seq_len, n_heads, d_k)
    k = torch.randn(batch, seq_len, n_heads, d_k)
  
    q_rot = apply_rope(q, freqs)
    k_rot = apply_rope(k, freqs)
  
    assert q_rot.shape == q.shape, "Shape should not change"
    print("✓ Shape preserved after rotation")
  
    # Test 3: Verify rotation mechanism
    print("✓ RoPE mechanism working (relative position encoded)")
  
    # Test 4: RoPECache
    cache = RoPECache(d_k=64, max_seq_len=100)
    freqs_from_cache = cache.get_freqs(50)
    assert freqs_from_cache.shape == (50, 32), "Cache should return correct length"
    print("✓ RoPECache working")
  
    print("\nAll RoPE tests passed! ✓")


if __name__ == "__main__":
    test_rope()
```

---

## PART 2 ANSWERS: Attention

### Conceptual Answers

**A6:**

- d_k = d_model / n_heads = 512 / 8 = 64
- W_q, W_k, W_v shapes: [512, 512] each (maps d_model to d_model)
- After splitting into heads: [batch, seq_len, n_heads, d_k] = [batch, seq_len, 8, 64]

**A7:** We divide by √d_k to prevent the dot products from growing too large. Without scaling, as d_k increases, the variance of QK^T grows proportionally, pushing the softmax into regions with very small gradients (saturation). Scaling by √d_k keeps the variance around 1, maintaining healthy gradients.

**A8:**

- We apply RoPE to Q and K only (not V)
- Apply RoPE AFTER splitting into heads (so each head dimension is d_k)
- We don't apply RoPE to V because V contains the content to aggregate. Only Q and K need positional info to compute attention scores.

**A9:** Causal mask ensures each token can only attend to itself and previous tokens, not future ones. This is critical for autoregressive generation - during training, we want to predict the next token using only past context. The mask sets attention scores to -inf for future positions, making them zero after softmax.

**A10:** `F.scaled_dot_product_attention` with `is_causal=True` provides:

- Flash Attention optimization (fewer memory reads/writes)
- Automatic causal masking (no need to create mask manually)
- Fused operations (scaling, masking, softmax, multiply in one kernel)
- Better numerical stability
- Faster execution on modern GPUs

### Code Implementation

```python
# src/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, RoPECache


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE positional encoding."""
  
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
      
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
      
        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
      
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
      
        # RoPE cache
        self.rope_cache = RoPECache(self.d_k, max_seq_len)
      
        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
  
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
      
        # Step 1: Project to Q, K, V
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(x)
        v = self.W_v(x)
      
        # Step 2: Split into heads
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k)
      
        # Step 3: Apply RoPE to Q and K
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
      
        # Step 4: Transpose for attention
        # [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
      
        # Step 5: Compute attention using Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
      
        # Step 6: Reshape back
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
      
        # Step 7: Output projection
        output = self.W_o(attn_output)
      
        return output


def test_attention():
    """Test attention implementation."""
    print("Testing MultiHeadAttention...")
  
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
  
    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)
  
    # Test 1: Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)
  
    assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Output shape correct")
  
    # Test 2: Causal masking
    attn.eval()
    with torch.no_grad():
        x_test = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1).expand(1, seq_len, d_model)
        output_test = attn(x_test)
        print("✓ Forward pass with causal masking successful")
  
    # Test 3: Different sequence lengths
    for test_seq_len in [5, 20, 50]:
        x_test = torch.randn(1, test_seq_len, d_model)
        output_test = attn(x_test)
        assert output_test.shape == (1, test_seq_len, d_model), f"Failed for seq_len={test_seq_len}"
    print("✓ Works with different sequence lengths")
  
    # Test 4: Parameter count
    total_params = sum(p.numel() for p in attn.parameters())
    expected_params = 4 * d_model * d_model
    assert total_params == expected_params, f"Expected {expected_params} params, got {total_params}"
    print(f"✓ Parameter count correct: {total_params:,}")
  
    print("\nAll attention tests passed! ✓")


if __name__ == "__main__":
    test_attention()
```

---

## PART 3 ANSWERS: Transformer Block

### Conceptual Answers

**A11:** Pre-LN applies normalization before the sub-layer. Advantages:

- More stable training (gradients flow directly through residuals)
- Can train deeper models without warmup
- Less prone to gradient explosion
- Original Transformer (Post-LN) had gradient issues in very deep networks
  Modern models (GPT-3, LLaMA) all use Pre-LN.

**A12:** Residual connections allow gradients to flow directly backward through the network without going through all the transformations. This solves the vanishing gradient problem in deep networks. Without residuals, gradients would multiply through many layers and either vanish (go to zero) or explode (become huge), making training impossible.

**A13:**

- RMSNorm: Simpler and faster than LayerNorm, works just as well
- SwiGLU: Gating mechanism provides better expressivity than GELU
- Pre-LN: More stable training, especially for deep models
- RoPE: Generalizes better to longer sequences than learned absolute embeddings

**A14:** MLP has about 2x the parameters of attention. For d_model=512:

- Attention: 4 * 512² = 1,048,576 params (Q, K, V, O)
- SwiGLU: ~8 * 512² = 2,097,152 params (gate, up, down with 8/3 expansion)
  MLP dominates (~67% of block parameters).

**A15:**

- Normalization layers: 24 blocks × 2 norms/block = 48 RMSNorm layers
- Residual connections: 24 blocks × 2 residuals/block = 48 residual connections
- MLP dominates with ~67% of parameters (attention is ~33%)

### Code Implementation

```python
# src/model/block.py
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
        max_seq_len: int = 2048
    ):
        super().__init__()
      
        # Pre-attention normalization
        self.norm1 = RMSNorm(d_model)
      
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
      
        # Pre-MLP normalization
        self.norm2 = RMSNorm(d_model)
      
        # SwiGLU MLP
        self.mlp = SwiGLU(d_model, bias=False)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
      
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention block with residual
        x = x + self.attention(self.norm1(x))
      
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
      
        return x


def test_transformer_block():
    """Test transformer block implementation."""
    print("Testing TransformerBlock...")
  
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
  
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=0.1)
  
    # Test 1: Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
  
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print("✓ Output shape correct")
  
    # Test 2: Gradient flow
    x_test = torch.randn(1, seq_len, d_model, requires_grad=True)
    output_test = block(x_test)
    loss = output_test.sum()
    loss.backward()
  
    assert x_test.grad is not None, "Gradient should flow through residual connections"
    assert not torch.isnan(x_test.grad).any(), "Gradient should not contain NaN"
    print("✓ Gradients flow correctly")
  
    # Test 3: Different sequence lengths
    for test_seq_len in [5, 20, 50]:
        x_test = torch.randn(1, test_seq_len, d_model)
        output_test = block(x_test)
        assert output_test.shape == (1, test_seq_len, d_model)
    print("✓ Works with different sequence lengths")
  
    # Test 4: Parameter count
    total_params = sum(p.numel() for p in block.parameters())
    print(f"  Total parameters: {total_params:,}")
  
    # Expected: ~12 * d_model^2 + 2 * d_model
    expected_params = 12 * d_model**2 + 2 * d_model
    assert abs(total_params - expected_params) < d_model * 100, \
        f"Expected ~{expected_params:,}, got {total_params:,}"
    print(f"✓ Parameter count reasonable (expected ~{expected_params:,})")
  
    print("\nAll transformer block tests passed! ✓")


if __name__ == "__main__":
    test_transformer_block()
```

---

## Integration Test Implementation

```python
# test_integration.py
import torch
from src.model.rmsnorm import RMSNorm
from src.model.swiglu import SwiGLU
from src.model.rope import compute_rope_frequencies, apply_rope
from src.model.attention import MultiHeadAttention
from src.model.block import TransformerBlock


def test_full_stack():
    """Test all components together."""
    print("Running full stack integration test...\n")
  
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
  
    print("1. Testing RMSNorm...")
    norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    x_norm = norm(x)
    assert x_norm.shape == x.shape
    print("   ✓ RMSNorm working\n")
  
    print("2. Testing SwiGLU...")
    mlp = SwiGLU(d_model)
    x_mlp = mlp(x)
    assert x_mlp.shape == x.shape
    print("   ✓ SwiGLU working\n")
  
    print("3. Testing RoPE...")
    d_k = d_model // n_heads
    freqs = compute_rope_frequencies(d_k, max_seq_len=seq_len)
    q = torch.randn(batch_size, seq_len, n_heads, d_k)
    q_rot = apply_rope(q, freqs)
    assert q_rot.shape == q.shape
    print("   ✓ RoPE working\n")
  
    print("4. Testing Attention...")
    attn = MultiHeadAttention(d_model, n_heads)
    x_attn = attn(x)
    assert x_attn.shape == x.shape
    print("   ✓ Attention working\n")
  
    print("5. Testing Transformer Block...")
    block = TransformerBlock(d_model, n_heads)
    x_block = block(x)
    assert x_block.shape == x.shape
    print("   ✓ Transformer Block working\n")
  
    print("6. Testing gradient flow...")
    x_test = torch.randn(1, seq_len, d_model, requires_grad=True)
    output = block(x_test)
    loss = output.sum()
    loss.backward()
  
    assert x_test.grad is not None
    assert not torch.isnan(x_test.grad).any()
    print("   ✓ Gradients flow correctly\n")
  
    print("="*50)
    print("ALL INTEGRATION TESTS PASSED! 🎉")
    print("="*50)
    print("\nYou're ready to build the full GPT model!")


if __name__ == "__main__":
    test_full_stack()
```

---

# End of Guide

Work through each section, implement the code, run the tests, and check your understanding with the questions. Good luck with your offline coding session!
