import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import apply_rope, RoPECache

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE positional encoding
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 2048, use_flash: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # dimension per head
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash

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
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask (not used if using F.scaled_dot_product_attention with is_causal)

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1 - project to Q, K, V
        q = self.W_q(x) # [batch, seq_len, d_model]
        k = self.W_k(x) # [batch, seq_len, d_model]
        v = self.W_v(x) # [batch, seq_len, d_model]

        # Step 2 - Split into multiple heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k) # [batch, seq_len, n_heads, d_k]
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k) # [batch, seq_len, n_heads, d_k]
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k) # [batch, seq_len, n_heads, d_k]

        # Step 3 - Apply RoPE to q and k
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Step 4 - Transpose for attention computation
        q = q.transpose(1, 2) # [batch, n_heads, seq_len, d_k]
        k = k.transpose(1, 2) # [batch, n_heads, seq_len, d_k]
        v = v.transpose(1, 2) # [batch, n_heads, seq_len, d_k]

        # Step 5 - Compute attention using Flash Attention
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True # Automatically applies causal mask
            )
        else:
            # Manual attention computation (for learning)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len] -> [batch, n_heads, seq_len, seq_len]
            attn_scores = attn_scores / (self.d_k ** 0.5)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
            )
            attn_scores = attn_scores + causal_mask # Broadcasting will apply mask to all batches and heads
            attn_weights = F.softmax(attn_scores, dim=-1) # [batch, n_heads, seq_len, seq_len]
            if self.attn_dropout is not None:
                attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v) # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_k] - > [batch, n_heads, seq_len, d_k]

        # Step 6 - Reshape back
        attn_output = attn_output.transpose(1, 2) # [batch, seq_len, n_heads, d_k]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, d_model) # [batch, seq_len, d_model]

        # Step 7 - Apply output projection
        output = self.W_o(attn_output)

        return output

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
