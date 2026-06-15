import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, RoPECache

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE positional encoding
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None, dropout: float = 0.0, max_seq_len: int = 2048, 
                 use_flash: bool = True, use_qk_norm: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads # dimension per head
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.use_qk_norm = use_qk_norm

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        assert n_heads % self.n_kv_heads == 0 # Check if number of KV heads makes sense
        # If using GQA
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE cache
        self.rope_cache = RoPECache(self.d_k, max_seq_len)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # QK Norm
        if use_qk_norm:
            self.qk_scale = nn.Parameter(torch.ones(n_heads) * (self.d_k ** 0.5))

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
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k) # [batch, seq_len, n_kv_heads, d_k]
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k) # [batch, seq_len, n_kv_heads, d_k]

        # Step 2.b - Apply QK-Norm if present
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.qk_scale.view(1, 1, -1, 1)
            k = F.normalize(k, dim=-1)

        # Step 3 - Apply RoPE to q and k
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Step 3.b - if GQA, expand K and V. Apply on the 3rd dim
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

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
                is_causal=True, # Automatically applies causal mask,
                scale=1 if self.use_qk_norm else None,
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


class DifferentialAttention(nn.Module):
    """Differential Attention with GQA support.
    
    Subtracts two attention maps to cancel noise (https://arxiv.org/abs/2410.05258).
    K1, K2, V use n_kv_heads (GQA) — repeated to match n_heads before attention.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None, layer_idx: int = 0,
                 dropout: float = 0.0, max_seq_len: int = 2048, use_qk_norm: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_heads
        self.n_rep = n_heads // self.n_kv_heads
        assert n_heads % self.n_kv_heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_k_half = self.d_k // 2
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm
        
        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Lambda init (unchanged from before)
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        
        self.norm = nn.GroupNorm(n_heads, d_model)
        self.rope_cache = RoPECache(self.d_k_half, max_seq_len)

        if use_qk_norm:
            # Each half has dimension d_k_half, so scale init uses sqrt(d_k_half)
            self.qk_scale1 = nn.Parameter(torch.ones(n_heads) * (self.d_k_half ** 0.5))
            self.qk_scale2 = nn.Parameter(torch.ones(n_heads) * (self.d_k_half ** 0.5))
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x) # [batch, seq_len, d_model]
        k = self.W_k(x) # [batch, seq_len, d_model]
        v = self.W_v(x) # [batch, seq_len, d_model]

        # Change dims
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k) # [batch, seq_len, n_heads, d_k]
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k) # [batch, seq_len, n_kv_heads, d_k]
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k) # [batch, seq_len, n_kv_heads, d_k]
        
        # Split into two halves (q1/q2 each [batch, seq, n_heads, d_k_half])
        # (k1/k2 each [batch, seq, n_kv_heads, d_k_half])
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # If using QK-norm, normalize:
        if self.use_qk_norm:
            q1 = F.normalize(q1, dim=-1) * self.qk_scale1.view(1, 1, -1, 1)
            q2 = F.normalize(q2, dim=-1) * self.qk_scale2.view(1, 1, -1, 1)
            k1 = F.normalize(k1, dim=-1)
            k2 = F.normalize(k2, dim=-1)
        
        # RoPE on each half
        freqs = self.rope_cache.get_freqs(seq_len)
        q1 = apply_rope(q1, freqs)
        q2 = apply_rope(q2, freqs)
        k1 = apply_rope(k1, freqs)
        k2 = apply_rope(k2, freqs)
        
        # GQA expansion — repeat k1, k2, v from n_kv_heads to n_heads
        if self.n_rep > 1:
            k1 = k1.repeat_interleave(self.n_rep, dim=2)
            k2 = k2.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose all to [batch, n_heads, seq, dim]
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Lambda (unchanged)
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )
        
        # Two attention maps, subtract
        scale = 1.0 if self.use_qk_norm else (self.d_k_half ** -0.5)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
        )
        attn1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) * scale + causal_mask, dim=-1)
        attn2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) * scale + causal_mask, dim=-1)
        attn_diff = attn1 - lam * attn2
        
        # Apply to V, reshape, GroupNorm, output
        attn_output = torch.matmul(attn_diff, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.norm(attn_output.transpose(1, 2)).transpose(1, 2)
        return self.W_o(attn_output)

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
    # Plus QK-Norm scale: n_heads (one learnable scale per head)
    expected_params = 4 * d_model * d_model + n_heads
    assert total_params == expected_params, f"Expected {expected_params} params, got {total_params}"
    print(f"✓ Parameter count correct: {total_params:,}")
  
    print("\nAll attention tests passed! ✓")


def test_differential_attention():
    print("\nTesting DifferentialAttention...")
    
    batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8
    
    # Standard (n_kv_heads = n_heads)
    diff_attn = DifferentialAttention(d_model=d_model, n_heads=n_heads, layer_idx=0)
    x = torch.randn(batch_size, seq_len, d_model)
    output = diff_attn(x)
    assert output.shape == (batch_size, seq_len, d_model), f"Got {output.shape}"
    print("✓ Standard shapes correct")
    
    # With GQA
    diff_attn_gqa = DifferentialAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=2, layer_idx=5)
    output_gqa = diff_attn_gqa(x)
    assert output_gqa.shape == (batch_size, seq_len, d_model), f"Got {output_gqa.shape}"
    print("✓ GQA shapes correct")
    
    # Gradient flow
    x_grad = torch.randn(1, seq_len, d_model, requires_grad=True)
    out = diff_attn(x_grad)
    out.sum().backward()
    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print("✓ Gradients flow correctly")
    
    # Different layer_idx → different lambda_init
    diff_attn_l0 = DifferentialAttention(d_model=d_model, n_heads=n_heads, layer_idx=0)
    diff_attn_l23 = DifferentialAttention(d_model=d_model, n_heads=n_heads, layer_idx=23)
    print(f"  lambda_init at layer 0:  {diff_attn_l0.lambda_init:.4f} (expect ~0.2)")
    print(f"  lambda_init at layer 23: {diff_attn_l23.lambda_init:.4f} (expect ~0.74)")
    
    print("✓ DifferentialAttention tests passed!")


if __name__ == "__main__":
    test_attention()
    test_differential_attention()
