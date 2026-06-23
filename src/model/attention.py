import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope, RoPECache
from .rmsnorm import RMSNorm

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE positional encoding
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None, dropout: float = 0.0, max_seq_len: int = 2048, 
                 use_flash: bool = True, use_qk_norm: bool = True, use_xsa: bool = False):
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
        self.use_xsa = use_xsa

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
                scale=1.0 if self.use_qk_norm else None,
            )
        else:
            # Manual attention computation (for learning)
            scale = 1.0 if self.use_qk_norm else (self.d_k ** -0.5)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len] -> [batch, n_heads, seq_len, seq_len]

            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
            )
            attn_scores = attn_scores + causal_mask # Broadcasting will apply mask to all batches and heads
            attn_weights = F.softmax(attn_scores, dim=-1) # [batch, n_heads, seq_len, seq_len]
            if self.attn_dropout is not None:
                attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v) # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_k] - > [batch, n_heads, seq_len, d_k]

        # Step 5.b — Exclusive Self-Attention (XSA): remove the component of the
        # attention output aligned with each token's own value vector.
        # attn_output and v are both [batch, n_heads, seq_len, d_k] here.
        # v was already GQA-expanded and transposed above, so shapes align.
        if self.use_xsa:
            # projection of attn_output onto v, per (batch, head, position)
            dot = (attn_output * v).sum(dim=-1, keepdim=True) # [batch, n_heads, seq_len, 1]
            denom = v.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6) # [bath, n_heads, seq_len, 1]
            projection = (dot / denom) * v # [batch, n_heads, seq_len, d_k]
            attn_output = attn_output - projection

        # Step 6 - Reshape back
        attn_output = attn_output.transpose(1, 2) # [batch, seq_len, n_heads, d_k]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, d_model) # [batch, seq_len, d_model]

        # Step 7 - Apply output projection
        output = self.W_o(attn_output)

        return output


class DifferentialAttention(nn.Module):
    """Differential Attention with GQA support.
    
    Computes two attention outputs and subtracts them, equivalent to applying
    (A1 - λA2) @ V without materializing the full attention maps. (https://arxiv.org/abs/2410.05258).
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
        assert self.d_k % 4 == 0, "DifferentialAttention + RoPE requires head_dim divisible by 4"
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
        self.register_buffer("lambda_init", torch.tensor(lambda_init, dtype=torch.float32))
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        
        self.head_norm = RMSNorm(self.d_k)
        self.rope_cache = RoPECache(self.d_k_half, max_seq_len)

        if use_qk_norm:
            # Each half has dimension d_k_half, so scale init uses sqrt(d_k_half)
            self.qk_scale1 = nn.Parameter(torch.ones(n_heads) * (self.d_k_half ** 0.5))
            self.qk_scale2 = nn.Parameter(torch.ones(n_heads) * (self.d_k_half ** 0.5))

    def compute_lambda(self) -> torch.Tensor:
        """
        Compute the current differential attention lambda.

        Returns:
            Scalar tensor.
        """
        return (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )
    
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
        
        # Lambda
        lam = self.compute_lambda().to(dtype=v.dtype)
        
        # Two attention outputs using Flash Attention for speedup
        attn1_output = F.scaled_dot_product_attention(
            q1, k1, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True, # Automatically applies causal mask,
            scale=1.0 if self.use_qk_norm else (self.d_k_half ** -0.5),
        )

        attn2_output = F.scaled_dot_product_attention(
            q2, k2, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True, # Automatically applies causal mask,
            scale=1.0 if self.use_qk_norm else (self.d_k_half ** -0.5),
        )
        # Apply to V, reshape, RMSNorm, output
        attn_output = attn1_output - lam * attn2_output # Subtract the two attention outputs, equivalent to (A1 - λA2) @ V
        attn_output = attn_output.transpose(1, 2) # [batch, seq, n_heads, d_k]
        attn_output = self.head_norm(attn_output) # RMSNorm over last dim d_k, per token per head
        scale_factor = (1 - self.lambda_init).to(dtype=attn_output.dtype)
        attn_output = attn_output * scale_factor
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)


# TESTS SECTION

def assert_no_future_leak(module, d_model: int, seq_len: int = 12, atol: float = 1e-5):
    """
    Check causal masking by changing a future token and verifying earlier outputs do not change.

    For a causal attention module, changing token t should not affect outputs at positions < t.
    """
    module.eval()

    with torch.no_grad():
        x1 = torch.randn(1, seq_len, d_model)
        x2 = x1.clone()

        # Change the last token massively.
        # Earlier positions should not be affected.
        x2[:, -1, :] = torch.randn_like(x2[:, -1, :]) * 1000.0

        out1 = module(x1)
        out2 = module(x2)

        # All positions before the final token should be unchanged.
        max_diff = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max().item()

        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=atol), (
            f"Future-token leak detected. Max diff before final position: {max_diff:.6e}"
        )

def test_attention():
    """Test MultiHeadAttention implementation."""
    print("Testing MultiHeadAttention...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # ---------------------------------------------------------------------
    # Test 1: Standard MHA forward pass
    # ---------------------------------------------------------------------
    attn = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=None,
        dropout=0.1,
        use_flash=True,
        use_qk_norm=True,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)

    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), "Output contains NaNs"
    print("✓ Standard MHA output shape correct")

    # ---------------------------------------------------------------------
    # Test 2: GQA forward pass
    # ---------------------------------------------------------------------
    n_kv_heads = 2
    attn_gqa = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dropout=0.1,
        use_flash=True,
        use_qk_norm=True,
    )

    output_gqa = attn_gqa(x)
    assert output_gqa.shape == (batch_size, seq_len, d_model), (
        f"Expected {(batch_size, seq_len, d_model)}, got {output_gqa.shape}"
    )
    assert not torch.isnan(output_gqa).any(), "GQA output contains NaNs"
    print("✓ GQA output shape correct")

    # ---------------------------------------------------------------------
    # Test 3: Different sequence lengths
    # ---------------------------------------------------------------------
    attn.eval()
    with torch.no_grad():
        for test_seq_len in [5, 20, 50]:
            x_test = torch.randn(1, test_seq_len, d_model)
            output_test = attn(x_test)
            assert output_test.shape == (1, test_seq_len, d_model), (
                f"Failed for seq_len={test_seq_len}: got {output_test.shape}"
            )
    print("✓ Works with different sequence lengths")

    # ---------------------------------------------------------------------
    # Test 4: Parameter count for standard MHA
    # ---------------------------------------------------------------------
    total_params = sum(p.numel() for p in attn.parameters())

    # Standard MHA:
    # W_q: d_model * d_model
    # W_k: d_model * d_model
    # W_v: d_model * d_model
    # W_o: d_model * d_model
    # QK-Norm scale: n_heads
    expected_params = 4 * d_model * d_model + n_heads

    assert total_params == expected_params, (
        f"Expected {expected_params:,} params, got {total_params:,}"
    )
    print(f"✓ Standard MHA parameter count correct: {total_params:,}")

    # ---------------------------------------------------------------------
    # Test 5: Parameter count for GQA
    # ---------------------------------------------------------------------
    total_params_gqa = sum(p.numel() for p in attn_gqa.parameters())

    d_k = d_model // n_heads
    expected_params_gqa = (
        d_model * d_model                      # W_q
        + d_model * (n_kv_heads * d_k)          # W_k
        + d_model * (n_kv_heads * d_k)          # W_v
        + d_model * d_model                    # W_o
        + n_heads                              # QK-Norm scale
    )

    assert total_params_gqa == expected_params_gqa, (
        f"Expected {expected_params_gqa:,} params, got {total_params_gqa:,}"
    )
    print(f"✓ GQA parameter count correct: {total_params_gqa:,}")

    # ---------------------------------------------------------------------
    # Test 6: Causal masking / no future-token leak
    # ---------------------------------------------------------------------
    assert_no_future_leak(attn, d_model=d_model, seq_len=seq_len, atol=1e-5)
    print("✓ Standard MHA causal mask prevents future-token leakage")

    assert_no_future_leak(attn_gqa, d_model=d_model, seq_len=seq_len, atol=1e-5)
    print("✓ GQA causal mask prevents future-token leakage")

    # ---------------------------------------------------------------------
    # Test 7: Manual attention path
    # ---------------------------------------------------------------------
    attn_manual = MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=None,
        dropout=0.0,
        use_flash=False,
        use_qk_norm=True,
    )

    output_manual = attn_manual(x)
    assert output_manual.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output_manual).any(), "Manual attention output contains NaNs"
    print("✓ Manual attention path works")

    print("\nAll MultiHeadAttention tests passed! ✓")


def test_differential_attention():
    """Test DifferentialAttention implementation."""
    print("\nTesting DifferentialAttention...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # ---------------------------------------------------------------------
    # Test 1: Standard DifferentialAttention forward pass
    # ---------------------------------------------------------------------
    diff_attn = DifferentialAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=None,
        layer_idx=0,
        dropout=0.1,
        use_qk_norm=True,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    output = diff_attn(x)

    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), "DifferentialAttention output contains NaNs"
    print("✓ Standard DifferentialAttention output shape correct")

    # ---------------------------------------------------------------------
    # Test 2: DifferentialAttention with GQA
    # ---------------------------------------------------------------------
    n_kv_heads = 2
    diff_attn_gqa = DifferentialAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        layer_idx=5,
        dropout=0.1,
        use_qk_norm=True,
    )

    output_gqa = diff_attn_gqa(x)

    assert output_gqa.shape == (batch_size, seq_len, d_model), (
        f"Expected {(batch_size, seq_len, d_model)}, got {output_gqa.shape}"
    )
    assert not torch.isnan(output_gqa).any(), "DifferentialAttention GQA output contains NaNs"
    print("✓ DifferentialAttention GQA output shape correct")

    # ---------------------------------------------------------------------
    # Test 3: Gradient flow
    # ---------------------------------------------------------------------
    diff_attn.train()
    x_grad = torch.randn(1, seq_len, d_model, requires_grad=True)

    out = diff_attn(x_grad)
    loss = out.sum()
    loss.backward()

    assert x_grad.grad is not None, "Input gradient is None"
    assert not torch.isnan(x_grad.grad).any(), "Input gradient contains NaNs"

    for name, param in diff_attn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaNs"

    print("✓ Gradients flow correctly through DifferentialAttention")

    # ---------------------------------------------------------------------
    # Test 4: lambda_init schedule
    # ---------------------------------------------------------------------
    diff_attn_l0 = DifferentialAttention(
        d_model=d_model,
        n_heads=n_heads,
        layer_idx=0,
    )

    diff_attn_l23 = DifferentialAttention(
        d_model=d_model,
        n_heads=n_heads,
        layer_idx=23,
    )

    assert abs(diff_attn_l0.lambda_init - 0.2) < 1e-4, (
        f"Expected lambda_init ~0.2 at layer 0, got {diff_attn_l0.lambda_init}"
    )

    assert 0.79 < diff_attn_l23.lambda_init < 0.81, (
        f"Expected lambda_init ~0.80 at layer 23, got {diff_attn_l23.lambda_init}"
    )

    print(f"  lambda_init at layer 0:  {diff_attn_l0.lambda_init:.4f} (expect ~0.20)")
    print(f"  lambda_init at layer 23: {diff_attn_l23.lambda_init:.4f} (expect ~0.80)")
    print("✓ lambda_init schedule looks correct")

    # ---------------------------------------------------------------------
    # Test 5: actual lambda value is finite
    # ---------------------------------------------------------------------
    lam = diff_attn.compute_lambda()

    assert lam.ndim == 0, f"Expected scalar lambda, got shape {lam.shape}"
    assert torch.isfinite(lam), f"Lambda is not finite: {lam.item()}"

    print(f"  actual lambda at init: {lam.item():.4f}")
    print("✓ actual lambda is finite")

    # ---------------------------------------------------------------------
    # Test 6: Causal masking / no future-token leak
    # ---------------------------------------------------------------------
    assert_no_future_leak(diff_attn, d_model=d_model, seq_len=seq_len, atol=1e-5)
    print("✓ DifferentialAttention causal mask prevents future-token leakage")

    assert_no_future_leak(diff_attn_gqa, d_model=d_model, seq_len=seq_len, atol=1e-5)
    print("✓ DifferentialAttention GQA causal mask prevents future-token leakage")

    # ---------------------------------------------------------------------
    # Test 7: Different sequence lengths
    # ---------------------------------------------------------------------
    diff_attn.eval()
    with torch.no_grad():
        for test_seq_len in [5, 20, 50]:
            x_test = torch.randn(1, test_seq_len, d_model)
            output_test = diff_attn(x_test)

            assert output_test.shape == (1, test_seq_len, d_model), (
                f"Failed for seq_len={test_seq_len}: got {output_test.shape}"
            )
            assert not torch.isnan(output_test).any(), (
                f"Output contains NaNs for seq_len={test_seq_len}"
            )

    print("✓ Works with different sequence lengths")

    print("✓ DifferentialAttention tests passed!")

def test_xsa():
    """Test Exclusive Self-Attention (XSA) variant of MultiHeadAttention."""
    print("\nTesting XSA (Exclusive Self-Attention)...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # 1. XSA forward pass (standard, no GQA)
    attn_xsa = MultiHeadAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=None,
        dropout=0.0, use_flash=True, use_qk_norm=True, use_xsa=True,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    out = attn_xsa(x)
    assert out.shape == (batch_size, seq_len, d_model), f"got {out.shape}"
    assert not torch.isnan(out).any(), "XSA output contains NaNs"
    print("✓ XSA output shape correct, no NaNs")

    # 2. XSA with GQA (the critical path — v must be repeated to n_heads
    #    before the projection subtraction)
    attn_xsa_gqa = MultiHeadAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=2,
        dropout=0.0, use_flash=True, use_qk_norm=True, use_xsa=True,
    )
    out_gqa = attn_xsa_gqa(x)
    assert out_gqa.shape == (batch_size, seq_len, d_model), f"got {out_gqa.shape}"
    assert not torch.isnan(out_gqa).any(), "XSA GQA output contains NaNs"
    print("✓ XSA GQA output shape correct, no NaNs")

    # 3. Gradient flow
    attn_xsa.train()
    x_grad = torch.randn(1, seq_len, d_model, requires_grad=True)
    loss = attn_xsa(x_grad).sum()
    loss.backward()
    assert x_grad.grad is not None and not torch.isnan(x_grad.grad).any()
    print("✓ XSA gradients flow correctly")

    # 4. Causality preserved — the projection subtraction is per-position,
    #    so changing a future token must not affect earlier outputs.
    assert_no_future_leak(attn_xsa, d_model=d_model, seq_len=seq_len, atol=1e-5)
    print("✓ XSA causal mask prevents future-token leakage")
    assert_no_future_leak(attn_xsa_gqa, d_model=d_model, seq_len=seq_len, atol=1e-5)
    print("✓ XSA GQA causal mask prevents future-token leakage")

    # 5. XSA actually changes the output (it's not a no-op vs plain attention)
    attn_plain = MultiHeadAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=None,
        dropout=0.0, use_flash=True, use_qk_norm=True, use_xsa=False,
    )
    # copy weights so only XSA differs
    attn_plain.load_state_dict(attn_xsa.state_dict())
    attn_xsa.eval(); attn_plain.eval()
    with torch.no_grad():
        diff = (attn_xsa(x) - attn_plain(x)).abs().max().item()
    assert diff > 1e-4, f"XSA output identical to plain (diff={diff}) — XSA not applied!"
    print(f"✓ XSA meaningfully changes output (max diff vs plain: {diff:.4f})")

    print("✓ XSA tests passed!")



if __name__ == "__main__":
    test_attention()
    test_differential_attention()
    test_xsa()
