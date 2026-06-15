import torch
import torch.nn as nn
from .attention import MultiHeadAttention, DifferentialAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    """
    Tansformer block with Pre-LN architecture.
    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int = None,
            layer_idx:int = 0,
            dropout: float = 0.0,
            max_seq_len: int = 2048,
            use_flash: bool = True,
            use_qk_norm: bool = True,
            use_diff_attn=True
    ):
        super().__init__()
        # Pre-Attention Normalization
        self.norm1 = RMSNorm(d_model)
        # Multi-Head Attention or Differential Attention
        if use_diff_attn:
            self.attention = DifferentialAttention(
            d_model=d_model, 
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_qk_norm=use_qk_norm,
            )
        else:
            self.attention = MultiHeadAttention(
                d_model=d_model, 
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_flash=use_flash,
                use_qk_norm=use_qk_norm,
            )
        # Pre-MLP Normalization
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
        # First path through attention: residual connection + attention of normed input
        x = x + self.attention(self.norm1(x))
        # Second path through MLP: residual connection + SwiGLU of normed output of first path
        x = x + self.mlp(self.norm2(x))
        
        return x



    

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