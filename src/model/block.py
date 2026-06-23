import torch
import torch.nn as nn
from .attention import MultiHeadAttention, DifferentialAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .mhc import MHCResidual

class TransformerBlock(nn.Module):
    """
    Transformer block with optional mHC residual streams.

    Normal mode:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    mHC mode:
        streams = mhc_attn(streams, Attention, RMSNorm)
        streams = mhc_mlp(streams, MLP, RMSNorm)
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
            use_diff_attn: bool = True,
            use_mhc: bool = True,
            n_streams: int = 4,
            use_xsa: bool = False,
    ):
        super().__init__()

        self.use_mhc = use_mhc
        self.n_streams = n_streams

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
        elif use_xsa:
            self.attention = MultiHeadAttention(
                d_model=d_model, 
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_flash=use_flash,
                use_qk_norm=use_qk_norm,
                use_xsa=True
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
                use_xsa=False
            )

        # Pre-MLP Normalization
        self.norm2 = RMSNorm(d_model)

        # SwiGLU MLP
        self.mlp = SwiGLU(d_model, bias=False)

        if use_mhc:
            self.mhc_attn = MHCResidual(
                d_model=d_model,
                n_streams=n_streams,
                identity_bias=3.0,
                sinkhorn_iters=2,
                write_init=1.0,
            )

            self.mhc_mlp = MHCResidual(
                d_model=d_model,
                n_streams=n_streams,
                identity_bias=3.0,
                sinkhorn_iters=2,
                write_init=1.0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if not self.use_mhc:
            # First path through attention: residual connection + attention of normed input
            x = x + self.attention(self.norm1(x))
            # Second path through MLP: residual connection + SwiGLU of normed output of first path
            x = x + self.mlp(self.norm2(x))
        
            return x
        
        else:
            # in mHC mode, x is actually streams: [S, B, T, D]
            streams = x
            streams = self.mhc_attn(streams, self.attention, self.norm1)
            streams = self.mhc_mlp(streams, self.mlp, self.norm2)

            return streams



    

# ============================================================================
# TESTS
# ============================================================================

def test_transformer_block():
    """Test transformer block implementation."""
    print("=" * 60)
    print("Testing Transformer Block")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    n_streams = 4

    x = torch.randn(batch_size, seq_len, d_model)

    # ------------------------------------------------------------------
    # 1. Normal block, no mHC
    # ------------------------------------------------------------------
    print("\n1. Testing normal TransformerBlock without mHC...")

    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        use_flash=False,
        use_diff_attn=True,
        use_mhc=False,
    )

    output = block(x)

    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    )
    assert not torch.isnan(output).any()
    print("   ✓ Normal block output shape correct")

    # ------------------------------------------------------------------
    # 2. Normal block gradient flow
    # ------------------------------------------------------------------
    print("\n2. Testing normal block gradient flow...")

    x_grad = torch.randn(1, seq_len, d_model, requires_grad=True)
    output_grad = block(x_grad)
    loss = output_grad.sum()
    loss.backward()

    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print("   ✓ Normal block gradients flow correctly")

    # ------------------------------------------------------------------
    # 3. mHC block
    # ------------------------------------------------------------------
    print("\n3. Testing TransformerBlock with mHC streams...")

    block_mhc = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        use_flash=False,
        use_diff_attn=True,
        use_mhc=True,
        n_streams=n_streams,
    )

    streams = x.unsqueeze(0).repeat(n_streams, 1, 1, 1)
    output_streams = block_mhc(streams)

    assert output_streams.shape == (n_streams, batch_size, seq_len, d_model), (
        f"Expected {(n_streams, batch_size, seq_len, d_model)}, got {output_streams.shape}"
    )
    assert not torch.isnan(output_streams).any()
    print("   ✓ mHC block output shape correct")

    # ------------------------------------------------------------------
    # 4. mHC gradient flow
    # ------------------------------------------------------------------
    print("\n4. Testing mHC block gradient flow...")

    streams_grad = torch.randn(
        n_streams, 1, seq_len, d_model, requires_grad=True
    )

    output_streams_grad = block_mhc(streams_grad)
    loss = output_streams_grad.sum()
    loss.backward()

    assert streams_grad.grad is not None
    assert not torch.isnan(streams_grad.grad).any()
    print("   ✓ mHC block gradients flow correctly")

    # ------------------------------------------------------------------
    # 5. Variable sequence lengths, normal mode
    # ------------------------------------------------------------------
    print("\n5. Testing normal block with variable sequence lengths...")

    block.eval()
    with torch.no_grad():
        for test_seq_len in [5, 20, 50, 100]:
            x_test = torch.randn(1, test_seq_len, d_model)
            output_test = block(x_test)

            assert output_test.shape == (1, test_seq_len, d_model), (
                f"Failed normal block for seq_len={test_seq_len}"
            )

    print("   ✓ Normal block works with variable sequence lengths")

    # ------------------------------------------------------------------
    # 6. Variable sequence lengths, mHC mode
    # ------------------------------------------------------------------
    print("\n6. Testing mHC block with variable sequence lengths...")

    block_mhc.eval()
    with torch.no_grad():
        for test_seq_len in [5, 20, 50, 100]:
            x_test = torch.randn(1, test_seq_len, d_model)
            streams_test = x_test.unsqueeze(0).repeat(n_streams, 1, 1, 1)

            output_test = block_mhc(streams_test)

            assert output_test.shape == (n_streams, 1, test_seq_len, d_model), (
                f"Failed mHC block for seq_len={test_seq_len}"
            )

    print("   ✓ mHC block works with variable sequence lengths")

    # ------------------------------------------------------------------
    # 7. Flash Attention path, explicitly no mHC and no DiffAttn
    # ------------------------------------------------------------------
    print("\n7. Testing standard Flash Attention path...")

    block_flash = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        use_flash=True,
        use_diff_attn=False,
        use_mhc=False,
    )

    output_flash = block_flash(x)

    assert output_flash.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output_flash).any()
    print("   ✓ Standard Flash Attention path works")

    print("\n" + "=" * 60)
    print("Transformer Block Tests Passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_transformer_block()