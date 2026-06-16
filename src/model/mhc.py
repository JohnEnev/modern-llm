import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn(log_A: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Convert a matrix into a doubly-stochastic matrix via iterative
    row/column normalization in log-space (numerically stable).
    
    Args:
        log_A: [N, N] matrix of log-weights (learnable, unconstrained)
        n_iters: number of normalization iterations
    
    Returns:
        [N, N] doubly-stochastic matrix (rows sum to 1, columns sum to 1)
    """
    orig_dtype = log_A.dtype

    # Do the normalization in fp32 for stability, especially under bf16/fp16 training.
    log_P = log_A.float()

    for _ in range(n_iters):
        # Normalize rows in log-space
        # log_softmax along dim=-1 makes each row sum to 1 (in exp-space)
        log_P = F.log_softmax(log_P, dim=-1)
        # Normalize columns in log-space
        # log_softmax along dim=-2 makes each column sum to 1
        log_P = F.log_softmax(log_P, dim=-2)
    return log_P.exp().to(orig_dtype)


class MHCResidual(nn.Module):
    """Practical mHC-style residual wrapper.

    Maintains S residual streams.
    Uses a Sinkhorn-constrained residual mixing matrix.
    Reads from streams using learned read weights.
    Writes sublayer output back using stream-specific write gates.
    
    Reference:
    - Hyper-Connections / mHC-inspired residual streams
    - mHC: https://arxiv.org/abs/2512.24880
    """
    
    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        identity_bias: float = 3.0,
        sinkhorn_iters: int = 5,
        write_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Residual stream mixing matrix, projected to Birkhoff polytope by Sinkhorn.
        self.log_A = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.register_buffer(
            "identity_bias",
            torch.eye(n_streams) * identity_bias,
        )

        # Learned readout into the sublayer input.
        # Initialized to read mostly stream 0, so the model starts close to normal transformer.
        read_logits = torch.full((n_streams,), -2.0)
        read_logits[0] = 2.0
        self.read_logits = nn.Parameter(read_logits)

        # Stream-specific write gates.
        # Important: not identical, or streams may remain symmetric.
        write = torch.zeros(n_streams)
        write[0] = write_init
        if n_streams > 1:
            write[1:] = 0.1 * write_init
        self.write_gates = nn.Parameter(write)

    def mixing_matrix(self):
        return sinkhorn(
            self.log_A + self.identity_bias,
            n_iters=self.sinkhorn_iters,
        )
    
    def forward(self, streams: torch.Tensor, sublayer, norm) -> torch.Tensor:
        """
        Args:
            streams: [S, B, T, D]
            sublayer: attention or MLP module
            norm: RMSNorm module applied to the aggregated input

        Returns:
            streams: [S, B, T, D]
        """

        assert streams.dim() == 4, f"Expected streams [S, B, T, D], got {streams.shape}"

        S, B, T, D = streams.shape

        assert S == self.n_streams, f"Expected {self.n_streams} streams, got {S}"
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Step 1 - Mix residual streams with constrained doubly-stochastic matrix.
        A = self.mixing_matrix().to(dtype=streams.dtype)
        mixed = torch.einsum("ij,jbtd->ibtd", A, streams)

        # Step 2 - Aggregate streams into one sublayer input.
        read_weights = F.softmax(self.read_logits, dim=0)
        x = torch.einsum("s,sbtd->btd", read_weights, mixed)

        # Step 3 - Run normal sublayer on aggregated representation.
        delta = sublayer(norm(x))

        # Step 4 - Write sublayer output back to streams with stream-specific gates.
        write = self.write_gates.view(S, 1, 1, 1).to(dtype=delta.dtype)
        streams = mixed + write * delta.unsqueeze(0)
        
        return streams
    
# TESTS SECTION

def test_sinkhorn():
    N = 4
    log_A = torch.randn(N, N)

    P = sinkhorn(log_A, n_iters=20)

    row_sums = P.sum(dim=-1)
    col_sums = P.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones(N), atol=1e-4), row_sums
    assert torch.allclose(col_sums, torch.ones(N), atol=1e-4), col_sums
    assert torch.all(P >= 0)

    print("✓ sinkhorn works")

def test_mhc_residual():
    S = 2
    B = 2
    T = 8
    D = 128

    mhc = MHCResidual(d_model=D, n_streams=S)

    streams = torch.randn(S, B, T, D)
    sublayer = nn.Linear(D, D, bias=False)
    norm = nn.LayerNorm(D)

    out = mhc(streams, sublayer, norm)

    assert out.shape == (S, B, T, D)
    assert not torch.isnan(out).any()

    A = mhc.mixing_matrix()
    assert torch.allclose(A.sum(dim=-1), torch.ones(S), atol=1e-4)
    assert torch.allclose(A.sum(dim=-2), torch.ones(S), atol=1e-4)

    print("✓ MHCResidual works")

def test_mhc_breaks_symmetry():
    S = 2
    B = 1
    T = 8
    D = 128

    mhc = MHCResidual(d_model=D, n_streams=S)

    x = torch.randn(B, T, D)
    streams = x.unsqueeze(0).repeat(S, 1, 1, 1)

    sublayer = nn.Linear(D, D, bias=False)
    norm = nn.LayerNorm(D)

    out = mhc(streams, sublayer, norm)

    stream_diff = (out[0] - out[1]).abs().max().item()

    assert stream_diff > 0, "Streams did not diverge"
    print(f"✓ streams diverge: max diff = {stream_diff:.6f}")


if __name__ == "__main__":
    test_sinkhorn()
    test_mhc_residual()
    test_mhc_breaks_symmetry()