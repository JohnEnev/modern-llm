import torch
import math


def compute_rope_frequencies(d_k: int, max_seq_len: int = 2048, base: float = 10000.0):
    """
    Compute rotation frequencies for RoPE.
    Returns (cos, sin) tensors instead of complex exponentials for torch.compile compatibility.

    Args:
        d_k: Head dimension (has to be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default is 10 000)

    Returns:
        cos_freqs: Tensor of shape [max_seq_len, d_k//2] containing cos(m * theta)
        sin_freqs: Tensor of shape [max_seq_len, d_k//2] containing sin(m * theta)
    """
    # Step 1: Compute theta values
    # theta_i = base^(-2i/dk) for i = 0, 1, ..., d_k//2-1
    i = torch.arange(0, d_k, 2).float()  # [0, 2, 4, ... d_k-2]
    thetas = 1.0 / base ** (i / d_k)  # [d_k//2]

    # Step 2: Create position indices
    positions = torch.arange(max_seq_len)  # [0, 1, 2, ..., max_seq_len-1]

    # Step 3: Compute outer product m * theta (all position-frequency pairs)
    angles = torch.outer(positions, thetas)  # [max_seq_len, d_k//2]

    # Step 4: Return cos and sin directly (avoids complex numbers, torch.compile friendly)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Apply rotary position embeddings using real-valued arithmetic.
    Previously used complex numbers, but torch.compile doesn't support complex ops efficiently.
    The math is identical: (a + bi)(cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)

    Args:
        x: Input tensor of shape [batch, seq_len, n_heads, d_k]
        freqs: Tuple of (cos, sin) each of shape [seq_len, d_k//2]

    Returns:
        Rotated tensor of same shape as x
    """
    # Step 1: Unpack cos/sin and move to same device as x
    cos_f, sin_f = freqs
    cos_f = cos_f.to(x.device).unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_k//2]
    sin_f = sin_f.to(x.device).unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_k//2]

    # Step 2: Split x into even/odd pairs (these are our "real" and "imaginary" parts)
    x1 = x[..., 0::2]  # [batch, seq, heads, d_k//2] — "real" components
    x2 = x[..., 1::2]  # [batch, seq, heads, d_k//2] — "imaginary" components

    # Step 3: Apply rotation using real arithmetic
    # This is the complex multiplication (a+bi)(cos+i*sin) expanded out:
    #   real part: a*cos - b*sin
    #   imag part: a*sin + b*cos
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x1 * sin_f + x2 * cos_f

    # Step 4: Interleave real and imaginary parts back together
    # stack gives [batch, seq, heads, d_k//2, 2], flatten merges last two dims to [batch, seq, heads, d_k]
    return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(x)


class RoPECache:
    """Cache for precomputed RoPE frequencies."""

    def __init__(self, d_k: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.base = base
        self.cos_f, self.sin_f = compute_rope_frequencies(d_k, max_seq_len, base)

    def get_freqs(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (cos, sin) frequencies for a given sequence length."""
        if seq_len > self.max_seq_len:
            # Recompute if sequence is longer than cache
            self.cos_f, self.sin_f = compute_rope_frequencies(self.d_k, seq_len, self.base)
            self.max_seq_len = seq_len
        return self.cos_f[:seq_len], self.sin_f[:seq_len]