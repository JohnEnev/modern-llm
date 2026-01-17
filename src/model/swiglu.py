import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU feed forward network. Uses the now common SwiGLU architecture.
    Has 3 learnable weight matrices: SwiGLU(x) = (SiLU(xWgate) * xWup) Wdown.
    Uses a 8/3 expansion parameter (so d_model is multiplied by this) to match the usual 4 * d_model in classic MLP.
    """
    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        hidden_dim = int(8/3 * d_model)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
