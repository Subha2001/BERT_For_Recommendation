import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # Debug: print tensor stats before CUDA operation
        if not torch.isfinite(x).all():
            print("[LayerNorm DEBUG] x contains NaN or Inf values!")
            print("x min:", x.min().item(), "x max:", x.max().item())
            print("x shape:", x.shape)
            print("x dtype:", x.dtype)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
