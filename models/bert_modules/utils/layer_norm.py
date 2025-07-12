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
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[DEBUG] NaN or Inf detected in input to LayerNorm!")
            print(f"[DEBUG] x: {x}")
            raise ValueError("NaN or Inf in input to LayerNorm")
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("[DEBUG] NaN or Inf detected in output of LayerNorm!")
            print(f"[DEBUG] out: {out}")
            raise ValueError("NaN or Inf in output of LayerNorm")
        return out
