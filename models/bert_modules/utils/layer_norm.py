import torch
def debug_tensor(name, x):
    try:
        print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}, min={x.min().item()}, max={x.max().item()}")
        if hasattr(x, 'isfinite') and not torch.isfinite(x).all():
            print(f"[DEBUG] {name} contains NaN or Inf!")
    except Exception as e:
        print(f"[DEBUG] {name}: Could not print stats due to error: {e}")
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
        debug_tensor("LayerNorm input x", x)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        debug_tensor("LayerNorm output", out)
        return out
