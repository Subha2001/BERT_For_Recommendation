import torch
import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    ##########################################################################
    # Forward pass
    # Purpose: Apply a sublayer to the input, followed by dropout and a residual connection.
    # Returns:
    #   - out: The output tensor after applying the sublayer, dropout, and adding the residual connection.
    ##########################################################################
    def forward(self, x, sublayer):
        normed = self.norm(x)
        sublayer_out = sublayer(normed)
        dropped = self.dropout(sublayer_out)
        out = x + dropped
        return out
