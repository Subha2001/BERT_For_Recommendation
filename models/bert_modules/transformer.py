import torch
def debug_tensor(name, x):
    try:
        print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}, min={x.min().item()}, max={x.max().item()}")
        if hasattr(x, 'isfinite') and not torch.isfinite(x).all():
            print(f"[DEBUG] {name} contains NaN or Inf!")
    except Exception as e:
        print(f"[DEBUG] {name}: Could not print stats due to error: {e}")
import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        debug_tensor("TransformerBlock input x", x)
        debug_tensor("TransformerBlock mask", mask)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        debug_tensor("TransformerBlock after input_sublayer", x)
        x = self.output_sublayer(x, self.feed_forward)
        debug_tensor("TransformerBlock after output_sublayer", x)
        out = self.dropout(x)
        debug_tensor("TransformerBlock after dropout", out)
        return out
