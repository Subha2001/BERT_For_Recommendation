import torch
def debug_tensor(name, x):
    try:
        print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}, min={x.min().item()}, max={x.max().item()}")
        if hasattr(x, 'isfinite') and not torch.isfinite(x).all():
            print(f"[DEBUG] {name} contains NaN or Inf!")
    except Exception as e:
        print(f"[DEBUG] {name}: Could not print stats due to error: {e}")
from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        num_genres = getattr(args, 'num_genres', None)  # Added newly

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout,
                                      num_genres=num_genres)  # Updated newly

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    # Updated newly
    def forward(self, x, genre=None):
        debug_tensor("BERT input x", x)
        if genre is not None:
            debug_tensor("BERT input genre", genre)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        debug_tensor("BERT mask", mask)
        x = self.embedding(x, genre)
        debug_tensor("BERT after embedding", x)
        for i, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(x, mask)
            debug_tensor(f"BERT after transformer block {i}", x)
        return x

    def init_weights(self):
        pass
