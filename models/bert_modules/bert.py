import torch
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
        # num_genres is an optional argument, default is None
        # If provided, it will be used for genre embedding
        # If not provided, genre embedding will not be used
        # This is useful for models that do not use genre information
        num_genres = getattr(args, 'num_genres', None)  # Added newly

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout,
                                      num_genres=num_genres)  # Updated newly

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    ###################################################################################
    # Forward pass
    # x: input sequence, shape [batch_size, seq_len]
    # genre: genre information, shape [batch_size, seq_len] (optional)
    ###################################################################################
    def forward(self, x, genre=None):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, genre)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x

    def init_weights(self):
        pass
