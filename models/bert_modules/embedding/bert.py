import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, num_genres=None):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        ####################################################################################
        # Added newly for genre embedding
        # num_genres: number of genres, if None, genre embedding is not used
        # genre embedding is used to add genre information to the sequence
        ####################################################################################
        # If num_genres is not None, genre embedding is created
        # If num_genres is None, genre embedding is not created
        # This allows for flexibility in using genre information
        if num_genres is not None:
            self.genre = nn.Embedding(num_genres, embed_size, padding_idx=0) # Create genre embedding that maps genre IDs to embedding vectors
        else:
            self.genre = None # If num_genres is None, genre embedding is set to None
        
    def forward(self, sequence, genre=None):
        x = self.token(sequence) + self.position(sequence)
        ####################################################################################
        # If genre is provided and genre embedding exists, add genre embedding to the output
        # This allows for the addition of genre information to the sequence embedding
        # NOTE: genre embedding will only affect output if genre input is meaningful and num_genres is set
        if self.genre is not None and genre is not None:
            x = x + self.genre(genre)
        return self.dropout(x)
