import torch
from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'
    
    #####################################################################
    # Forward pass
    # x: input tensor, genre: optional genre tensor
    # Returns: output tensor after passing through BERT and linear layer
    #####################################################################
    def forward(self, x, genre=None):
        x = self.bert(x, genre)
        out = self.out(x)
        return out
