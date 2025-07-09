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
    
    #  Updated newly
    def forward(self, x, genre=None):
        x = self.bert(x, genre)
        return self.out(x)
