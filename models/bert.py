import torch
def debug_tensor(name, x, min_allowed=None, max_allowed=None):
    try:
        if hasattr(x, 'isfinite') and not torch.isfinite(x).all():
            print(f"[DEBUG] {name} contains NaN or Inf! shape={x.shape}, dtype={x.dtype}, min={x.min().item()}, max={x.max().item()}")
        if min_allowed is not None and x.min().item() < min_allowed:
            print(f"[DEBUG] {name} min value {x.min().item()} below allowed {min_allowed}")
        if max_allowed is not None and x.max().item() > max_allowed:
            print(f"[DEBUG] {name} max value {x.max().item()} above allowed {max_allowed}")
    except Exception as e:
        print(f"[DEBUG] {name}: Could not print stats due to error: {e}")
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
        debug_tensor("BERTModel input x", x)
        if genre is not None:
            debug_tensor("BERTModel input genre", genre)
        x = self.bert(x, genre)
        debug_tensor("BERTModel output from bert", x, min_allowed=0)
        out = self.out(x)
        debug_tensor("BERTModel output final", out, min_allowed=0)
        return out
