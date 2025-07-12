import torch
from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        # Support genre input: batch = (seqs, labels, genres) or (seqs, labels)
        if len(batch) == 3:
            seqs, labels, genres = batch # Updated newly
            logits = self.model(seqs, genres)  # B x T x V
        else:
            seqs, labels = batch
            logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, multi_genre=None):
        # batch = (seqs, candidates, labels, genres) or (seqs, candidates, labels)
        if len(batch) == 4:
            seqs, candidates, labels, genres = batch
            scores = self.model(seqs, genres)  # B x T x V
        else:
            seqs, candidates, labels = batch
            genres = None
            scores = self.model(seqs)
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        # Flatten all arrays to 1D for correct masking
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        genre_metrics = {}
        single_genres = ['Action', 'Sci-Fi', 'Thriller', 'Comedy', 'Suspense']
        if genres is not None:
            genres = genres.reshape(-1)
            genres = genres.cpu().numpy()
            for genre_name in single_genres:
                genre_id = self.args.genre2id[genre_name] if hasattr(self.args, 'genre2id') else single_genres.index(genre_name)
                idx = (genres == genre_id)
                if idx.sum() > 0:
                    genre_scores = scores[idx]
                    genre_labels = labels[idx]
                    genre_metrics[genre_name] = recalls_and_ndcgs_for_ks(genre_scores, genre_labels, self.metric_ks)
                else:
                    genre_metrics[genre_name] = None
            # Multi-genre: user-defined list of genre names
            if multi_genre:
                multi_ids = [self.args.genre2id[g] if hasattr(self.args, 'genre2id') else single_genres.index(g) for g in multi_genre]
                idx = [g in multi_ids for g in genres]
                idx = torch.tensor(idx)
                if idx.sum() > 0:
                    multi_scores = scores[idx]
                    multi_labels = labels[idx]
                    genre_metrics['Multi-Genre'] = recalls_and_ndcgs_for_ks(multi_scores, multi_labels, self.metric_ks)
                else:
                    genre_metrics['Multi-Genre'] = None
        # Overall metrics
        genre_metrics['Overall'] = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return genre_metrics
