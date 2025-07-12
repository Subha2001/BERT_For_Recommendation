from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks, per_genre_recalls_and_ndcgs
import torch

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

    def calculate_metrics(self, batch):
        # Support genre input: batch = (seqs, candidates, labels, genres) or (seqs, candidates, labels)
        if len(batch) == 4:
            seqs, candidates, labels, genres = batch # Updated newly
            scores = self.model(seqs, genres)  # B x T x V
            scores = scores[:, -1, :]  # B x V
            scores = scores.gather(1, candidates)  # B x C
            # genres: B x C (genre for each candidate)
            # labels: B x C (ground truth for each candidate)
            # Group scores and labels by genre
            genre_scores_dict = {}
            genre_labels_dict = {}
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    # Robust genre indexing
                    if hasattr(genres, 'ndim') and genres.ndim == 2:
                        genre = genres[i][j]
                    elif hasattr(genres, 'ndim') and genres.ndim == 1:
                        genre = genres[j]
                    else:
                        genre = genres[i]
                    if genre not in genre_scores_dict:
                        genre_scores_dict[genre] = []
                        genre_labels_dict[genre] = []
                    genre_scores_dict[genre].append(scores[i, j].unsqueeze(0))
                    genre_labels_dict[genre].append(labels[i, j].unsqueeze(0))
            # Stack tensors for each genre
            for genre in genre_scores_dict:
                genre_scores_dict[genre] = torch.cat(genre_scores_dict[genre], dim=0).unsqueeze(0)
                genre_labels_dict[genre] = torch.cat(genre_labels_dict[genre], dim=0).unsqueeze(0)
            metrics = per_genre_recalls_and_ndcgs(genre_scores_dict, genre_labels_dict, self.metric_ks)
        else:
            seqs, candidates, labels = batch
            scores = self.model(seqs)
            scores = scores[:, -1, :]  # B x V
            scores = scores.gather(1, candidates)  # B x C
            metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics