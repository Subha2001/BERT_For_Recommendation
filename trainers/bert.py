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
        import numpy as np
        print('DEBUG: scores shape:', scores.shape)
        print('DEBUG: labels shape:', labels.shape)
        if genres is not None:
            print('DEBUG: genres shape:', genres.shape)
            print('DEBUG: unique genres:', np.unique(genres.cpu().numpy()))
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

        # Save original scores and labels for overall metrics
        scores_2d = scores
        labels_2d = labels

        # Flatten only for genre filtering
        genre_metrics = {}
        single_genres = ['Action', 'Sci-Fi', 'Thriller', 'Comedy', 'Suspense']
        if genres is not None:
            flat_scores = scores_2d.reshape(-1)
            flat_labels = labels_2d.reshape(-1)
            flat_genres = genres.reshape(-1).cpu().numpy()
            # Only compute genre metrics if shapes match
            if flat_genres.shape[0] == flat_scores.shape[0] == flat_labels.shape[0]:
                for genre_name in single_genres:
                    genre_id = self.args.genre2id[genre_name] if hasattr(self.args, 'genre2id') else single_genres.index(genre_name)
                    idx = (flat_genres == genre_id)
                    print(f'DEBUG: genre {genre_name} (id {genre_id}) mask count:', idx.sum())
                    if idx.sum() > 0:
                        genre_scores = flat_scores[idx]
                        genre_labels = flat_labels[idx]
                        # Reshape to (N, 1) for metric functions
                        genre_metrics[genre_name] = recalls_and_ndcgs_for_ks(genre_scores.unsqueeze(1), genre_labels.unsqueeze(1), self.metric_ks)
                    else:
                        genre_metrics[genre_name] = None
                # Multi-genre: user-defined list of genre names
                if multi_genre:
                    multi_ids = [self.args.genre2id[g] if hasattr(self.args, 'genre2id') else single_genres.index(g) for g in multi_genre]
                    idx = [g in multi_ids for g in flat_genres]
                    print(f'DEBUG: multi-genre {multi_genre} mask count:', np.sum(idx))
                    idx = torch.tensor(idx)
                    if idx.sum() > 0:
                        multi_scores = flat_scores[idx]
                        multi_labels = flat_labels[idx]
                        genre_metrics['Multi-Genre'] = recalls_and_ndcgs_for_ks(multi_scores.unsqueeze(1), multi_labels.unsqueeze(1), self.metric_ks)
                    else:
                        genre_metrics['Multi-Genre'] = None
            else:
                # Shapes do not match, skip genre metrics
                for genre_name in single_genres:
                    genre_metrics[genre_name] = None
                genre_metrics['Multi-Genre'] = None
        # Overall metrics
        genre_metrics['Overall'] = recalls_and_ndcgs_for_ks(scores_2d, labels_2d, self.metric_ks)
        return genre_metrics