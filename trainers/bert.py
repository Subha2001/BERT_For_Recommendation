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
        # batch = (seqs, candidates, labels, genres) or (seqs, candidates, labels)
        # batch = (seqs, candidates, labels, sequence_genres, candidate_genres) or (seqs, candidates, labels)
        if len(batch) == 5:
            seqs, candidates, labels, sequence_genres, candidate_genres = batch
            scores = self.model(seqs, sequence_genres)  # B x T x V
            genres = candidate_genres
        elif len(batch) == 4:
            seqs, candidates, labels, genres = batch
            scores = self.model(seqs, genres)  # B x T x V
        else:
            seqs, candidates, labels = batch
            genres = None
            scores = self.model(seqs)
        # Remove debug prints for cleaner output

        scores = scores[:, -1, :]  # B x V
        # Debug: check candidate and label indices
        max_cand = candidates.max().item() if hasattr(candidates, 'max') else None
        min_cand = candidates.min().item() if hasattr(candidates, 'min') else None
        vocab_size = scores.size(1)
        if max_cand is not None and (max_cand >= vocab_size or min_cand < 0):
            print(f"[DEBUG] Candidate indices out of bounds: min={min_cand}, max={max_cand}, vocab_size={vocab_size}")
            print(f"[DEBUG] Candidates: {candidates}")
            assert max_cand < vocab_size and min_cand >= 0, "Candidate indices out of bounds!"
        scores = scores.gather(1, candidates)  # B x C

        # Calculate genre-based metrics
        genre_metrics = {}
        single_genres = ['Action', 'Sci-Fi', 'Thriller', 'Comedy', 'Suspense']
        if genres is not None:
            flat_scores = scores.reshape(-1)
            flat_labels = labels.reshape(-1)
            flat_genres = genres.reshape(-1).cpu().numpy()
            # Only compute genre metrics if shapes match
            if flat_genres.shape[0] == flat_scores.shape[0] == flat_labels.shape[0]:
                for genre_name in single_genres:
                    genre_id = self.args.genre2id[genre_name] if hasattr(self.args, 'genre2id') else single_genres.index(genre_name)
                    idx = (flat_genres == genre_id)
                    if idx.sum() > 0:
                        genre_scores = flat_scores[idx]
                        genre_labels = flat_labels[idx]
                        genre_metrics[genre_name] = recalls_and_ndcgs_for_ks(genre_scores.unsqueeze(1), genre_labels.unsqueeze(1), self.metric_ks)
                # Multi-genre: user-defined list of genre names
                if multi_genre:
                    multi_ids = [self.args.genre2id[g] if hasattr(self.args, 'genre2id') else single_genres.index(g) for g in multi_genre]
                    idx = [g in multi_ids for g in flat_genres]
                    idx = torch.tensor(idx)
                    if idx.sum() > 0:
                        multi_scores = flat_scores[idx]
                        multi_labels = flat_labels[idx]
                        genre_metrics['Multi-Genre'] = recalls_and_ndcgs_for_ks(multi_scores.unsqueeze(1), multi_labels.unsqueeze(1), self.metric_ks)
        # Only return non-empty genre metrics
        filtered_metrics = {k: v for k, v in genre_metrics.items() if v is not None}
        return filtered_metrics