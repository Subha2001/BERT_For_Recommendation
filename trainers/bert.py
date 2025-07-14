from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks, per_genre_recalls_and_ndcgs
import torch
import torch.nn as nn

class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
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
        ##############################################################################
        # Made changes in Calculate loss for BERT model
        ##############################################################################
        # Support genre input: batch = (seqs, labels, genres) or (seqs, labels)
        # NOTE: Model will use genre information only if genre input is meaningful and genre embedding exists
        if len(batch) == 3:
            seqs, labels, genres = batch  # Updated newly
            seqs = seqs.to(self.device)
            labels = labels.to(self.device)
            genres = genres.to(self.device)
            logits = self.model(seqs, genres)  # B x T x V
        else:
            seqs, labels = batch
            seqs = seqs.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(seqs)

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        ##############################################################################
        # Made changes in Calculate metrics for BERT model
        ##############################################################################
        # Support genre input: batch = (seqs, candidates, labels, genres) or (seqs, candidates, labels)
        if len(batch) == 4:
            seqs, candidates, labels, genres = batch  # Updated newly
            seqs = seqs.to(self.device)
            candidates = candidates.to(self.device)
            labels = labels.to(self.device)
            genres = genres.to(self.device)

            scores = self.model(seqs, genres)  # B x T x V
            scores = scores[:, -1, :]  # B x V

            # Gather scores for candidates to match labels shape
            if (candidates < 0).any() or (candidates >= scores.size(1)).any():
                raise ValueError(
                    f"Invalid candidate indices detected! "
                    f"Min: {candidates.min().item()}, Max: {candidates.max().item()}, "
                    f"Scores dim size: {scores.size(1)}"
                )
            scores = scores.gather(1, candidates)  # B x C

            # Group scores and labels by genre
            genre_scores_dict = {}
            genre_labels_dict = {}
            for idx in range(scores.size(0)):
                if genres.dim() == 2:
                    genre_val = genres[idx, 0].item()
                else:
                    genre_val = genres[idx].item()

                genre_scores_dict.setdefault(genre_val, []).append(scores[idx])
                genre_labels_dict.setdefault(genre_val, []).append(labels[idx])

            # Identify top 5 single genres by count
            genre_counts = {g: len(v) for g, v in genre_scores_dict.items()}
            top_5_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:5]

            # Identify multi-genre items
            multi_genre_scores = []
            multi_genre_labels = []
            if genres.dim() == 2:
                for idx in range(scores.size(0)):
                    genre_count = (genres[idx] > 0).sum().item()
                    if genre_count > 1:
                        multi_genre_scores.append(scores[idx])
                        multi_genre_labels.append(labels[idx])

            # Compute Recall@5 and Recall@1 for top 5 genres and multi-genre
            recall5_list = []
            recall1_list = []
            for genre in top_5_genres:
                if not genre_scores_dict.get(genre):
                    recall5_list.append(0.0)
                    recall1_list.append(0.0)
                    continue
                m5 = per_genre_recalls_and_ndcgs(
                    {genre: torch.stack(genre_scores_dict[genre], dim=0)},
                    {genre: torch.stack(genre_labels_dict[genre], dim=0)},
                    [5]
                )
                m1 = per_genre_recalls_and_ndcgs(
                    {genre: torch.stack(genre_scores_dict[genre], dim=0)},
                    {genre: torch.stack(genre_labels_dict[genre], dim=0)},
                    [1]
                )
                recall5_list.append(m5[genre].get('Recall@5', 0.0))
                recall1_list.append(m1[genre].get('Recall@1', 0.0))

            # Multi-genre
            if multi_genre_scores and multi_genre_labels:
                m5 = per_genre_recalls_and_ndcgs(
                    {'multi_genre': torch.stack(multi_genre_scores, dim=0)},
                    {'multi_genre': torch.stack(multi_genre_labels, dim=0)},
                    [5]
                )
                m1 = per_genre_recalls_and_ndcgs(
                    {'multi_genre': torch.stack(multi_genre_scores, dim=0)},
                    {'multi_genre': torch.stack(multi_genre_labels, dim=0)},
                    [1]
                )
                recall5_list.append(m5['multi_genre'].get('Recall@5', 0.0))
                recall1_list.append(m1['multi_genre'].get('Recall@1', 0.0))
            else:
                recall5_list.append(0.0)
                recall1_list.append(0.0)

            # Pad to always have 6 entries (5 single + 1 multi)
            while len(recall5_list) < 6:
                recall5_list.append(0.0)
            while len(recall1_list) < 6:
                recall1_list.append(0.0)

            # Build metrics dict
            metrics_dict = {
                f'Recall@5_genre{i+1}': (
                    0.0 if isinstance(s, float) and (s != s) else s
                )
                for i, s in enumerate(recall5_list[:-1])
            }
            metrics_dict['Recall@5_multigenre'] = (
                0.0 if isinstance(recall5_list[-1], float) and (recall5_list[-1] != recall5_list[-1])
                else recall5_list[-1]
            )
            metrics_dict.update({
                f'Recall@1_genre{i+1}': (
                    0.0 if isinstance(s, float) and (s != s) else s
                )
                for i, s in enumerate(recall1_list[:-1])
            })
            metrics_dict['Recall@1_multigenre'] = (
                0.0 if isinstance(recall1_list[-1], float) and (recall1_list[-1] != recall1_list[-1])
                else recall1_list[-1]
            )

            # Add all overall metrics
            overall = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
            metrics_dict.update(overall)
            # Ensure expected keys exist
            for k in self.metric_ks:
                metrics_dict.setdefault(f'NDCG@{k}', 0.0)
                metrics_dict.setdefault(f'Recall@{k}', 0.0)
            return metrics_dict

        else:
            seqs, candidates, labels = batch
            seqs = seqs.to(self.device)
            candidates = candidates.to(self.device)
            labels = labels.to(self.device)

            scores = self.model(seqs)
            scores = scores[:, -1, :]  # B x V

            # Defensive check
            if (candidates < 0).any() or (candidates >= scores.size(1)).any():
                raise ValueError(
                    f"Invalid candidate indices detected! "
                    f"Min: {candidates.min().item()}, Max: {candidates.max().item()}, "
                    f"Scores dim size: {scores.size(1)}"
                )
            scores = scores.gather(1, candidates)  # B x C

            return recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)