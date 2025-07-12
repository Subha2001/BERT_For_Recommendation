from .base import AbstractTrainer

def debug_tensor(name, x):
    try:
        print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}, min={x.min().item()}, max={x.max().item()}")
        if not torch.isfinite(x).all():
            print(f"[DEBUG] {name} contains NaN or Inf!")
    except Exception as e:
        print(f"[DEBUG] {name}: Could not print stats due to error: {e}")

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
            debug_tensor("model output (scores)", scores)
            scores = scores[:, -1, :]  # B x V
            debug_tensor("scores after slicing", scores)
            debug_tensor("candidates", candidates)
            print("[DEBUG] scores dim size:", scores.size(1))
            # Check for out-of-bounds indices
            if candidates.max().item() >= scores.size(1):
                print("[ERROR] Candidate index out of bounds! Max candidate:", candidates.max().item(), "scores dim size:", scores.size(1))
            scores = scores.gather(1, candidates)  # B x C
            debug_tensor("scores after gather", scores)
            debug_tensor("labels", labels)
            debug_tensor("genres", genres)
            # genres: B x C (genre for each candidate)
            # labels: B x C (ground truth for each candidate)
            # Group scores and labels by genre (robust flattening)
            flat_scores = scores.flatten()
            flat_labels = labels.flatten()
            if hasattr(genres, 'flatten'):
                flat_genres = genres.flatten()
            else:
                flat_genres = torch.tensor(genres).flatten()

            # Truncate all arrays to the shortest length to avoid shape mismatch
            min_len = min(flat_scores.shape[0], flat_labels.shape[0], flat_genres.shape[0])
            flat_scores = flat_scores[:min_len]
            flat_labels = flat_labels[:min_len]
            flat_genres = flat_genres[:min_len]

            genre_scores_dict = {}
            genre_labels_dict = {}
            for idx in range(min_len):
                genre = flat_genres[idx].item()
                if genre not in genre_scores_dict:
                    genre_scores_dict[genre] = []
                    genre_labels_dict[genre] = []
                genre_scores_dict[genre].append(flat_scores[idx].unsqueeze(0))
                genre_labels_dict[genre].append(flat_labels[idx].unsqueeze(0))

            # Identify top 5 single genres by count
            genre_counts = {g: len(genre_scores_dict[g][0]) for g in genre_scores_dict}
            top_5_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:5]


            # Truncate all arrays to the shortest length to avoid shape mismatch
            min_len = min(flat_scores.shape[0], flat_labels.shape[0], flat_genres.shape[0])
            flat_scores = flat_scores[:min_len]
            flat_labels = flat_labels[:min_len]
            flat_genres = flat_genres[:min_len]

            genre_scores_dict = {}
            genre_labels_dict = {}
            for idx in range(min_len):
                genre = flat_genres[idx].item()
                if genre not in genre_scores_dict:
                    genre_scores_dict[genre] = []
                    genre_labels_dict[genre] = []
                genre_scores_dict[genre].append(flat_scores[idx].unsqueeze(0))
                genre_labels_dict[genre].append(flat_labels[idx].unsqueeze(0))

            # Identify top 5 single genres by count
            genre_counts = {g: len(genre_scores_dict[g]) for g in genre_scores_dict}
            top_5_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:5]

            # Identify multi-genre (assume it contains '|' in the genre name)
            multi_genre = None
            for g in genre_scores_dict:
                if isinstance(g, str) and '|' in g:
                    multi_genre = g
                    break

            # ...existing code...

            metrics_list = []
            # Compute Recall@5 for top 5 single genres
            for genre in top_5_genres:
                score_len = len(genre_scores_dict.get(genre, []))
                label_len = len(genre_labels_dict.get(genre, []))
                print(f"Genre: {genre}, score list len: {score_len}, label list len: {label_len}")
                if score_len == 0 or label_len == 0:
                    print(f"Skipping empty genre: {genre}")
                    metrics_list.append(0.0)
                    continue
                m = per_genre_recalls_and_ndcgs(
                    {genre: torch.stack(genre_scores_dict[genre], dim=0)},
                    {genre: torch.stack(genre_labels_dict[genre], dim=0)},
                    self.metric_ks
                )
                metrics_list.append(m[genre].get('Recall@5', 0.0))

            # Compute Recall@5 for multi-genre
            if multi_genre is not None:
                score_len = len(genre_scores_dict.get(multi_genre, []))
                label_len = len(genre_labels_dict.get(multi_genre, []))
                print(f"Multi-genre: {multi_genre}, score list len: {score_len}, label list len: {label_len}")
                if score_len == 0 or label_len == 0:
                    print(f"Skipping empty multi-genre: {multi_genre}")
                    metrics_list.append(0.0)
                else:
                    m = per_genre_recalls_and_ndcgs(
                        {multi_genre: torch.stack(genre_scores_dict[multi_genre], dim=0)},
                        {multi_genre: torch.stack(genre_labels_dict[multi_genre], dim=0)},
                        self.metric_ks
                    )
                    metrics_list.append(m[multi_genre].get('Recall@5', 0.0))
            else:
                metrics_list.append(0.0)  # If no multi-genre found

            # Pad metrics_list to always have 6 elements (5 single genres + 1 multi-genre)
            while len(metrics_list) < 6:
                metrics_list.append(0.0)

            # Return as dict for base trainer compatibility
            metrics_dict = {f'Recall@5_genre{i+1}': score for i, score in enumerate(metrics_list[:-1])}
            metrics_dict['Recall@5_multigenre'] = metrics_list[-1]
            # Also add overall metrics for logger compatibility
            overall_metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
            metrics_dict.update(overall_metrics)
            # Ensure all expected keys are present for logger compatibility
            for k in self.metric_ks:
                metrics_dict.setdefault(f'NDCG@{k}', 0.0)
                metrics_dict.setdefault(f'Recall@{k}', 0.0)
            return metrics_dict
        else:
            seqs, candidates, labels = batch
            scores = self.model(seqs)
            scores = scores[:, -1, :]  # B x V
            scores = scores.gather(1, candidates)  # B x C
            metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics