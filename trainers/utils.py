def recall_at_k(pred_list, gt_set, k):
    """
    Compute recall@k for a single user/column.
    pred_list: list of predicted movie ids (length >= k)
    gt_set: set of ground truth movie ids
    k: int
    Returns: recall@k (float)
    """
    if not gt_set:
        return None
    pred_topk = set(pred_list[:k])
    return len(pred_topk & gt_set) / min(len(gt_set), k)
import torch


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


# New function for per-genre and multi-genre evaluation
def per_genre_recalls_and_ndcgs(scores_dict, labels_dict, ks):
    """
    scores_dict: dict of {genre: scores_tensor}
    labels_dict: dict of {genre: labels_tensor}
    ks: list of k values
    Returns: dict of metrics for each genre and overall
    """
    genre_metrics = {}
    for genre in scores_dict:
        genre_scores = scores_dict[genre]
        genre_labels = labels_dict[genre]
        genre_metrics[genre] = recalls_and_ndcgs_for_ks(genre_scores, genre_labels, ks)
    # Optionally, compute overall metrics by averaging across genres
    overall_metrics = {}
    for k in ks:
        recall_list = [genre_metrics[g]['Recall@%d' % k] for g in genre_metrics if 'Recall@%d' % k in genre_metrics[g]]
        ndcg_list = [genre_metrics[g]['NDCG@%d' % k] for g in genre_metrics if 'NDCG@%d' % k in genre_metrics[g]]
        if recall_list:
            overall_metrics['Recall@%d' % k] = sum(recall_list) / len(recall_list)
        if ndcg_list:
            overall_metrics['NDCG@%d' % k] = sum(ndcg_list) / len(ndcg_list)
    genre_metrics['Overall'] = overall_metrics
    return genre_metrics