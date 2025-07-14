import torch

#####################################################################################
# Added newly
#####################################################################################
def recall_at_k(pred_list, gt_set, k):
    """
    Compute recall@k for a single user/column.
    pred_list: list of predicted movie ids (length >= k)
    gt_set: set of ground truth movie ids
    k: int
    Returns: recall@k (float)
    """
    # Defensive check: if there are no ground-truth items, recall is undefined
    if not gt_set: # Ground truth set empty?
        return None # Return None to indicate we can’t compute recall
    
    # Take only the top-k predictions
    pred_topk = set(pred_list[:k]) # slice first k items and convert to set
    # Compute intersection between predicted and actual, normalize by min(|gt|, k)
    return len(pred_topk & gt_set) / min(len(gt_set), k)

###########################################################################
# End of newly added code
###########################################################################

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
        # Defensive check for out-of-bounds indices
        if (cut >= labels_float.size(1)).any():
            raise ValueError(f"Invalid indices in cut: max {cut.max().item()}, labels dim {labels_float.size(1)}")
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

# #####################################################################################
# New function for per-genre and multi-genre evaluation
# #####################################################################################
def per_genre_recalls_and_ndcgs(scores_dict, labels_dict, ks):
    """
    scores_dict: dict of {genre: scores_tensor}
    labels_dict: dict of {genre: labels_tensor}
    ks: list of k values
    Returns: dict of metrics for each genre and overall
    """
    # Initialize dictionary to store metrics per genre
    genre_metrics = {}
    # Iterate over each genre and compute metrics
    for genre in scores_dict:
        genre_scores = scores_dict[genre] # Tensor of scores for the genre
        genre_labels = labels_dict[genre] # Tensor of labels for the genre
        genre_metrics[genre] = recalls_and_ndcgs_for_ks(genre_scores, genre_labels, ks) # Compute metrics for the genre
    
    # Optionally, compute overall metrics by averaging across genres
    overall_metrics = {} # Initialize overall metrics
    # Iterate over each k value to compute overall recall and NDCG
    for k in ks:
        # Collect recall and NDCG for each genre for the current k
        recall_list = [genre_metrics[g]['Recall@%d' % k] for g in genre_metrics if 'Recall@%d' % k in genre_metrics[g]]
        ndcg_list = [genre_metrics[g]['NDCG@%d' % k] for g in genre_metrics if 'NDCG@%d' % k in genre_metrics[g]]
        
        # Compute overall recall and NDCG for the current k
        if recall_list:
            overall_metrics['Recall@%d' % k] = sum(recall_list) / len(recall_list) # Average recall across genres
        
        # Compute overall NDCG for the current k
        if ndcg_list:
            overall_metrics['NDCG@%d' % k] = sum(ndcg_list) / len(ndcg_list) # Average NDCG across genres

    genre_metrics['Overall'] = overall_metrics # Add overall metrics to the genre metrics dictionary
    return genre_metrics

###############################################################################
# End of newly added code
###############################################################################