import torch
import os
from models import model_factory
from options import args
import numpy as np

# Genre name to ID mapping (1-based, as per MovieLens 1M)
genre_name_to_id = {
    'Action': 1,
    'Adventure': 2,
    'Animation': 3,
    "Children's": 4,
    'Comedy': 5,
    'Crime': 6,
    'Documentary': 7,
    'Drama': 8,
    'Fantasy': 9,
    'Film-Noir': 10,
    'Horror': 11,
    'Musical': 12,
    'Mystery': 13,
    'Romance': 14,
    'Sci-Fi': 15,
    'Thriller': 16,
    'War': 17,
    'Western': 18
}

# Reverse mapping for output
genre_id_to_name = {v: k for k, v in genre_name_to_id.items()}

def genre_names_to_ids(genre_names):
    return [genre_name_to_id[g] for g in genre_names]

def load_model(model_path):
    # Ensure model_init_seed is set
    if not hasattr(args, 'model_init_seed') or args.model_init_seed is None:
        args.model_init_seed = 42
    # Ensure num_items is set
    if not hasattr(args, 'num_items') or args.num_items is None:
        args.num_items = 3706
    # Ensure bert_hidden_units is set
    if not hasattr(args, 'bert_hidden_units') or args.bert_hidden_units is None:
        args.bert_hidden_units = 256  # Set to your training value if different
    # Ensure bert_max_len is set
    if not hasattr(args, 'bert_max_len') or args.bert_max_len is None:
        args.bert_max_len = 100
    # Ensure bert_dropout is set
    if not hasattr(args, 'bert_dropout') or args.bert_dropout is None:
        args.bert_dropout = 0.1
    # Ensure bert_num_blocks and bert_num_heads are set
    if not hasattr(args, 'bert_num_blocks') or args.bert_num_blocks is None:
        args.bert_num_blocks = 2
    if not hasattr(args, 'bert_num_heads') or args.bert_num_heads is None:
        args.bert_num_heads = 4
    # Ensure num_genres is set
    if not hasattr(args, 'num_genres') or args.num_genres is None:
        args.num_genres = 19  # 18 genres + 1 for padding (0)
    model = model_factory(args)
    model.eval()
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    return model

def predict_user_genre_top5(user_id, movie_id, interaction_seq, genre, model_path='downloaded_model/best_acc_model.pth'):
    """
    genre: list of genre names (strings) or IDs (ints)
    Returns: (user_id, genre, top5_movie_ids)
    """
    # Map genre names to IDs if needed
    genre_ids = [genre_name_to_id[g] if isinstance(g, str) else g for g in genre]
    model = load_model(model_path)
    # Pad input to bert_max_len
    max_len = args.bert_max_len if hasattr(args, 'bert_max_len') else 100
    pad_len = max_len - len(interaction_seq)
    padded_seq = [0] * pad_len + interaction_seq
    padded_genre = [0] * pad_len + genre_ids
    seq_tensor = torch.LongTensor([padded_seq])
    genre_tensor = torch.LongTensor([padded_genre])
    # Forward pass to get top5 movies
    with torch.no_grad():
        logits = model(seq_tensor, genre_tensor)  # [1, T, V]
        scores = logits[:, -1, :]  # [1, V]
        top5_scores, top5_indices = torch.topk(scores, 5, dim=-1)
        top5_movie_ids = top5_indices[0].cpu().numpy().tolist()
    return user_id, genre, top5_movie_ids

def predict_top5_per_genre(user_id, interaction_seq, genre_list, model_path='downloaded_model/best_acc_model.pth', genre_id_to_name=None):
    # Map genre names to IDs if needed
    genre_ids = [genre_name_to_id[g] if isinstance(g, str) else g for g in genre_list]
    model = load_model(model_path)
    max_len = args.bert_max_len if hasattr(args, 'bert_max_len') else 100
    pad_len = max_len - len(interaction_seq)
    padded_seq = [0] * pad_len + interaction_seq

    genre_to_top5 = {}
    for genre_id in genre_ids:
        # For each genre, create a genre tensor filled with that genre
        padded_genre = [genre_id] * max_len
        seq_tensor = torch.LongTensor([padded_seq])
        genre_tensor = torch.LongTensor([padded_genre])
        with torch.no_grad():
            logits = model(seq_tensor, genre_tensor)
            scores = logits[:, -1, :]
            top5_scores, top5_indices = torch.topk(scores, 5, dim=-1)
            top5_movie_ids = top5_indices[0].cpu().numpy().tolist()
        genre_to_top5[genre_id] = top5_movie_ids

    # Print as table
    if genre_id_to_name is None:
        genre_id_to_name = {v: k for k, v in genre_name_to_id.items()}
    header = ["User ID"] + [genre_id_to_name.get(gid, str(gid)) for gid in genre_ids]
    print("\t".join(header))
    row = [str(user_id)] + [", ".join(map(str, genre_to_top5[gid])) for gid in genre_ids]
    print("\t".join(row))
    return user_id, genre_to_top5

def predict_top5_genres(user_id, interaction_seq, model_path='downloaded_model/best_acc_model.pth'):
    """
    Predicts the top 5 genres for the user based on their interaction sequence.
    Returns: list of top 5 genre IDs
    """
    model = load_model(model_path)
    max_len = args.bert_max_len if hasattr(args, 'bert_max_len') else 100
    pad_len = max_len - len(interaction_seq)
    padded_seq = [0] * pad_len + interaction_seq
    # For genre prediction, try all genres and pick the top-scoring ones
    genre_scores = []
    for genre_id in range(1, 19):  # 1 to 18
        padded_genre = [genre_id] * max_len
        seq_tensor = torch.LongTensor([padded_seq])
        genre_tensor = torch.LongTensor([padded_genre])
        with torch.no_grad():
            logits = model(seq_tensor, genre_tensor)
            scores = logits[:, -1, :].max(dim=-1)[0].item()  # Max movie score for this genre
        genre_scores.append((genre_id, scores))
    # Sort by score descending and take top 5
    top5_genres = [gid for gid, _ in sorted(genre_scores, key=lambda x: x[1], reverse=True)[:5]]
    return top5_genres

if __name__ == "__main__":
    # Example usage with genre names
    user_id = 1
    movie_id = 15
    interaction_seq = [15, 25, 35, 45, 55]  # Example sequence
    genre = ['Action', 'Sci-Fi', 'Thriller', 'Comedy', 'Action']  # Example genre sequence (as names)
    result = predict_user_genre_top5(user_id, movie_id, interaction_seq, genre)
    print("User ID:", result[0])
    print("Genre:", result[1])
    print("Top 5 Movie IDs:", result[2])

    # Predict top 5 genres for the user, then recommend top 5 movies for each
    top5_genres = predict_top5_genres(user_id, interaction_seq)
    print("Predicted Top 5 Genres:", [genre_id_to_name[g] for g in top5_genres])
    predict_top5_per_genre(user_id, interaction_seq, top5_genres)
