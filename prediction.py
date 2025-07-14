import torch
import sys
import os
from models import model_factory
from options import args
import numpy as np

TF_ENABLE_ONEDNN_OPTS=0 # Disable oneDNN optimizations for reproducibility

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
    # Filter out genre embedding if shape mismatch
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
    model.load_state_dict(filtered_state, strict=False)
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
    if genre_id_to_name is None:
        genre_id_to_name = {v: k for k, v in genre_name_to_id.items()}
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

    # Dynamic top-1 multi-genre column based on user's top 5 genres
    from itertools import combinations
    from collections import Counter
    sid2genre = None
    if hasattr(model, 'sid2genre'):
        sid2genre = model.sid2genre
    else:
        sid2genre_path = 'Data/ml-1m/sid2genre.npy'
        if os.path.exists(sid2genre_path):
            sid2genre = np.load(sid2genre_path, allow_pickle=True).item()
    # Find all pairs from top 5 genres
    genre_pairs = list(combinations(genre_ids, 2))
    pair_count = Counter()
    if sid2genre is not None:
        # Count how many movies exist for each pair
        for m, genres_of_m in sid2genre.items():
            if isinstance(genres_of_m, (list, tuple)):
                for pair in genre_pairs:
                    if pair[0] in genres_of_m and pair[1] in genres_of_m:
                        pair_count[pair] += 1
        # Pick the most common pair (with most movies)
        if pair_count:
            best_pair = pair_count.most_common(1)[0][0]
        else:
            best_pair = genre_pairs[0] if genre_pairs else (None, None)
        multi_pair_movies = []
        multi_pair_name = f"{genre_id_to_name.get(best_pair[0], str(best_pair[0]))} | {genre_id_to_name.get(best_pair[1], str(best_pair[1]))}"
        if best_pair[0] is not None and best_pair[1] is not None:
            # Get model scores for all movies
            seq_tensor = torch.LongTensor([padded_seq])
            genre_tensor = torch.LongTensor([[best_pair[0]] * max_len])
            with torch.no_grad():
                logits = model(seq_tensor, genre_tensor)
                scores = logits[:, -1, :].squeeze().cpu().numpy()
            # Filter movies with both genres in the best pair
            candidate_movies = [m for m, genres_of_m in sid2genre.items() if isinstance(genres_of_m, (list, tuple)) and best_pair[0] in genres_of_m and best_pair[1] in genres_of_m]
            # Sort by model score
            candidate_scores = [(m, scores[m]) for m in candidate_movies if m < len(scores)]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            multi_pair_movies = [m for m, _ in candidate_scores[:5]]
            while len(multi_pair_movies) < 5:
                multi_pair_movies.append(0)
        else:
            multi_pair_movies = [0]*5
    else:
        multi_pair_movies = [0]*5
        multi_pair_name = "Multi-Genre"

    # Print as table with user_id repeated 5 times
    header = ["User ID"] + [genre_id_to_name.get(gid, str(gid)) for gid in genre_ids] + [multi_pair_name]
    print("\t".join(header))
    row = [str(user_id)] * 5 + [", ".join(map(str, genre_to_top5[gid])) for gid in genre_ids] + [", ".join(map(str, multi_pair_movies))]
    print("\t".join(row))

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
    user_id = 100
    movie_id = 400
    interaction_seq = [250, 250, 305, 45, 55]  # Example sequence
    genre = ['Western', 'Horror', 'Crime', 'Comedy', 'Action']  # Example genre sequence (as names)
    result = predict_user_genre_top5(user_id, movie_id, interaction_seq, genre)
    print("User ID:", result[0])
    print("Movie ID:", movie_id)
    print("Genre:", result[1])
    print("Top 5 Movie IDs:", result[2])

    # Predict top 5 genres for the user, then recommend top 5 movies for each
    top5_genres = predict_top5_genres(user_id, interaction_seq)
    print("Predicted Top 5 Genres:", [genre_id_to_name[g] for g in top5_genres])
    predict_top5_per_genre(user_id, interaction_seq, top5_genres)

    # ===== Debugging: Check if genre input affects model output =====
    print("\n[DEBUG] Checking if genre input affects model output...")
    genre_test_1 = ['Action', 'Adventure', 'Western', 'Sci-Fi', 'Mystery']
    genre_test_2 = ['Comedy', 'Drama', 'Romance', 'War', 'Horror']
    result_test_1 = predict_user_genre_top5(user_id, movie_id, interaction_seq, genre_test_1)
    result_test_2 = predict_user_genre_top5(user_id, movie_id, interaction_seq, genre_test_2)
    print(f"Genre 1: {genre_test_1}\nTop 5 Movie IDs: {result_test_1[2]}")
    print(f"Genre 2: {genre_test_2}\nTop 5 Movie IDs: {result_test_2[2]}")
    print(result_test_1)
    print(result_test_2)
    if result_test_1[2] != result_test_2[2]:
        print("[RESULT] Model output changes with genre input. Genre embedding is working.")
    else:
        print("[RESULT] Model output does NOT change with genre input. Check training and genre embedding.")

    # ===== Debugging: Print genre embedding weights =====
    print("\n[DEBUG] Inspecting genre embedding weights in loaded model...")
    model = load_model('downloaded_model/best_acc_model.pth')
    genre_emb = None
    try:
        genre_emb = model.embedding.genre
    except AttributeError:
        print("No genre embedding found in model.")
    if genre_emb is None:
        print("Genre embedding is not initialized.")
    else:
        weights = genre_emb.weight.detach().cpu().numpy()
        print("Genre embedding weights shape:", weights.shape)
        print("First 5 genre embedding vectors:")
        print(weights[:5])
        print("Mean of genre embedding weights:", weights.mean())
        print("Std of genre embedding weights:", weights.std())
        if abs(weights.mean()) < 1e-3 and weights.std() < 1e-3:
            print("WARNING: Genre embedding weights are nearly zero. Model will ignore genre input.")
        else:
            print("Genre embedding weights are non-trivial.")