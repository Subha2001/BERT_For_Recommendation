import torch
import os
import numpy as np
from models import model_factory
from options import args



# Genre name to ID mapping (1-based, as per MovieLens 1M)
genre_name_to_id = {
    'Action': 1, 'Adventure': 2, 'Animation': 3, "Children's": 4, 'Comedy': 5,
    'Crime': 6, 'Documentary': 7, 'Drama': 8, 'Fantasy': 9, 'Film-Noir': 10,
    'Horror': 11, 'Musical': 12, 'Mystery': 13, 'Romance': 14, 'Sci-Fi': 15,
    'Thriller': 16, 'War': 17, 'Western': 18
}
# Reverse mapping for output (ID to name)
genre_id_to_name = {v: k for k, v in genre_name_to_id.items()}

# Convert a list of genre names to their corresponding IDs
def genre_names_to_ids(genre_names):
    return [genre_name_to_id[g] for g in genre_names]




# Load the trained model from a checkpoint file
def load_model(model_path):
    # Ensure required args are set (set default if missing or None)
    if not hasattr(args, 'model_init_seed') or args.model_init_seed is None:
        args.model_init_seed = 42  # Set default seed
    if not hasattr(args, 'num_items') or args.num_items is None:
        args.num_items = 3706  # Set default number of items (movies)
    if not hasattr(args, 'bert_hidden_units') or args.bert_hidden_units is None:
        args.bert_hidden_units = 256  # Set default hidden units for BERT
    if not hasattr(args, 'bert_max_len') or args.bert_max_len is None:
        args.bert_max_len = 100  # Set default max sequence length
    if not hasattr(args, 'bert_dropout') or args.bert_dropout is None:
        args.bert_dropout = 0.1  # Set default dropout
    if not hasattr(args, 'bert_num_blocks') or args.bert_num_blocks is None:
        args.bert_num_blocks = 2  # Set default number of transformer blocks
    if not hasattr(args, 'bert_num_heads') or args.bert_num_heads is None:
        args.bert_num_heads = 4  # Set default number of attention heads
    if not hasattr(args, 'num_genres') or args.num_genres is None:
        args.num_genres = 19  # Set default number of genres (18 + 1 padding)

    # Create model instance using the factory
    model = model_factory(args)
    model.eval()  # Set model to evaluation mode
    # Load model weights from checkpoint
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']  # Unwrap if nested
    model_state = model.state_dict()
    # Filter out weights that don't match the model's state dict
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered_state, strict=False)  # Load weights
    return model



# Predict top 5 recommended movies for each genre in the input list
def predict_top5_per_genre(user_id, interaction_seq, genre_list, model_path='downloaded_model/best_acc_model.pth'):
    # Convert genre names to IDs if needed
    genre_ids = [genre_name_to_id[g] if isinstance(g, str) else g for g in genre_list]
    model = load_model(model_path)  # Load the trained model
    max_len = args.bert_max_len  # Get max sequence length
    pad_len = max_len - len(interaction_seq)  # Calculate padding needed
    padded_seq = [0] * pad_len + interaction_seq  # Pad the sequence

    genre_to_top5 = {}  # Dictionary to store top 5 movies per genre
    for genre_id in genre_ids:
        padded_genre = [genre_id] * max_len  # Create genre tensor for this genre
        seq_tensor = torch.LongTensor([padded_seq])  # Convert sequence to tensor
        genre_tensor = torch.LongTensor([padded_genre])  # Convert genre to tensor
        with torch.no_grad():
            logits = model(seq_tensor, genre_tensor)  # Forward pass
            scores = logits[:, -1, :]  # Get scores for last position
            top5_scores, top5_indices = torch.topk(scores, 5, dim=-1)  # Top 5
            top5_movie_ids = top5_indices[0].cpu().numpy().tolist()  # Convert to list
        genre_to_top5[genre_id] = top5_movie_ids  # Store result

    # Multi-genre recommendation (most common pair among input genres)
    sid2genre_path = 'Data/ml-1m/sid2genre.npy'  # Path to genre mapping
    sid2genre = np.load(sid2genre_path, allow_pickle=True).item() if os.path.exists(sid2genre_path) else None
    from itertools import combinations  # For genre pairs
    from collections import Counter  # For counting pairs
    genre_pairs = list(combinations(genre_ids, 2))  # All pairs of input genres
    pair_count = Counter()
    if sid2genre is not None:
        # Count how many movies exist for each genre pair
        for m, genres_of_m in sid2genre.items():
            if isinstance(genres_of_m, (list, tuple)):
                for pair in genre_pairs:
                    if pair[0] in genres_of_m and pair[1] in genres_of_m:
                        pair_count[pair] += 1
        # Pick the most common pair (with most movies)
        best_pair = pair_count.most_common(1)[0][0] if pair_count else (None, None)
        multi_pair_movies = []  # List to store top 5 movies for the best pair
        multi_pair_name = f"{genre_id_to_name.get(best_pair[0], str(best_pair[0]))} | {genre_id_to_name.get(best_pair[1], str(best_pair[1]))}"
        if best_pair[0] is not None and best_pair[1] is not None:
            # Get model scores for all movies for the best pair
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
                multi_pair_movies.append(0)  # Pad with zeros if less than 5
        else:
            multi_pair_movies = [0]*5  # Default if no valid pair
    else:
        multi_pair_movies = [0]*5  # Default if sid2genre not found
        multi_pair_name = "Multi-Genre"



    # Print results as a table
    header = ["User ID"] + [genre_id_to_name.get(gid, str(gid)) for gid in genre_ids] + [multi_pair_name]
    print("\t".join(header))  # Print header row
    row = [str(user_id)] + [", ".join(map(str, genre_to_top5[gid])) for gid in genre_ids] + [", ".join(map(str, multi_pair_movies))]
    print("\t".join(row))  # Print result row




# Main block: run test with example user and genre sequence
if __name__ == "__main__":
    user_id = 100  # Example user ID
    movie_id = 400  # Example movie ID
    interaction_seq = [250, 250, 305, 45, 550]  # Example interaction sequence
    input_genre_sequence = ['Action', 'Crime', 'Western', 'Sci-Fi', 'Adventure']  # Input genres

    print("User ID:", user_id)  # Print user ID
    print("Movie ID: ", movie_id)  # Print Movie ID
    print("Interaction Sequence:", interaction_seq)  # Print interaction sequence
    print("Input Genre Sequence:", input_genre_sequence)  # Print input genres
    print("----------OUTPUT OF EACH GENRE----------")  # Separator
    predict_top5_per_genre(user_id, interaction_seq, input_genre_sequence)  # Run prediction