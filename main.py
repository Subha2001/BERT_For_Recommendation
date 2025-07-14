import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    ############################################################################
    # Ensure export_root exists
    ############################################################################
    # Create dataset and print genre diversity
    from datasets import dataset_factory
    dataset = dataset_factory(args)

    # Automatically set num_items and num_genres from dataset
    import pandas as pd
    ratings_df = pd.read_csv('Data/ml-1m/ratings.dat', sep='::', header=None, engine='python') # Read ratings file with custom separator
    ratings_df.columns = ['uid', 'sid', 'rating', 'timestamp'] # Set column names
    all_item_ids = set(ratings_df['sid'].unique()) # Get unique item IDs from ratings
    max_item_id = max(all_item_ids) # Find the maximum item ID
    args.num_items = max_item_id # Set num_items to max_item_id + 1
    genre_ids = list(dataset.sid2genre.values()) # Get genre IDs from dataset
    max_genre_id = max(genre_ids) # Find the maximum genre ID
    args.num_genres = max_genre_id + 1 # Set num_genres to max_genre_id + 1
    print(f"[INFO] num_genres set to {args.num_genres}") # Print number of genres
    genre_counts = {} # Initialize a dictionary to count genres
    # Count occurrences of each genre ID
    for g in genre_ids:
        genre_counts[g] = genre_counts.get(g, 0) + 1 # Increment count for each genre ID
    ######################################################################
    # Newly added code ends here
    ######################################################################
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)

    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
