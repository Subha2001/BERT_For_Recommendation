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
    ratings_df = pd.read_csv('Data/ml-1m/ratings.dat', sep='::', header=None, engine='python')
    ratings_df.columns = ['uid', 'sid', 'rating', 'timestamp']
    all_item_ids = set(ratings_df['sid'].unique())
    max_item_id = max(all_item_ids)
    args.num_items = max_item_id
    genre_ids = list(dataset.sid2genre.values())
    max_genre_id = max(genre_ids)
    args.num_genres = max_genre_id + 1
    genre_counts = {}
    for g in genre_ids:
        genre_counts[g] = genre_counts.get(g, 0) + 1
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
