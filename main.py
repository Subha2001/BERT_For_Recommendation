import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    # Create dataset and print genre diversity
    from datasets import dataset_factory
    dataset = dataset_factory(args)
    # Automatically set num_items and num_genres from dataset
    all_item_ids = set()
    for user_seq in dataset.train.values():
        all_item_ids.update(user_seq)
    max_item_id = max(all_item_ids)
    args.num_items = max_item_id
    genre_ids = list(dataset.sid2genre.values())
    max_genre_id = max(genre_ids)
    args.num_genres = max_genre_id + 1
    print(f"[MAIN DEBUG] Set args.num_items to {args.num_items}")
    print(f"[MAIN DEBUG] Set args.num_genres to {args.num_genres}")
    genre_counts = {}
    for g in genre_ids:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    print(f"[MAIN DEBUG] sid2genre unique genres: {list(genre_counts.keys())}")
    print(f"[MAIN DEBUG] sid2genre genre counts: {genre_counts}")
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
