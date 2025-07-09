import torch
from models import model_factory
from dataloaders import dataloader_factory
from options import args

def test_genre_pipeline():
    # Ensure seeds are set for negative samplers
    if getattr(args, 'train_negative_sampling_seed', None) is None:
        args.train_negative_sampling_seed = 42
    if getattr(args, 'test_negative_sampling_seed', None) is None:
        args.test_negative_sampling_seed = 42

    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    model.eval()

    # Test a batch from train_loader
    batch = next(iter(train_loader))
    print("Batch element shapes:", [x.shape for x in batch])

    # Try forward pass with genre if present
    if len(batch) == 3:
        seqs, labels, genres = batch
        out = model(seqs, genres)
        print("Model output shape (train):", out.shape)
    else:
        seqs, labels = batch
        out = model(seqs)
        print("Model output shape (train, no genre):", out.shape)

    # Test a batch from test_loader
    batch = next(iter(test_loader))
    if len(batch) == 4:
        seqs, candidates, labels, genres = batch
        out = model(seqs, genres)
        print("Model output shape (test):", out.shape)
    else:
        seqs, candidates, labels = batch
        out = model(seqs)
        print("Model output shape (test, no genre):", out.shape)

if __name__ == "__main__":
    test_genre_pipeline()
