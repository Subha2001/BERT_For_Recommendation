from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        self.dataset = dataset
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        # Always use sid2genre from dataset
        self.sid2genre = self.dataset.sid2genre
        # ...existing code...

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        ###########################################################################
        # Create BertTrainDataset with sid2genre
        ###########################################################################
        # sid2genre should be available in self or constructed from dataset
        sid2genre = getattr(self, 'sid2genre', None) # Try to get sid2genre attribute from self
        # If sid2genre is not found
        if sid2genre is None:
            # Check if dataset has sid2genre
            if hasattr(self, 'dataset') and hasattr(self.dataset, 'sid2genre'): 
                sid2genre = self.dataset.sid2genre # Use sid2genre from dataset
                # If sid2genre is not available in dataset
            else: 
                # fallback: create a dummy mapping (all genres 0)
                all_sids = set() # Initialize a set to collect all sids
                # Iterate over all user sequences in training data
                for user_seq in self.train.values(): 
                    all_sids.update(user_seq) # Add all sids from user_seq to all_sids set
                sid2genre = {sid: 0 for sid in all_sids} # Map each sid to genre 0 (dummy mapping)
                ###########################################################################
                # End of newly added code
                ###########################################################################
        dataset = BertTrainDataset(self.train, sid2genre, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        ###########################################################################
        # Create BertEvalDataset with sid2genre
        ###########################################################################
        sid2genre = getattr(self, 'sid2genre', None) # Try to get sid2genre attribute from self
        # If sid2genre is not found
        if sid2genre is None:
            # Check if dataset has sid2genre
            if hasattr(self, 'dataset') and hasattr(self.dataset, 'sid2genre'):
                sid2genre = self.dataset.sid2genre # Use sid2genre from dataset
            else:
                # fallback: create a dummy mapping (all genres 0)
                all_sids = set() # Initialize a set to collect all sids
                # Iterate over all user sequences in training data
                for user_seq in self.train.values():
                    all_sids.update(user_seq) # Add all sids from user_seq to all_sids set
                sid2genre = {sid: 0 for sid in all_sids} # Map each sid to genre 0 (dummy mapping)
                ###########################################################################
                # End of newly added code
                ###########################################################################
        dataset = BertEvalDataset(self.train, sid2genre, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, sid2genre, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.sid2genre = sid2genre  # Added newly
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        # Check for all-zero genre sequences (warn user)
        all_genres = [self.sid2genre.get(s, 0) for user in self.users for s in self._getseq(user)]
        if all(g == 0 for g in all_genres):
            print("[WARNING] All genre sequences are zero. Model will not learn genre information.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        genres = [self.sid2genre.get(s, 0) for s in seq]

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        # Truncate all to max_len
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        genres = genres[-self.max_len:]
        # Pad all to max_len
        pad_len = self.max_len - len(tokens)
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels
        genres = [0] * pad_len + genres
        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(genres)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, sid2genre, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.sid2genre = sid2genre # Added newly
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        genres = [self.sid2genre.get(s, 0) for s in seq]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = seq + [self.mask_token]
        genres = genres + [0]  # genre for mask token
        # Truncate all to max_len
        seq = seq[-self.max_len:]
        genres = genres[-self.max_len:]
        # Pad all to max_len
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        genres = [0] * padding_len + genres
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor(genres)