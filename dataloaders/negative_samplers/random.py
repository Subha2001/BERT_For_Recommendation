from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items (genre-balanced)')

        # Try to get sid2genre from self or dataset
        sid2genre = getattr(self, 'sid2genre', None)
        if sid2genre is None:
            # Try to get from dataset attribute if available
            if hasattr(self, 'dataset') and hasattr(self.dataset, 'sid2genre'):
                sid2genre = self.dataset.sid2genre
            else:
                # fallback: all genres 0
                sid2genre = {sid: 0 for sid in range(1, self.item_count + 1)}

        # Build genre to sid mapping
        genre2sids = {}
        for sid in range(1, self.item_count + 1):
            genres = sid2genre.get(sid, 0)
            # If genres is a list (multi-genre), add sid to all
            if isinstance(genres, (list, tuple)):
                for g in genres:
                    genre2sids.setdefault(g, set()).add(sid)
            else:
                genre2sids.setdefault(genres, set()).add(sid)

        all_genres = sorted(genre2sids.keys())

        for user in trange(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            per_genre = max(1, self.sample_size // len(all_genres))
            genre_sample_counts = {g: 0 for g in all_genres}
            # First, try to sample per_genre from each genre
            for g in all_genres:
                available = list(genre2sids[g] - seen)
                np.random.shuffle(available)
                count = 0
                for sid in available:
                    if sid not in samples:
                        samples.append(sid)
                        count += 1
                        if count >= per_genre or len(samples) >= self.sample_size:
                            break
                genre_sample_counts[g] = count
                if len(samples) >= self.sample_size:
                    break
            # If not enough, fill up from all remaining unseen
            if len(samples) < self.sample_size:
                all_unseen = set(range(1, self.item_count + 1)) - seen - set(samples)
                all_unseen = list(all_unseen)
                np.random.shuffle(all_unseen)
                for sid in all_unseen:
                    samples.append(sid)
                    if len(samples) >= self.sample_size:
                        break

            negative_samples[user] = samples[:self.sample_size]

        return negative_samples
