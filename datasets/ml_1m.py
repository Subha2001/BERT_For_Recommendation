from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None, engine='python')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    
    # Added newly
    def load_ratings_with_genres(self):
        folder_path = self._get_rawdata_folder_path()
        ratings_path = folder_path.joinpath('ratings.dat')
        movies_path = folder_path.joinpath('movies.dat')
        ratings_df = pd.read_csv(ratings_path, sep='::', header=None, engine='python')
        ratings_df.columns = ['uid', 'sid', 'rating', 'timestamp']
        movies_df = pd.read_csv(movies_path, sep='::', header=None, engine='python')
        movies_df.columns = ['sid', 'title', 'genre']
        merged = pd.merge(ratings_df, movies_df[['sid', 'genre']], on='sid', how='left')
        return merged

    @property
    def sid2genre(self):
        import os
        movies_path = 'Data/ml-1m/movies.dat'
        # ...existing code...
        if not os.path.exists(movies_path):
            print(f"[ERROR] movies.dat does not exist at: {movies_path}")
        movies_df = pd.read_csv(movies_path, sep='::', header=None, engine='python', encoding='latin-1')
        movies_df.columns = ['sid', 'title', 'genre']
        # Map each sid to a genre id (you may want to encode genres as integers)
        unique_genres = {g: i+1 for i, g in enumerate(sorted(set(movies_df['genre'])))}
        sid2genre = {row['sid']: unique_genres.get(row['genre'], 0) for _, row in movies_df.iterrows()}
        return sid2genre


