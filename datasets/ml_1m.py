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
    
    ###########################################################################
    # Added newly
    ###########################################################################
    def load_ratings_with_genres(self):
        folder_path = self._get_rawdata_folder_path() # Get the path to the raw data folder
        ratings_path = folder_path.joinpath('ratings.dat')  # Path to ratings.dat
        movies_path = folder_path.joinpath('movies.dat') # Path to movies.dat
        ratings_df = pd.read_csv(ratings_path, sep='::', header=None, engine='python') # Read ratings data
        ratings_df.columns = ['uid', 'sid', 'rating', 'timestamp'] # Set column names for ratings data
        movies_df = pd.read_csv(movies_path, sep='::', header=None, engine='python') # Read movies data
        movies_df.columns = ['sid', 'title', 'genre'] # Set column names for movies data
        merged = pd.merge(ratings_df, movies_df[['sid', 'genre']], on='sid', how='left') # Merge ratings with genres
        return merged

    @property
    def sid2genre(self):
        import os
        movies_path = 'Data/ml-1m/movies.dat' # Path to movies.dat file
        if not os.path.exists(movies_path): # Check if the file exists
            print(f"[ERROR] movies.dat does not exist at: {movies_path}")
        movies_df = pd.read_csv(movies_path, sep='::', header=None, engine='python', encoding='latin-1') # Read movies data
        movies_df.columns = ['sid', 'title', 'genre'] # Set column names for movies data
        # Map each sid to a genre id (you may want to encode genres as integers)
        unique_genres = {g: i+1 for i, g in enumerate(sorted(set(movies_df['genre'])))} # Create a mapping of genres to unique integers
        sid2genre = {row['sid']: unique_genres.get(row['genre'], 0) for _, row in movies_df.iterrows()} # Create a mapping of sid to genre id
        return sid2genre
    ###########################################################################
    # End of newly added code
    ###########################################################################


