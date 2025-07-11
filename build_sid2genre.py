import numpy as np
import os

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

def parse_genres(genres_str):
    genres = genres_str.split('|')
    return [genre_name_to_id[g] for g in genres if g in genre_name_to_id]

def build_sid2genre(ml1m_movies_path, output_path):
    sid2genre = {}
    with open(ml1m_movies_path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) != 3:
                continue
            movie_id = int(parts[0])
            genres = parse_genres(parts[2])
            sid2genre[movie_id] = genres
    np.save(output_path, sid2genre)
    print(f"sid2genre saved to {output_path}. Total movies: {len(sid2genre)}")

if __name__ == "__main__":
    ml1m_movies_path = os.path.join('Data', 'ml-1m', 'movies.dat')
    output_path = os.path.join('Data', 'ml-1m', 'sid2genre.npy')
    build_sid2genre(ml1m_movies_path, output_path)