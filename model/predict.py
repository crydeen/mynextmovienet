import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from model import MyNextMovieNet


def recommend_movies_for_new_user(model, preferred_movie_ids, movie_id_to_index, index_to_movie_id, top_n=5):
    """
    Recommends movies for a new user based on their preferred movies.

    Parameters:
    - model: The trained recommendation model.
    - preferred_movie_ids: List of movie IDs that the new user prefers.
    - movie_id_to_index: Dictionary mapping movie IDs to movie indices.
    - index_to_movie_id: Dictionary mapping movie indices to movie IDs.
    - top_n: Number of movie recommendations to return.

    Returns:
    - List of recommended movie IDs.
    """
    
    # Convert preferred movie IDs to indices
    preferred_movie_indices = [movie_id_to_index[movie_id] for movie_id in preferred_movie_ids]

    # Fetch embeddings for each preferred movie and average them to represent the user's taste
    preferred_embeddings = torch.stack([model.get_movie_embedding(idx) for idx in preferred_movie_indices])
    user_preference_embedding = preferred_embeddings.mean(dim=0)
    
    # Compute similarity scores with all movies in the dataset
    all_movie_embeddings = model.movie_embedding.weight.data  # Get all movie embeddings
    similarity_scores = torch.matmul(all_movie_embeddings, user_preference_embedding)

    # Get the top N movies with the highest similarity scores
    top_movie_indices = torch.topk(similarity_scores, top_n*2).indices
    recommended_movie_ids = [index_to_movie_id[idx.item()] for idx in top_movie_indices]

    for movie_id in recommended_movie_ids:
        if movie_id in preferred_movie_ids:
            recommended_movie_ids.remove(movie_id)

    return recommended_movie_ids

def predict(preferred_movie_ids):
    df_ratings = pd.read_csv("movielens/rating.csv", usecols=['userId', 'movieId', 'rating'])
    df_ratings['userIdx'] = df_ratings['userId'].astype('category').cat.codes
    df_ratings['movieIdx'] = df_ratings['movieId'].astype('category').cat.codes
    num_users = df_ratings['userId'].nunique()
    num_movies = df_ratings['movieId'].nunique()

    # movie_ids = sorted(df_ratings.movieId.unique())
    movie_ids = df_ratings['movieId'].unique()
    # movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    # index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}
    # index_to_movie_id = dict(enumerate(df_ratings['movieId']))
    # movie_id_to_index = {v: k for k, v in index_to_movie_id.items()}

    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(df_ratings['movieId'].astype('category').cat.categories)}
    index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}
    
    model = MyNextMovieNet(num_users, num_movies)
    model.load_state_dict(torch.load('model/deep_model_v2.pt', weights_only=True))
    model.eval()
    # new_user_embedding = get_new_user_embedding(preferred_movie_ids, model)
    # recommended_movies = recommend_for_new_user(new_user_embedding, model, top_k=10)
    # print(f"Recommended movie IDs: {recommended_movies}")
    recommended_movie_ids = recommend_movies_for_new_user(model, preferred_movie_ids, movie_id_to_index, index_to_movie_id)
    print("Recommended movie IDs:", recommended_movie_ids)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefs")
    args = parser.parse_args()
    pref_movie_ids = [int(x) for x in args.prefs.split(",")]
    predict(pref_movie_ids)