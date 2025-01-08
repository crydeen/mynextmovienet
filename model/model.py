from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import argparse

class MyNextMovieNet(nn.Module):
  def __init__(self, num_users, num_movies, embedding_dim=32):
    super(MyNextMovieNet, self).__init__()
    self.user_embedding = nn.Embedding(num_users, embedding_dim)
    self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
    self.fc1 = nn.Linear(embedding_dim * 2, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.dropout = nn.Dropout(0.5)
    self.bn1 = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(64)
    self.output = nn.Linear(32, 1)

  def forward(self, user_ids, movie_ids):
    user_embedded = self.user_embedding(user_ids)
    movie_embedded = self.movie_embedding(movie_ids)
    x = torch.cat([user_embedded, movie_embedded], dim=1)
    x = torch.relu(self.bn1(self.fc1(x)))
    x = self.dropout(x)
    x = torch.relu(self.bn2(self.fc2(x)))
    x = self.dropout(x)
    x = torch.relu(self.fc3(x))
    return self.output(x)
  
  def get_movie_embedding(self, movie_idx):
    """
    Fetches the movie embedding for a given movie index.
    """
    # Ensure the input movie_idx is a tensor of appropriate type (Long)
    movie_idx = torch.tensor(movie_idx, dtype=torch.long)

    # Return the movie embedding for the specified index
    return self.movie_embedding(movie_idx)

def train():
  start = time.time()
  model_path = "model/deep_model_v2.pt"
  df_ratings = pd.read_csv("movielens/rating.csv", usecols=['userId', 'movieId', 'rating'])
  df_ratings['userIdx'] = df_ratings['userId'].astype('category').cat.codes
  df_ratings['movieIdx'] = df_ratings['movieId'].astype('category').cat.codes

  movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(df_ratings['movieId'].astype('category').cat.categories)}
  index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}

  user_id_to_index = {user_id: idx for idx, user_id in enumerate(df_ratings['userId'].astype('category').cat.categories)}
  index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}


  train_data, test_data = train_test_split(df_ratings, test_size=0.2, random_state=42)

  # Convert to TensorDataset for DataLoader
  train_dataset = TensorDataset(torch.tensor(train_data['userIdx'].values, dtype=torch.long),
                               torch.tensor(train_data['movieIdx'].values, dtype=torch.long),
                               torch.tensor(train_data['rating'].values, dtype=torch.float32))

  test_dataset = TensorDataset(torch.tensor(test_data['userIdx'].values, dtype=torch.long),
                              torch.tensor(test_data['movieIdx'].values, dtype=torch.long),
                              torch.tensor(test_data['rating'].values, dtype=torch.float32))

  # DataLoader for batching
  train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

  num_users = len(user_id_to_index)
  num_movies = len(movie_id_to_index)
  model = MyNextMovieNet(num_users, num_movies)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  end = time.time() - start
  print(f"Finished preprocessing, entering training loop: {end:.2f} seconds")
  epochs = 15
  lowest_validation_loss = 1000
  for epoch in range(epochs):
    model.train()
    running_loss = 0
    running_mae = 0
    for user_idx, movie_idx, rating in train_loader:
      optimizer.zero_grad()

      outputs = model(user_idx, movie_idx).squeeze(-1)
      loss = criterion(outputs, rating)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      running_mae += torch.mean(torch.abs(outputs - rating)).item()  # MAE for batch

    model.eval()
    val_loss = 0.0
    running_mae_val = 0.0
    with torch.no_grad():
      # test_outputs = model(X_test_tensor).squeeze()
      # val_loss = criterion(test_outputs, y_test_tensor)
      # val_mae = torch.mean(torch.abs(test_outputs - y_test_tensor))

      for user_idx, movie_idx, rating in test_loader:
            predictions = model(user_idx, movie_idx).squeeze(-1)
            val_loss += criterion(predictions, rating).item()
            running_mae_val += torch.mean(torch.abs(predictions - rating)).item()

    if val_loss < lowest_validation_loss:
      lowest_validation_loss = val_loss
      torch.save(model.state_dict(), model_path)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train MAE: {running_mae/len(train_loader):.2f}, Val Loss: {val_loss/len(test_loader):.4f}, Val MAE: {running_mae_val/len(test_loader):.2f}, Time: {(time.time()-start)/60:.4f} minutes")

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
  parser.add_argument("mode", help="Please choose train or predict", choices=["train","predict"])
  parser.add_argument("--prefs")
  args = parser.parse_args()
  if args.mode == "train":
     print("Train the Model")
     train()
  else:
    print("Predict with the model")
    if not args.prefs:
      print("Please input user movie prefs")
    else:
      pref_movie_ids = [int(x) for x in args.prefs.split(",")]
      predict(pref_movie_ids)
  