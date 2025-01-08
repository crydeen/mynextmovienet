from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary
import numpy as np
import pandas as pd
import time
import argparse
import os
import glob
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
df_movie_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "ml32_movie_info.csv"))

# Reform Ratings Dataframe
# df_list = []
# csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "min_rating_[1-5].csv"))
# for file in sorted(csv_files):
#     df = pd.read_csv(file)
#     df_list.append(df)
# df_ratings = pd.concat(df_list, ignore_index=True)
df_ratings = pd.read_csv(os.path.join(os.path.dirname(__file__), "ml32_rating.csv"), usecols=['userId', 'movieId', 'rating'])

class MyNextMovieNet(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=32):
        super(MyNextMovieNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.output = nn.Linear(32, 1)

    def forward(self, user_ids, movie_ids, genre_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        genre_embedded = self.genre_embedding(genre_ids).mean(dim=1)
        x = torch.cat([user_embedded, movie_embedded, genre_embedded], dim=1)
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
    
    def get_genre_embedding(self, genre_tensor):
        mask = mask_trailing_zeros(genre_tensor)
        # Get the genre embeddings for the padded genre tensor
        genre_embeds = self.genre_embedding(genre_tensor)  # Shape: [batch_size, max_genre_length, embedding_dim]
        
        # Apply mask and calculate the average, summing only valid genre embeddings
        masked_genre_embeds = genre_embeds[mask]
        avg_genre_embeds = masked_genre_embeds.mean(dim=0)
        return avg_genre_embeds

class MyNextMovieNetOG(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super(MyNextMovieNetOG, self).__init__()
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
    
    def get_genre_embedding(self, genre_tensor):
        mask = mask_trailing_zeros(genre_tensor)
        # Get the genre embeddings for the padded genre tensor
        genre_embeds = self.genre_embedding(genre_tensor)  # Shape: [batch_size, max_genre_length, embedding_dim]
        
        # Apply mask and calculate the average, summing only valid genre embeddings
        masked_genre_embeds = genre_embeds[mask]
        avg_genre_embeds = masked_genre_embeds.mean(dim=0)
        return avg_genre_embeds

def mask_trailing_zeros(tensor, pad_value=0):
    """
    Creates a mask that excludes only trailing `pad_value` in the tensor.
    Args:
        tensor (torch.Tensor): Input tensor of shape [seq_len].
        pad_value (int, optional): Value to treat as padding. Default is 0.
    Returns:
        torch.Tensor: Mask tensor of shape [seq_len], where valid values are True.
    """
    # Reverse the tensor
    reversed_tensor = torch.flip(tensor, dims=[0])
    
    # Create a mask that marks valid (non-pad_value) entries
    valid_mask = reversed_tensor != pad_value
    
    # Use cumulative sum to propagate validity backwards
    valid_cumsum = valid_mask.cumsum(dim=0)
    
    # Mask where valid_cumsum > 0, indicating valid elements
    reversed_final_mask = valid_cumsum > 0
    
    # Reverse the mask back to match the original order
    final_mask = torch.flip(reversed_final_mask, dims=[0])
    return final_mask

def process_genre(genre_string, genre_to_id):
    genre_ids = [genre_to_id[genre] for genre in genre_string.split('|')]
    return torch.tensor(genre_ids, dtype=torch.long)

def compute_precision(true_ratings, predicted_ratings, k=10):
    """
    Compute precision and recall at k for the predicted ratings.

    Args:
        true_ratings (list): List of true ratings.
        predicted_ratings (list): List of predicted ratings.
        k (int): Top-k recommendations to evaluate.

    Returns:
        precision (float): Precision at k.
    """

    # Sort by predicted ratings and keep top-k
    sorted_indices = torch.argsort(torch.tensor(predicted_ratings), descending=True)[:k]

    # Identify relevant items (e.g., actual ratings >= 4.0)
    relevant_items = torch.tensor(true_ratings) >= 4.0
    
    recommended_items = torch.zeros_like(relevant_items, dtype=torch.bool)
    recommended_items[sorted_indices] = True

    # Calculate precision 
    true_positive = (relevant_items & recommended_items).sum().item()
    precision = true_positive / recommended_items.sum().item()
    return precision

def calculate_metrics(predicted_ids, ground_truth_ids, k=10):
    true_positives = 0
    false_negatives = 0
    
    for pred, truth in zip(predicted_ids, ground_truth_ids):
        pred_set = set(pred[:k])  # Top-k predictions
        truth_set = set(truth)

        tp = len(pred_set & truth_set)  # Intersection of predicted and ground truth
        fn = len(truth_set - pred_set)  # Ground truth not predicted

        true_positives += tp
        false_negatives += fn

    precision = true_positives / (k * len(predicted_ids)) if predicted_ids else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def plot_metrics(
    train_losses, val_losses, train_precisions, val_precisions, file_path=os.path.join(os.path.dirname(__file__),"metrics_plot.png")
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 10))

    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Precision and Recall plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_precisions, label="Train Precision")
    plt.plot(epochs, val_precisions, label="Validation Precision")
    plt.title("Precision Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(file_path, format="png", dpi=300)
    print(f"Plot saved to {file_path}")

    # Display the plot
    plt.show()

def train(genre_bool):
    start = time.time()
    if genre_bool:
        print("Including genre information in training")
    else:
        print("Omitting genre information in training")
    global df_ratings

    # Adding genre information, skip if --no-genre
    if genre_bool:
        df_genres = df_movie_info[['ml_movieId','ml_genres']]
        df_genres['movieId'] = df_movie_info['ml_movieId']
        df_genres.drop("ml_movieId", axis=1, inplace=True)
        df_ratings = pd.merge(df_ratings, df_genres, on="movieId", how="left")
        unique_genres = sorted(set(g for genres in df_movie_info['ml_genres'] for g in genres.split('|')))
        genre_to_id = {genre: idx for idx, genre in enumerate(unique_genres)}
        genre_tensor = [process_genre(genres, genre_to_id) for genres in df_ratings['ml_genres']]
        padded_genre_tensor = pad_sequence(genre_tensor, batch_first=True, padding_value=0)

        print(f'genre processed: {(time.time()-start)/60:.2f} minutes')
        num_genres = len(unique_genres)

        print("Number of genres:",num_genres)

    df_ratings['userIdx'] = df_ratings['userId'].astype('category').cat.codes
    df_ratings['movieIdx'] = df_ratings['movieId'].astype('category').cat.codes

    user_tensor = torch.tensor(df_ratings['userIdx'].values, dtype=torch.long)
    movie_tensor = torch.tensor(df_ratings['movieIdx'].values, dtype=torch.long)
    rating_tensor = torch.tensor(df_ratings['rating'].values, dtype=torch.float32)

    print(f'tensors processed: {(time.time()-start)/60:.2f} minutes')

    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(df_ratings['movieId'].astype('category').cat.categories)}
    index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(df_ratings['userId'].astype('category').cat.categories)}
    index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}

    if genre_bool:
        model_path = os.path.join(os.path.dirname(__file__), 'deep_model_ml32_genre_v3.pt')
        train_user_ids, test_user_ids, train_movie_ids, test_movie_ids, train_genres, test_genres, train_ratings, test_ratings = train_test_split(
            user_tensor, movie_tensor, padded_genre_tensor, rating_tensor, test_size=0.2, random_state=42
        )
        print(f'split processed: {(time.time()-start)/60:.2f} minutes')

        train_dataset = TensorDataset(
            train_user_ids, 
            train_movie_ids, 
            train_genres, 
            train_ratings
        )

        print(f'train processed: {(time.time()-start)/60:.2f} minutes')

        test_dataset = TensorDataset(
            test_user_ids, 
            test_movie_ids, 
            test_genres, 
            test_ratings
        )

        print(f'test processed: {(time.time()-start)/60:.2f} minutes')
    else: 
        model_path = os.path.join(os.path.dirname(__file__), 'deep_model_ml32_v2.pt')
        train_user_ids, test_user_ids, train_movie_ids, test_movie_ids, train_ratings, test_ratings = train_test_split(
            user_tensor, movie_tensor, rating_tensor, test_size=0.2, random_state=42
        )
        print(f'split processed: {(time.time()-start)/60:.2f} minutes')

        train_dataset = TensorDataset(
            train_user_ids, 
            train_movie_ids, 
            train_ratings
        )

        print(f'train processed: {(time.time()-start)/60:.2f} minutes')

        test_dataset = TensorDataset(
            test_user_ids, 
            test_movie_ids,
            test_ratings
        )

        print(f'test processed: {(time.time()-start)/60:.2f} minutes')

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    num_users = len(user_id_to_index)
    num_movies = len(movie_id_to_index)
    
    print("Number of users:",num_users)
    print("Number of movies:",num_movies)
    
    if genre_bool:
        model = MyNextMovieNet(num_users, num_movies, num_genres)
    else:
        model = MyNextMovieNetOG(num_users, num_movies)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    end = time.time() - start
    print(f"Finished preprocessing, entering training loop: {end:.2f} seconds")
    epochs = 20
    movies_recommended = 20
    lowest_validation_loss = 1000

    # Metrics storage
    train_losses = []
    val_losses = []
    train_precisions = []
    val_precisions = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        train_ratings_list = []
        train_predictions_list = []
        if genre_bool:
            for user_idx, movie_idx, genre_vecs, rating in train_loader:
                optimizer.zero_grad()

                outputs = model(user_idx, movie_idx, genre_vecs).squeeze(-1)
                loss = criterion(outputs, rating)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_mae += torch.mean(torch.abs(outputs - rating)).item()  # MAE for batch

                train_ratings_list.extend(rating.tolist())
                train_predictions_list.extend(outputs.tolist())
        else:
            for user_idx, movie_idx, rating in train_loader:
                optimizer.zero_grad()

                outputs = model(user_idx, movie_idx).squeeze(-1)
                loss = criterion(outputs, rating)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_mae += torch.mean(torch.abs(outputs - rating)).item()  # MAE for batch

                train_ratings_list.extend(rating.tolist())
                train_predictions_list.extend(outputs.tolist())
        # Compute Precision and Recall for train set
        train_precision = compute_precision(
            train_ratings_list, train_predictions_list, k=movies_recommended
        )

        model.eval()
        val_loss = 0.0
        running_mae_val = 0.0
        val_ratings = []
        val_predictions = []
        with torch.no_grad():
            if genre_bool:
                for user_idx, movie_idx, genre_vecs, rating in test_loader:
                    predictions = model(user_idx, movie_idx, genre_vecs).squeeze(-1)
                    val_loss += criterion(predictions, rating).item()
                    running_mae_val += torch.mean(torch.abs(predictions - rating)).item()

                    val_ratings.extend(rating.tolist())
                    val_predictions.extend(predictions.tolist())
            else:
                for user_idx, movie_idx, rating in test_loader:
                    predictions = model(user_idx, movie_idx).squeeze(-1)
                    val_loss += criterion(predictions, rating).item()
                    running_mae_val += torch.mean(torch.abs(predictions - rating)).item()

                    val_ratings.extend(rating.tolist())
                    val_predictions.extend(predictions.tolist())
        val_precision = compute_precision(
            val_ratings, val_predictions, k=movies_recommended
        )

        # Metric calculations
        train_loss = running_loss/len(train_loader)
        train_MAE = running_mae/len(train_loader)
        validation_loss = val_loss/len(test_loader)
        validation_MAE = running_mae_val/len(test_loader)

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            torch.save(model.state_dict(), model_path)
        

        # Metrics Storage
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {train_loss:.4f}, "
              f"Train MAE: {train_MAE:.4f}, "
              f"Val Loss: {validation_loss:.4f}, "
              f"Val MAE: {validation_MAE:.4f}, "
              f"Train Precision: {train_precision:.4f}, "
              f"Val Precision: {val_precision:.4f}, "
              f"Time: {(time.time()-start)/60:.4f} minutes"
        )

        # Plotting metrics
    plot_metrics(
        train_losses, val_losses, train_precisions, val_precisions
    )

def recommend_movies_for_new_user(model, preferred_movie_ids, movie_id_to_index, index_to_movie_id, genre_bool, top_n=5):
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
    # Genre Embeddings
    
    
    # genre_tensors = []
    # for movie_id in preferred_movie_ids:
    #     genre_string = genre_dict.get(movie_id, "")  # Retrieve genre string for the movie
    #     genre_tensor = process_genre(genre_string)  # Convert genre string to tensor (using a function like `process_genre`)
    #     genre_tensors.append(genre_tensor)
    
    if genre_bool:
        unique_genres = sorted(set(g for genres in df_movie_info['ml_genres'] for g in genres.split('|')))
        genre_to_id = {genre: idx for idx, genre in enumerate(unique_genres)}

        genre_tensor = [process_genre(genres, genre_to_id) for genres in df_movie_info.loc[df_movie_info['ml_movieId'].isin(preferred_movie_ids)]['ml_genres']]
        # print(genre_tensor)
        padded_genre_tensor = pad_sequence(genre_tensor, batch_first=True, padding_value=0)
    
    # Convert preferred movie IDs to indices
    preferred_movie_indices = [movie_id_to_index[movie_id] for movie_id in preferred_movie_ids]

    # Fetch embeddings for each preferred movie and average them to represent the user's movie taste
    preferred_embeddings = torch.stack([model.get_movie_embedding(idx) for idx in preferred_movie_indices])
    avg_movie_embedding = preferred_embeddings.mean(dim=0)

    if genre_bool:
        # print(padded_genre_tensor)
        # Fetch embeddings for each preferred movie and average them to represent the user's genre taste
        genre_embeddings = torch.stack([model.get_genre_embedding(gt) for gt in padded_genre_tensor])
        # print(genre_embeddings)
        avg_genre_embedding = genre_embeddings.mean(dim=0)

        # Combine average embeddings to form a new user preference vector
        # user_preference_vector = torch.cat([avg_movie_embedding, avg_genre_embedding], dim=0)
        user_preference_embedding = (avg_movie_embedding + avg_genre_embedding) / 2
    else:
        user_preference_embedding = avg_movie_embedding
    
    # Compute similarity scores with all movies in the dataset
    all_movie_embeddings = model.movie_embedding.weight.data  # Get all movie embeddings
    # all_genre_embeddings = model.genre_embedding.weight.data

    # all_combined_embeddings = torch.cat([all_movie_embeddings, all_genre_embeddings], dim=1)

    # Calculate Similarity
    similarity_scores = F.cosine_similarity(all_movie_embeddings, user_preference_embedding)

    # Get the top N movies with the highest similarity scores
    top_movie_indices = torch.topk(similarity_scores, top_n+len(preferred_movie_ids)).indices
    recommended_movie_ids = [index_to_movie_id[idx.item()] for idx in top_movie_indices]

    remove_indexes = []

    for index, movie_id in enumerate(recommended_movie_ids):
        if movie_id in preferred_movie_ids:
            remove_indexes.append(index)

    for index in sorted(remove_indexes, reverse=True):
       del recommended_movie_ids[index]

    return recommended_movie_ids[:top_n]

def predict(preferred_movie_ids, genre_bool):
    # df_ratings = pd.read_csv(os.path.join(os.path.dirname(__file__), "rating.csv"), usecols=['userId', 'movieId', 'rating'])
    global df_ratings
    df_ratings['userIdx'] = df_ratings['userId'].astype('category').cat.codes
    df_ratings['movieIdx'] = df_ratings['movieId'].astype('category').cat.codes

    num_users = df_ratings['userId'].nunique()
    num_movies = df_ratings['movieId'].nunique()

    print("Number of users:",num_users)
    print("Number of movies:",num_movies)

    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(df_ratings['movieId'].astype('category').cat.categories)}
    index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}

    if genre_bool:
        df_genres = df_movie_info[['ml_genres']]
        df_genres['ml_genres'] = df_genres['ml_genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(df_genres['ml_genres'])
        num_genres = len(genre_encoded[0])
        print("Number of genres:",num_genres)
        model = MyNextMovieNet(num_users, num_movies, num_genres)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'deep_model_ml32_genre_v3.pt'), weights_only=True))
    else:
        model = MyNextMovieNetOG(num_users, num_movies)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'deep_model_ml32_v2.pt'), weights_only=True))
    
    model.eval()

    recommended_movie_ids = recommend_movies_for_new_user(model, preferred_movie_ids, movie_id_to_index, index_to_movie_id, genre_bool)
    print("\nPreference: \n Movie IDs:",preferred_movie_ids)
    pref_movie_names = df_movie_info.loc[df_movie_info['ml_movieId'].isin(preferred_movie_ids)]['title'].to_list()
    pref_genres = df_movie_info.loc[df_movie_info['ml_movieId'].isin(preferred_movie_ids)]['ml_genres'].to_list()
    print("Movie Names:",pref_movie_names)
    print("Genres:",pref_genres)
    print("\nRecommendation: \nMovie IDs:", recommended_movie_ids)
    rec_movie_names = df_movie_info.loc[df_movie_info['ml_movieId'].isin(recommended_movie_ids)]['title'].to_list()
    rec_genres = df_movie_info.loc[df_movie_info['ml_movieId'].isin(recommended_movie_ids)]['ml_genres'].to_list()
    print("Movie names:",rec_movie_names)
    print("Genres:",rec_genres)

def load_model(genre_bool):
    
    df_ratings['userIdx'] = df_ratings['userId'].astype('category').cat.codes
    df_ratings['movieIdx'] = df_ratings['movieId'].astype('category').cat.codes
    num_users = df_ratings['userId'].nunique()
    num_movies = df_ratings['movieId'].nunique()

    # Genres
    if genre_bool:
        df_genres = df_movie_info[['ml_genres']]
        df_genres['ml_genres'] = df_genres['ml_genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(df_genres['ml_genres'])
        num_genres = len(genre_encoded[0])

    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(df_ratings['movieId'].astype('category').cat.categories)}
    index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}
    if genre_bool:
        model = MyNextMovieNet(num_users, num_movies, num_genres)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'deep_model_ml32_genre_v3.pt'), weights_only=True))
    else:
        model = MyNextMovieNetOG(num_users, num_movies)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'deep_model_ml32_v2.pt'), weights_only=True))
    model.eval()

    return model, movie_id_to_index, index_to_movie_id

def movie_info_lookup(movie_id):
    movie_info = df_movie_info.loc[df_movie_info['ml_movieId'] == movie_id]
    info_dict = {}
    for i, info in enumerate(movie_info):
        info_dict[info] = movie_info.values[0][i]
    return info_dict

def get_popular_movies():
    popular_movies = df_movie_info.sort_values(by='ml_rating_count', ascending=False)
    return popular_movies.to_dict('records')

def model_summary():
    df_ratings['userIdx'] = df_ratings['userId'].astype('category').cat.codes
    df_ratings['movieIdx'] = df_ratings['movieId'].astype('category').cat.codes
    num_users = df_ratings['userId'].nunique()
    num_movies = df_ratings['movieId'].nunique()

    # Genres
    if genre_bool:
        df_genres = df_movie_info[['ml_genres']]
        df_genres['ml_genres'] = df_genres['ml_genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(df_genres['ml_genres'])
        num_genres = len(genre_encoded[0])

    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(df_ratings['movieId'].astype('category').cat.categories)}
    index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}
    batch_size = 4096
    if genre_bool:
        model = MyNextMovieNet(num_users, num_movies, num_genres)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'deep_model_ml32_genre_v3.pt'), weights_only=True))
        summary(
            model, 
            input_data=(torch.randint(0, num_users, (batch_size,)), torch.randint(0, num_movies, (batch_size,)),  torch.randint(0, num_genres, (batch_size, 5))), 
            col_names=["input_size", "output_size", "num_params"])
    else:
        model = MyNextMovieNetOG(num_users, num_movies)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'deep_model_ml32_v2.pt'), weights_only=True))
        summary(
            model, 
            input_data=(torch.randint(0, num_users, (batch_size,)), torch.randint(0, num_movies, (batch_size,))), 
            col_names=["input_size", "output_size", "num_params"])
    model.eval()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Please choose train or predict", choices=["train","predict","summary"])
    parser.add_argument("--prefs")
    parser.add_argument("--genre", action='store_true')
    parser.add_argument('--no-genre', dest='genre', action='store_false')
    args = parser.parse_args()
    genre_bool = args.genre
    print(genre_bool)
    if args.mode == "train":
        train(genre_bool)
    elif args.mode == "summary":
        model_summary()
    else:
        if not args.prefs:
            print("Please input user movie prefs")
        else:
            pref_movie_ids = [int(x) for x in args.prefs.split(",")]
            predict(pref_movie_ids, genre_bool)