"""
Recommendation module for the Spotify Music Recommendation System.

This module handles generating song recommendations based on user input.
"""

import numpy as np

# Import project modules
import config
from data_loader import get_data
from model import get_model
from utils import find_closest_songs


def get_song_features(song_idx, processed_df, scaled_features):
    """
    Get the features for a specific song by index.
    
    Args:
        song_idx (int): Index of the song in the DataFrame
        processed_df (pandas.DataFrame): Preprocessed DataFrame
        scaled_features (numpy.ndarray): Scaled feature matrix
        
    Returns:
        tuple: (song_info, feature_vector)
            - song_info: Dictionary with song metadata
            - feature_vector: Numpy array of scaled features
    """
    # TODO: Implement getting song features
    # Get the song from the DataFrame
    song = processed_df.iloc[song_idx]
    
    # Extract the feature vector
    feature_vector = scaled_features[song_idx]
    
    # Create a dictionary with song metadata
    song_info = {
        'name': song['name'],
        'artists': song['artists'],
        'year': song['year'] if 'year' in song else None,
        'popularity': song['popularity'] if 'popularity' in song else None,
        'index': song_idx
    }
    
    return song_info, feature_vector


def find_similar_songs(feature_vector, model, processed_df, scaled_features, n_recommendations=None, exclude_indices=None):
    """
    Find songs similar to a given feature vector.
    
    Args:
        feature_vector (numpy.ndarray): Feature vector of a song
        model (SpotifyKNNModel): Trained KNN model
        processed_df (pandas.DataFrame): Preprocessed DataFrame
        scaled_features (numpy.ndarray): Scaled feature matrix
        n_recommendations (int, optional): Number of recommendations to return. Defaults to config value.
        exclude_indices (list, optional): List of indices to exclude from recommendations
        
    Returns:
        list: List of dictionaries with song information and similarity scores
    """
    # TODO: Implement finding similar songs
    # Use default value if not specified
    n_recommendations = n_recommendations or config.TOP_N_RECOMMENDATIONS
    exclude_indices = exclude_indices or []
    
    # Find nearest neighbors
    distances, indices = model.find_neighbors(
        feature_vector,
        n_neighbors=n_recommendations + len(exclude_indices) + 1  # +1 for the song itself
    )
    
    # Flatten the arrays
    distances = distances.flatten()
    indices = indices.flatten()
    
    # Create a list of recommendations
    recommendations = []
    
    # Skip the first result (the song itself) and any excluded indices
    for dist, idx in zip(distances[1:], indices[1:]):
        # Skip excluded indices
        if idx in exclude_indices:
            continue
        
        # Get the song information
        song = processed_df.iloc[idx]
        
        # Add to recommendations
        recommendations.append({
            'name': song['name'],
            'artists': song['artists'],
            'year': song['year'] if 'year' in song else None,
            'popularity': song['popularity'] if 'popularity' in song else None,
            'distance': dist,
            'index': idx
        })
        
        # Stop once we have enough recommendations
        if len(recommendations) >= n_recommendations:
            break
    
    return recommendations


def get_recommendations_for_song(song_name, artist=None, year=None, n_recommendations=None):
    """
    Get recommendations for a specific song.
    
    Args:
        song_name (str): Name of the song
        artist (str, optional): Artist name for better matching
        year (int, optional): Year of release for better matching
        n_recommendations (int, optional): Number of recommendations to return
        
    Returns:
        tuple: (input_song, recommendations)
            - input_song: Dictionary with input song information
            - recommendations: List of dictionaries with recommended song information
    """
    # TODO: Implement getting recommendations for a song
    # Get the data and model
    data_result = get_data()
    if data_result is None:
        print("Failed to get data.")
        return None, []
    
    processed_df, scaled_features, _ = data_result
    
    model = get_model()
    if model is None:
        print("Failed to get model.")
        return None, []
    
    # Find the song in the dataset
    matches = find_closest_songs(song_name, processed_df, artist=artist, year=year)
    
    if not matches:
        print(f"No matches found for '{song_name}'.")
        return None, []
    
    # Use the first match
    match = matches[0]
    song_idx = match['index']
    
    # Get the song features
    song_info, feature_vector = get_song_features(song_idx, processed_df, scaled_features)
    
    # Find similar songs
    recommendations = find_similar_songs(
        feature_vector,
        model,
        processed_df,
        scaled_features,
        n_recommendations=n_recommendations,
        exclude_indices=[song_idx]  # Exclude the input song
    )


def get_kmeans_recommendations(song_idx, processed_df, scaled_features, kmeans_model, n_recommendations=None):
    """
    Get recommendations for a song using a hybrid KMeans+similarity approach.
    """
    n_recommendations = n_recommendations or config.TOP_N_RECOMMENDATIONS

    # Predict the cluster for the input song
    input_vector = scaled_features[song_idx].reshape(1, -1)
    cluster_label = kmeans_model.predict(input_vector)[0]

    # Get all indices in the same cluster (excluding the input song)
    cluster_indices = kmeans_model.get_cluster_members(cluster_label, processed_df)
    cluster_indices = [idx for idx in cluster_indices if idx != song_idx]

    if len(cluster_indices) < 2:
        # Fall back to popular songs if cluster is too small
        popular_indices = get_popular_songs(processed_df, n=n_recommendations)
        return [popular_song for popular_song in popular_indices]

    # Get the feature vector for the input song
    input_features = scaled_features[song_idx].reshape(1, -1)

    # Calculate similarities to all songs in the cluster
    cluster_features = scaled_features[cluster_indices]
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(input_features, cluster_features)[0]

    # Sort cluster indices by similarity to input song
    similarity_sorted_indices = [cluster_indices[i] for i in similarities.argsort()[::-1]]

    # Build the recommendations list based on similarity within cluster
    recommendations = []
    for idx in similarity_sorted_indices[:n_recommendations]:
        song = processed_df.iloc[idx]
        recommendations.append({
            'name': song['name'],
            'artists': song['artists'],
            'year': song['year'] if 'year' in song else None,
            'popularity': song['popularity'] if 'popularity' in song else None,
            'similarity': similarities[cluster_indices.index(idx)],
            'index': idx
        })

    return recommendations


def format_recommendations(recommendations):
    """
    Format recommendations for display.
    """
    if not recommendations:
        return "No recommendations found."

    result = "Recommended songs:\n"

    for i, rec in enumerate(recommendations):
        # Format the recommendation
        song_info = f"{i + 1}. {rec['name']} by {rec['artists']}"

        # Add year if available
        if rec['year']:
            song_info += f" ({rec['year']})"

        # Add similarity score
        if 'distance' in rec:
            similarity = 1.0 / (1.0 + rec['distance'])  # Convert distance to similarity
            song_info += f" - Similarity: {similarity:.2f}"
        elif 'similarity' in rec:
            song_info += f" - Similarity: {rec['similarity']:.2f}"

        result += song_info + "\n"

    return result


def main():
    """
    Main function to test the recommendation functionality.
    """
    # TODO: Implement main function
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get song recommendations')
    parser.add_argument('--song', type=str, required=True, help='Song name')
    parser.add_argument('--artist', type=str, help='Artist name (optional)')
    parser.add_argument('--year', type=int, help='Year of release (optional)')
    parser.add_argument('--count', type=int, default=config.TOP_N_RECOMMENDATIONS, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Get recommendations
    song_info, recommendations = get_recommendations_for_song(
        args.song,
        artist=args.artist,
        year=args.year,
        n_recommendations=args.count
    )
    
    if song_info:
        print(f"\nInput song: {song_info['name']} by {song_info['artists']}")
        print("\n" + format_recommendations(recommendations))
    else:
        print("Failed to get recommendations.")


if __name__ == "__main__":
    main()
