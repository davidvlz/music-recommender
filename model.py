"""
KNN model implementation for the Spotify Music Recommendation System.

This module defines the KNN model class used for finding similar songs
based on audio features.
"""

import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Import configuration
import config


class SpotifyKNNModel:
    """
    K-Nearest Neighbors model for finding similar songs based on audio features.
    """
    
    def __init__(self, n_neighbors=None, metric=None):
        """
        Initialize the KNN model.
        
        Args:
            n_neighbors (int, optional): Number of neighbors to use. Defaults to config value.
            metric (str, optional): Distance metric to use. Defaults to config value.
        """
        self.n_neighbors = n_neighbors or config.N_NEIGHBORS
        self.metric = metric or config.DISTANCE_METRIC
        self.model = None
        self.is_fitted = False
    
    def fit(self, feature_matrix):
        """
        Fit the KNN model to the feature matrix.
        
        Args:
            feature_matrix (numpy.ndarray): Matrix of scaled audio features
            
        Returns:
            self: The fitted model
        """
        # TODO: Implement model fitting
        print(f"Fitting KNN model with {self.n_neighbors} neighbors and {self.metric} metric...")
        
        # Initialize and fit the model
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because the song itself will be included
            metric=self.metric,
            algorithm='auto'
        )
        self.model.fit(feature_matrix)
        self.is_fitted = True
        
        print("KNN model fitted successfully.")
        return self
    
    def find_neighbors(self, feature_vector, n_neighbors=None):
        """
        Find the nearest neighbors to a given feature vector.
        
        Args:
            feature_vector (numpy.ndarray): Feature vector of a song
            n_neighbors (int, optional): Number of neighbors to return. Defaults to model's n_neighbors.
            
        Returns:
            tuple: (distances, indices) of the nearest neighbors
        """
        # TODO: Implement neighbor finding
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # Use the model's n_neighbors if not specified
        if n_neighbors is None:
            n_neighbors = self.n_neighbors + 1  # +1 because the song itself will be included
        
        # Reshape the feature vector if needed
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(
            feature_vector,
            n_neighbors=n_neighbors
        )
        
        return distances, indices
    
    def save(self, file_path=None):
        """
        Save the fitted model to disk.
        
        Args:
            file_path (str, optional): Path to save the model. Defaults to config value.
            
        Returns:
            bool: True if successful, False otherwise
        """
        # TODO: Implement model saving
        if not self.is_fitted:
            print("Warning: Saving an unfitted model.")
        
        file_path = file_path or config.MODEL_FILE
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load(cls, file_path=None):
        """
        Load a fitted model from disk.
        
        Args:
            file_path (str, optional): Path to load the model from. Defaults to config value.
            
        Returns:
            SpotifyKNNModel: The loaded model, or None if loading failed
        """
        # TODO: Implement model loading
        file_path = file_path or config.MODEL_FILE
        
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {file_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


def get_model(feature_matrix=None):
    """
    Get a KNN model, either by loading a saved model or creating and fitting a new one.
    
    Args:
        feature_matrix (numpy.ndarray, optional): Feature matrix to fit the model if needed
        
    Returns:
        SpotifyKNNModel: The loaded or newly fitted model, or None if failed
    """
    # TODO: Implement model getting function
    # Try to load an existing model
    model = SpotifyKNNModel.load()
    
    # If no model exists and feature_matrix is provided, create and fit a new one
    if model is None and feature_matrix is not None:
        print("No saved model found. Creating and fitting a new model...")
        model = SpotifyKNNModel()
        model.fit(feature_matrix)
        model.save()
    
    return model


if __name__ == "__main__":
    # Example usage
    from data_loader import get_data
    
    # Get the data
    processed_df, scaled_features, scaler = get_data()
    
    if processed_df is not None:
        # Get or create the model
        model = get_model(scaled_features)
        
        if model is not None:
            # Test the model with a random song
            random_idx = np.random.randint(0, len(processed_df))
            random_song = processed_df.iloc[random_idx]
            random_features = scaled_features[random_idx]
            
            print(f"\nFinding songs similar to: {random_song['name']} by {random_song['artists']}")
            
            # Find nearest neighbors
            distances, indices = model.find_neighbors(random_features)
            
            # Print the results
            print("\nNearest neighbors:")
            for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):  # Skip the first one (the song itself)
                similar_song = processed_df.iloc[idx]
                print(f"{i+1}. {similar_song['name']} by {similar_song['artists']} (Distance: {dist:.4f})")
