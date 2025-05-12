"""
KMeans model implementation for the Spotify Music Recommendation System.

This module defines the KMeans model class used for clustering songs
based on audio features.
"""

import os
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Import configuration
import config

class SpotifyKMeansModel:
    """
    KMeans model for clustering songs based on audio features.
    """

    def __init__(self, n_clusters=None, random_state=42):
        """
        Initialize the KMeans model.

        Args:
            n_clusters (int, optional): Number of clusters to use. Defaults to config value.
            random_state (int, optional): Random state for reproducibility.
        """
        self.n_clusters = n_clusters or config.N_CLUSTERS
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def fit(self, feature_matrix):
        """
        Fit the KMeans model to the feature matrix.

        Args:
            feature_matrix (numpy.ndarray): Matrix of scaled audio features

        Returns:
            self: The fitted model
        """
        print(f"Fitting KMeans model with {self.n_clusters} clusters...")
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(feature_matrix)
        self.is_fitted = True
        print("KMeans model fitted successfully.")
        return self

    def predict(self, feature_matrix):
        """
        Predict the cluster for given feature(s).

        Args:
            feature_matrix (numpy.ndarray): Feature matrix or vector

        Returns:
            np.ndarray: Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.model.predict(feature_matrix)

    def get_cluster_members(self, cluster_label, processed_df):
        """
        Get indices of all songs in a given cluster.

        Args:
            cluster_label (int): Cluster label
            processed_df (pandas.DataFrame): Preprocessed DataFrame

        Returns:
            list: List of indices in the cluster
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model has not been fitted yet.")


        return [i for i, label in enumerate(self.model.labels_) if label == cluster_label]

    def analyze_clusters(self, processed_df):
        """
        Analyze the distribution of songs across clusters.

        Args:
            processed_df (pandas.DataFrame): Preprocessed DataFrame

        Returns:
            dict: Dictionary with cluster statistics
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model has not been fitted yet.")

        # Get cluster counts
        unique_labels, counts = np.unique(self.model.labels_, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))

        # Calculate statistics
        stats = {
            'total_clusters': len(unique_labels),
            'empty_clusters': self.n_clusters - len(unique_labels),
            'min_cluster_size': min(counts) if counts.size > 0 else 0,
            'max_cluster_size': max(counts) if counts.size > 0 else 0,
            'avg_cluster_size': np.mean(counts) if counts.size > 0 else 0,
            'median_cluster_size': np.median(counts) if counts.size > 0 else 0,
            'cluster_sizes': cluster_sizes
        }

        # Find potential outlier clusters (very small or very large)
        mean = stats['avg_cluster_size']
        std = np.std(counts) if counts.size > 0 else 0
        stats['small_clusters'] = [label for label, count in cluster_sizes.items() if count < mean - std]
        stats['large_clusters'] = [label for label, count in cluster_sizes.items() if count > mean + std]

        # Print a summary
        print("\n----- KMeans Cluster Analysis -----")
        print(f"Total number of clusters: {stats['total_clusters']} (configured: {self.n_clusters})")
        print(f"Empty clusters: {stats['empty_clusters']}")
        print(f"Cluster size range: {stats['min_cluster_size']} to {stats['max_cluster_size']} songs")
        print(f"Average cluster size: {stats['avg_cluster_size']:.2f} songs")
        print(f"Median cluster size: {stats['median_cluster_size']} songs")

        print("\nPotentially problematic clusters:")
        print(f"  Very small clusters (< mean-std): {stats['small_clusters']}")
        print(f"  Very large clusters (> mean+std): {stats['large_clusters']}")
        print("-------------------------------------\n")

        return stats

    def save(self, file_path=None):
        """
        Save the fitted model to disk.

        Args:
            file_path (str, optional): Path to save the model. Defaults to config value.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_fitted:
            print("Warning: Saving an unfitted model.")

        file_path = file_path or config.KMEANS_MODEL_FILE

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"KMeans model saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving KMeans model: {e}")
            return False

    @classmethod
    def load(cls, file_path=None):
        """
        Load a fitted model from disk.

        Args:
            file_path (str, optional): Path to load the model from. Defaults to config value.

        Returns:
            SpotifyKMeansModel: The loaded model, or None if loading failed
        """
        file_path = file_path or config.KMEANS_MODEL_FILE

        if not os.path.exists(file_path):
            print(f"KMeans model file not found: {file_path}")
            return None

        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"KMeans model loaded from {file_path}")
            return model
        except Exception as e:
            print(f"Error loading KMeans model: {e}")
            return None

def get_kmeans_model(feature_matrix=None):
    """
    Get a KMeans model, either by loading a saved model or creating and fitting a new one.

    Args:
        feature_matrix (numpy.ndarray, optional): Feature matrix to fit the model if needed

    Returns:
        SpotifyKMeansModel: The loaded or newly fitted model, or None if failed
    """
    model = SpotifyKMeansModel.load()
    if model is None and feature_matrix is not None:
        print("No saved KMeans model found. Creating and fitting a new model...")
        model = SpotifyKMeansModel()
        model.fit(feature_matrix)
        model.save()
    return model

if __name__ == "__main__":
    from data_loader import get_data

    processed_df, scaled_features, scaler = get_data()
    if processed_df is not None:
        model = get_kmeans_model(scaled_features)
        if model is not None:
            # Print cluster sizes
            unique, counts = np.unique(model.model.labels_, return_counts=True)
            print("Cluster sizes:")
            for label, count in zip(unique, counts):
                print(f"Cluster {label}: {count} songs")
