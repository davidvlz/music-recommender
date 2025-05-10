"""
Model training module for the Spotify Music Recommendation System.

This module handles training the KNN model on the preprocessed Spotify data.
"""

import os
import time

# Import project modules
import config
from data_loader import get_data
from model import SpotifyKNNModel


def train_model(force_retrain=False):
    """
    Train the KNN model on the preprocessed Spotify data.

    Args:
        force_retrain (bool, optional): If True, retrain the model even if a saved model exists.

    Returns:
        tuple: (model, processed_df, scaled_features) if successful, None otherwise
    """
    # ─── If a model already exists and we're not forcing retrain ─────────────────
    if os.path.exists(config.MODEL_FILE) and not force_retrain:
        print(f"Trained model already exists at {config.MODEL_FILE}.")
        print("Use force_retrain=True to retrain the model.")

        # Load the model
        from model import get_model
        model = get_model()

        # Load the data (now returns df, X, y, scaler)
        processed_df, scaled_features, _, scaler = get_data()

        return model, processed_df, scaled_features

    # ─── Otherwise, train a brand‑new model ────────────────────────────────────
    print("Training new model...")

    # Fetch processed data
    data_result = get_data()
    if data_result is None:
        print("Failed to get data for training.")
        return None

    # Unpack all four values, ignore y and scaler for now
    processed_df, scaled_features, _, scaler = data_result

    # Initialize and fit the KNN model
    start_time = time.time()
    model = SpotifyKNNModel(
        n_neighbors=config.N_NEIGHBORS,
        metric=config.DISTANCE_METRIC
    )
    model.fit(scaled_features)

    # Save the model to disk
    model.save()

    elapsed = time.time() - start_time
    print(f"Model training completed in {elapsed:.2f} seconds.")

    return model, processed_df, scaled_features


def evaluate_model(model, processed_df, scaled_features, n_samples=10):
    """
    Evaluate the trained model by testing it on random samples.
    """
    import numpy as np

    print(f"\nEvaluating model on {n_samples} random samples.")
    for i in range(n_samples):
        idx = np.random.randint(0, len(processed_df))
        song = processed_df.iloc[idx]
        print(f"\nSample {i + 1}: {song['name']} by {song['artists']}")
        distances, indices = model.find_neighbors(scaled_features[idx])
        print("Top 3 similar songs:")
        for rank, (dist, nbr_idx) in enumerate(zip(distances[0][1:4], indices[0][1:4]), start=1):
            similar = processed_df.iloc[nbr_idx]
            print(f"  {rank}. {similar['name']} by {similar['artists']} (Distance: {dist:.4f})")
    print("\nModel evaluation complete.")


def main():
    """
    CLI entry to train (and optionally evaluate) the model.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train the KNN recommendation model')
    parser.add_argument('--force', action='store_true', help='Retrain even if model exists')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--samples', type=int, default=5, help='Number of eval samples')
    args = parser.parse_args()

    result = train_model(force_retrain=args.force)
    if result is None:
        print("Model training failed.")
        return

    model, processed_df, scaled_features = result
    if args.evaluate:
        evaluate_model(model, processed_df, scaled_features, n_samples=args.samples)

    print("\nModel is ready for use.")


if __name__ == "__main__":
    main()
