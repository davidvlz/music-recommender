"""
Data loading and preprocessing module for the Spotify Music Recommendation System.

This module handles downloading the Spotify dataset, loading it into memory,
and preprocessing it for use in the recommendation system.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import kagglehub

# Import configuration
import config

def download_data():
    """
    (Optional) If you’re pulling from Kaggle, use kagglehub to download raw CSVs.
    """
    # kagglehub.download("spotify/your-dataset", to=config.DATA_DIR)
    pass

def load_data():
    """
    Load the raw CSV into a DataFrame.
    """
    if not os.path.exists(config.DATA_FILE):
        print(f"Raw data file not found at {config.DATA_FILE}")
        return None
    return pd.read_csv(config.DATA_FILE)

def preprocess_data(df: pd.DataFrame):
    """
    Do any cleaning, null‑handling, feature engineering here,
    then standardize AUDIO_FEATURES.
    Returns (processed_df, X_scaled, scaler).
    """
    # Example: drop rows with missing features
    df = df.dropna(subset=config.AUDIO_FEATURES + ["popularity"])
    X = df[config.AUDIO_FEATURES].values

    # Fit or load scaler
    if os.path.exists(config.SCALER_FILE):
        with open(config.SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler().fit(X)
        with open(config.SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

    X_scaled = scaler.transform(X)
    return df, X_scaled, scaler

def save_processed_data(df, X_scaled, scaler):
    """
    Persist the cleaned DataFrame and scaler for faster reloads.
    """
    with open(config.PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(df, f)
    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

def load_processed_data():
    """
    Load previously saved DataFrame and scaler.
    """
    with open(config.PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
    with open(config.SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(df[config.AUDIO_FEATURES].values)
    return df, X, scaler

def get_data():
    """
    Main entry point. Returns:
      df : DataFrame (includes original columns + “popularity”)
      X  : np.ndarray of shape (n_samples, n_features)
      y  : np.ndarray of 0/1 labels (non‑hit/hit)
      scaler : the fitted StandardScaler
    """
    # 1) Load (or download) raw data
    if not os.path.exists(config.DATA_FILE):
        download_data()
    # 2) Try loading preprocessed
    if os.path.exists(config.PROCESSED_DATA_FILE) and os.path.exists(config.SCALER_FILE):
        df, X, scaler = load_processed_data()
    else:
        raw = load_data()
        if raw is None:
            return None
        df, X, scaler = preprocess_data(raw)
        save_processed_data(df, X, scaler)

    # 3) Build binary label y based on popularity threshold
    #    Compute median if threshold unset
    threshold = config.POPULARITY_THRESHOLD
    if threshold is None:
        threshold = df["popularity"].median()
    y = (df["popularity"] >= threshold).astype(int).values

    return df, X, y, scaler

if __name__ == "__main__":
    df, X, y, scaler = get_data()
    if df is not None:
        print("✅ Data loaded successfully!")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Hit rate: {y.mean():.3f}")
    else:
        print("❌ Failed to load data.")
