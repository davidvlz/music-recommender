# config.py

"""
Configuration settings for the Spotify Music Recommendation System.
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Data & Models ────────────────────────────────────────────────────────────

DATA_FILE               = os.path.join(DATA_DIR, "data.csv")
PROCESSED_DATA_FILE     = os.path.join(DATA_DIR, "processed_data.pkl")
SCALER_FILE             = os.path.join(DATA_DIR, "scaler.pkl")

MODEL_FILE              = os.path.join(MODELS_DIR, "knn_model.pkl")
KMEANS_MODEL_FILE       = os.path.join(MODELS_DIR, "kmeans_model.pkl")
DECISIONTREE_MODEL_FILE = os.path.join(MODELS_DIR, "decision_tree_model.pkl")

# ─── Hyperparameters ──────────────────────────────────────────────────────────

N_NEIGHBORS      = 5
DISTANCE_METRIC  = "euclidean"
N_CLUSTERS       = 25
POPULARITY_THRESHOLD = None  # None→use median at runtime

# ─── Features & Matching ──────────────────────────────────────────────────────

AUDIO_FEATURES        = [
    "danceability","energy","key","loudness","mode","speechiness",
    "acousticness","instrumentalness","liveness","valence","tempo"
]
FUZZY_MATCH_CUTOFF    = 0.6
MAX_MATCHES_TO_SHOW   = 5
TOP_N_RECOMMENDATIONS = 10

# ─── UI & Visualization ────────────────────────────────────────────────────────

MENU_PROMPT = """
Spotify Music Recommendation System
----------------------------------
1. Get song recommendations
2. Model management
3. Predict Hit?
4. Analysis
5. Exit
"""
VISUALIZATION_ENABLED = True
FIGURE_SIZE           = (11, 7)
