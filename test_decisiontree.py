"""
test_decisiontree.py

Quick‐and‐dirty script to verify Decision Tree predictions on a small sample.
"""

import sys
import os

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)

from data_loader import get_data
from decisiontree import SpotifyDecisionTreeModel
import pandas as pd

def main():
    # 1) Load data
    data = get_data()
    if data is None:
        print("❌  Cannot load data. Did you run data_loader successfully?")
        return
    df, X, y, _ = data

    # 2) Load the trained Decision Tree
    model = SpotifyDecisionTreeModel.load()
    if model is None:
        print("❌  No saved Decision Tree model found. Run `python decisiontree.py` first with retrain=True.")
        return

    # 3) Predict on the first 10 samples
    X_sample = X[:10]
    preds     = model.predict(X_sample)
    probs     = model.model.predict_proba(X_sample)[:, 1]  # probability of “hit”

    # 4) Build a DataFrame for display
    #    Adjust these column names if your CSV uses different field names
    display_df = df.loc[:9, ["name", "artists", "popularity"]].copy()
    display_df["predicted_hit"]   = preds
    display_df["hit_probability"] = probs.round(3)

    # 5) Print it out
    print("\nSample Decision Tree Predictions:\n")
    print(display_df.to_string(index=False))

if __name__ == "__main__":
    main()
