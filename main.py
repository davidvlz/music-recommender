# main.py

"""
Main entry point for the Spotify Music Recommendation System CLI.
"""

import os
os.environ['TERM'] = 'xterm-color'
import sys
import config
import numpy as np
from sklearn.metrics import silhouette_score
from data_loader import get_data
from train import train_model
from model import get_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from kmeans_model import get_kmeans_model
from recommend import (
    find_similar_songs,
    get_kmeans_recommendations,
    format_recommendations
)
from utils import (
    find_closest_songs,
    prompt_for_song_selection,
    clear_screen
)
from decisiontree import get_decision_tree_model, SpotifyDecisionTreeModel
from visualize import (
    visualize_kmeans_silhouette,
    visualize_kmeans_pca, visualize_song_similarity, visualize_feature_importance, visualize_kmeans_cluster
)
from regression import train_regressors


def initialize_system():
    """
    Ensure all models/data are loaded or trained.
    Returns: (knn_model, df, scaled_features, kmeans_model)
    """
    print("Initializing Spotify Music Recommendation System...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # ── KNN ─────────────────────────────────────────────────────────────────────────
    if os.path.exists(config.MODEL_FILE):
        print("Loading existing KNN model...")
        knn_model = get_model() or train_model(force_retrain=True)[0]
        df, scaled, y, scaler = get_data()
    else:
        print("Training new KNN model...")
        knn_model, df, scaled = train_model()

    # ── KMeans ───────────────────────────────────────────────────────────────────────
    kmeans_model = get_kmeans_model(scaled)

    return knn_model, df, scaled, kmeans_model


# Single initialization
model, processed_df, scaled_features, kmeans_model = initialize_system()
print("\nAnalyzing KMeans clusters...")
kmeans_model.analyze_clusters(processed_df)


def display_menu():
    """
    Show main menu and return user choice.
    """
    print(config.MENU_PROMPT)
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in ['1','2','3','4','5']:
            return choice
        print("Invalid choice. Please enter 1-5.")


def handle_song_recommendations():
    """
    Option 1: KNN + KMeans song recommendations.
    """
    name = input("Enter song name: ")
    matches = prompt_for_song_selection(
        find_closest_songs(name, processed_df)
    )
    if not matches:
        print("No matches found."); input(); return

    idx = matches['index']
    knn_recs = find_similar_songs(
        scaled_features[idx],
        model,
        processed_df,
        scaled_features,
        exclude_indices=[idx]
    )
    km_recs = get_kmeans_recommendations(
        idx, processed_df, scaled_features, kmeans_model
    )

    print("\nKNN Recommendations:")
    print(format_recommendations(knn_recs))
    print("\nKMeans Recommendations:")
    print(format_recommendations(km_recs))

    if config.VISUALIZATION_ENABLED and input("\nVisualize? (y/n): ").lower()=='y':
        sim_idxs = [rec['index'] for rec in knn_recs]
        # 1) 2D similarity plot
        visualize_song_similarity(idx, sim_idxs, processed_df, scaled_features)
        # 2) feature‑importance bar chart
        visualize_feature_importance(model, processed_df, idx, sim_idxs)
        # 3) KMeans cluster plot
        visualize_kmeans_cluster(idx, kmeans_model, processed_df, scaled_features)

    input("\nPress Enter to continue.")


def handle_model_management():
    """
    Option 2: Retrain or inspect models.
    """
    submenu = """
Model Management
----------------------------------
1. Retrain KNN model
2. Retrain KMeans model
3. Retrain Decision Tree model
4. Retrain all models
5. Return to main menu
"""
    print(submenu)
    choice = input("Choice (1-5): ").strip()
    if choice == '1':
        train_model(force_retrain=True)
    elif choice == '2':
        from kmeans_model import SpotifyKMeansModel
        SpotifyKMeansModel().fit(scaled_features).save()
    elif choice == '3':
        df, X, y, _ = get_data()
        get_decision_tree_model(X, y, retrain=True)
    elif choice == '4':
        # Retrain all three in correct order
        for path in [
            config.MODEL_FILE,
            config.KMEANS_MODEL_FILE,
            config.DECISIONTREE_MODEL_FILE
        ]:
            if os.path.exists(path):
                os.remove(path)
        # 1) KNN
        knn_model, df, scaled = train_model(force_retrain=True)
        # 2) KMeans
        from kmeans_model import SpotifyKMeansModel
        SpotifyKMeansModel().fit(scaled).save()
        # 3) Decision Tree
        df, X, y, _ = get_data()
        get_decision_tree_model(X, y, retrain=True)
    input("\nDone. Press Enter to continue.")


def handle_hit_prediction():
    """
    Option 3: Show actual popularity vs. hit prediction.
    """
    name = input("Enter song name for hit prediction: ")
    matches = prompt_for_song_selection(
        find_closest_songs(name, processed_df)
    )
    if not matches:
        print("No matches found."); input(); return

    idx = matches['index']
    actual = processed_df.iloc[idx]['popularity']
    dt = SpotifyDecisionTreeModel.load()
    if dt is None:
        print("Decision Tree model missing; retrain first."); input(); return

    x = scaled_features[idx].reshape(1, -1)
    pred = dt.predict(x)[0]
    proba = dt.model.predict_proba(x)[0][1]

    song = processed_df.iloc[idx]['name']
    print(f"\nSong: {song}")
    print(f"Actual popularity : {actual}")
    print(f"Model prediction : {'Hit 🎉' if pred else 'Non-Hit 💔'} (p={proba:.3f})")
    input("\nPress Enter to continue.")

def handle_analysis():
    """
    Option 4:
      • KMeans silhouette + PCA on a 2 000‐song subset
      • Regression metrics on a 20 000‐song subset
    """
    # ─── KMeans viz subset ───────────────────────────────────────────────────────
    km_sample = min(2000, len(scaled_features))
    km_idx    = np.random.choice(len(scaled_features), size=km_sample, replace=False)
    km_feats  = scaled_features[km_idx]
    km_labels = kmeans_model.model.labels_[km_idx]

    print(f"\n--- KMeans Silhouette Analysis (subset of {km_sample}) ---")
    score = silhouette_score(km_feats, km_labels)
    print(f"Overall silhouette score: {score:.3f}")

    print(f"\n--- KMeans PCA Projection (subset of {km_sample}) ---")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(km_feats)
    plt.figure(figsize=config.FIGURE_SIZE)
    for lbl in np.unique(km_labels):
        mask = km_labels == lbl
        plt.scatter(proj[mask, 0], proj[mask, 1], alpha=0.6, label=f"Cluster {lbl}")
    plt.title("KMeans Clusters (PCA - Subset)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=0.5, bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

    # ─── Regression on a smaller subset ─────────────────────────────────────────
    reg_sample = min(20000, len(processed_df))
    reg_idx    = np.random.choice(len(processed_df), size=reg_sample, replace=False)
    reg_df     = processed_df.iloc[reg_idx]
    reg_feats  = scaled_features[reg_idx]

    print(f"\n--- Regression Models on Subset Popularity (subset of {reg_sample}) ---")
    train_regressors(reg_df, reg_feats)

    input("\nPress Enter to continue.")


def main():
    try:
        while True:
            clear_screen()
            choice = display_menu()

            if choice == '1':
                handle_song_recommendations()
            elif choice == '2':
                handle_model_management()
            elif choice == '3':
                handle_hit_prediction()
            elif choice == '4':
                handle_analysis()
            else:  # '5'
                print("Goodbye!"); break

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
