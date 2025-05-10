# visualize.py

"""
Visualization module for the Spotify Music Recommendation System.

This module provides functions for visualizing the recommendation system,
including song similarity, feature importance, KMeans analysis, silhouette
plots, and PCA projections.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", message="Tight layout not applied")
warnings.filterwarnings("ignore", category=UserWarning)

import config


def plot_feature_distributions(processed_df):
    """
    Plot distributions of audio features.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    features = [f for f in config.AUDIO_FEATURES if f in processed_df.columns]
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    plt.figure(figsize=config.FIGURE_SIZE)
    for i, feat in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        plt.hist(processed_df[feat], bins=30, alpha=0.7)
        plt.title(feat)
    plt.suptitle("Audio Feature Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def visualize_song_similarity(song_idx, similar_indices, processed_df, scaled_features, method='pca'):
    """
    Visualize song similarity in 2D via PCA or t-SNE.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    idxs = [song_idx] + similar_indices
    feats = scaled_features[idxs]

    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title = "Song Similarity (PCA)"
    else:
        reducer = TSNE(n_components=2, random_state=42)
        title = "Song Similarity (t-SNE)"

    proj = reducer.fit_transform(feats)
    plt.figure(figsize=config.FIGURE_SIZE)
    plt.scatter(proj[1:,0], proj[1:,1], alpha=0.7, label="Similar Songs")
    plt.scatter(proj[0,0], proj[0,1], color='red', s=100, label="Input Song")
    for i, idx in enumerate(idxs):
        song = processed_df.iloc[idx]
        plt.annotate(f"{song['name']} - {song['artists']}",
                     (proj[i,0], proj[i,1]),
                     textcoords="offset points", xytext=(0,10), ha='center')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_feature_importance(model, processed_df, song_idx, similar_indices):
    """
    Bar chart of feature differences between an input song and its neighbors.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    feats = [f for f in config.AUDIO_FEATURES if f in processed_df.columns]
    inp = processed_df.iloc[song_idx][feats]
    others = processed_df.iloc[similar_indices][feats]
    avg = others.mean()
    diff = inp - avg

    # sort by absolute difference
    order = np.argsort(np.abs(diff.values))[::-1]
    names = [feats[i] for i in order]
    vals = diff.values[order]

    plt.figure(figsize=config.FIGURE_SIZE)
    bars = plt.bar(range(len(names)), vals, color=[ 'green' if v>0 else 'red' for v in vals ])
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.axhline(0, color='black', lw=1)
    plt.title("Feature Differences (Input − Avg Similar)")
    plt.tight_layout()
    plt.show()


def plot_song_features_radar(song_indices, processed_df, title=None):
    """
    Radar (spider) chart comparing multiple songs’ audio features.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    feats = [f for f in config.AUDIO_FEATURES if f in processed_df.columns]
    angles = np.linspace(0, 2*np.pi, len(feats), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, subplot_kw={'polar':True})
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feats)

    for idx in song_indices:
        song = processed_df.iloc[idx]
        vals = []
        for f in feats:
            v = song[f]
            # normalize tempo, loudness if necessary
            if f == 'tempo':
                v = max(0, min(1, (v-50)/150))
            elif f == 'loudness':
                v = max(0, min(1, (v+60)/60))
            vals.append(v)
        vals = np.concatenate((vals, [vals[0]]))
        ax.plot(angles, vals, linewidth=2, label=song['name'])
        ax.fill(angles, vals, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(0.1,0.1))
    if title: ax.set_title(title)
    else: ax.set_title("Feature Comparison Radar")
    plt.tight_layout()
    plt.show()


def visualize_kmeans_cluster(song_idx, kmeans_model, processed_df, scaled_features, method='pca'):
    """
    Plot the cluster of an input song in 2D PCA or t-SNE space.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    vec = scaled_features[song_idx].reshape(1, -1)
    label = kmeans_model.predict(vec)[0]
    members = kmeans_model.get_cluster_members(label, processed_df)
    feats = scaled_features[members]

    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title = f"KMeans Cluster {label} (PCA)"
    else:
        reducer = TSNE(n_components=2, random_state=42)
        title = f"KMeans Cluster {label} (t-SNE)"

    proj = reducer.fit_transform(feats)
    plt.figure(figsize=config.FIGURE_SIZE)
    plt.scatter(proj[:,0], proj[:,1], alpha=0.6, label="Cluster Members", color='blue')
    pos = members.index(song_idx)
    plt.scatter(proj[pos,0], proj[pos,1], color='red', s=120, label="Input Song")
    song = processed_df.iloc[song_idx]
    plt.annotate(f"{song['name']} - {song['artists']}", (proj[pos,0], proj[pos,1]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='red')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_kmeans_silhouette(scaled_features, kmeans_model):
    """
    Silhouette analysis for a fitted KMeans model.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    labels = kmeans_model.model.labels_
    n_clusters = kmeans_model.n_clusters
    score = silhouette_score(scaled_features, labels)
    print(f"Overall silhouette score: {score:.3f}")

    sil_vals = silhouette_samples(scaled_features, labels)
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    y_lower = 10
    for i in range(n_clusters):
        vals = np.sort(sil_vals[labels==i])
        y_upper = y_lower + len(vals)
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5*len(vals), str(i))
        y_lower = y_upper + 10
    ax.set_title("Silhouette Plot for KMeans Clusters")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(score, color="red", linestyle="--")
    plt.show()


def visualize_kmeans_pca(scaled_features, kmeans_model):
    """
    Scatter‐plot KMeans clusters projected to 2D PCA space.
    """
    if not config.VISUALIZATION_ENABLED:
        print("Visualization is disabled.")
        return

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(scaled_features)
    labels = kmeans_model.model.labels_

    plt.figure(figsize=config.FIGURE_SIZE)
    for lbl in np.unique(labels):
        idxs = labels == lbl
        plt.scatter(proj[idxs,0], proj[idxs,1], label=f"Cluster {lbl}", alpha=0.6)
    plt.title("KMeans Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=0.5, bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()
