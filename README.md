# Spotify Music Recommendation System

A command-line tool that recommends songs based on audio features using K-Nearest Neighbors (KNN), Decision Trees, KMeans clustering, and regression analysis.

## Project Overview

This project implements a music recommendation and analysis system using a Spotify dataset. It supports:

- **Content-based filtering** via KNN on audio features  
- **Hit prediction** (binary classification) with a Decision Tree  
- **Cluster analysis** with KMeans (including silhouette & PCA visualizations)  
- **Popularity regression** (predict raw popularity scores)  
- **Fuzzy matching** for robust song lookup  
- **Optional visualizations** in 2D and radar charts  

## Key Features

1. **Audio-feature recommendations** (KNN)  
2. **Hit vs. non-hit prediction** (Decision Tree, median-split on popularity)  
3. **Cluster analysis**  
   - Silhouette score (cluster quality)  
   - PCA projection of clusters  
4. **Popularity regression**  
   - Linear, Ridge, and Random Forest regressors  
   - RMSE & R² metrics  
5. **Fuzzy matching** on song title (and optional artist/year)  
6. **Interactive CLI** with menus for recommendations, model management, hit prediction, analysis, and more  

## Installation

### How to run the program

1. **Extract** the ZIP and **navigate** into the project directory:

   ```bash
   cd music-recommender
   ```

2. **Create and activate** a virtual environment:

   macOS / Linux
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   Windows (PowerShell)
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   Windows (CMD)
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   python main.py
   ```

## ▶️ Usage

On first run, the system will:
- Download and preprocess the Spotify dataset
- Train KNN, KMeans, Decision Tree, and regression models
- Save all models under `models/`

### Main Menu
```
1. Get song recommendations  
2. Model management  
3. Predict Hit?  
4. Analysis  
5. Exit  
```

#### 1. Get song recommendations
- Enter a song name (optionally include artist/year)
- Select from fuzzy-matched list
- View KNN & KMeans recommendations
- Optionally visualize similarity (PCA, radar, etc.)

#### 2. Model management
- Retrain KNN
- Retrain KMeans
- Retrain Decision Tree
- Retrain all models
- Return to main menu

#### 3. Predict Hit?
- Enter a song name
- Shows actual popularity vs. "Hit 🎉" or "Non-Hit 💔" (with probability)

#### 4. Analysis
- Silhouette score on a 2,000-song subset
- PCA cluster plot on same subset
- Regression metrics (RMSE & R²) on a 20,000-song subset
- Saves regression models:
  - `linear_popularity.pkl`
  - `ridge_popularity.pkl`
  - `randomforest_popularity.pkl`

#### 5. Exit
- Quit the application

## 🗂 Project Structure

```
├── config.py            # constants, hyperparameters, file paths  
├── data_loader.py       # download, load, preprocess, scale data, build labels  
├── model.py             # KNN model wrapper  
├── kmeans_model.py      # KMeans clustering wrapper  
├── decisiontree.py      # Decision Tree classifier for hit prediction  
├── train.py             # KNN training & evaluation script  
├── regression.py        # Train & evaluate regression models on popularity  
├── recommend.py         # Recommendation logic (KNN & KMeans)  
├── visualize.py         # Visualization helpers (PCA, silhouette, radar, t-SNE)  
├── utils.py             # Fuzzy matching, CLI helpers  
├── main.py              # CLI entry point & menu handlers  
├── requirements.txt     # Python dependencies  
└── models/              # Saved `.pkl` model files  
```

## 🔍 How It Works

### Data Processing
Features:
- danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo

Missing values dropped; features scaled via StandardScaler.

Hit label: binary 1 if popularity ≥ median (configurable), else 0.

### Recommendation (KNN)
- User inputs song name → fuzzy match → select track
- KNN finds K nearest neighbors in feature space (N_NEIGHBORS, DISTANCE_METRIC)
- Top-N recommendations displayed

### Hit Prediction (Decision Tree)
- Trained on same features to predict hit/non-hit
- Outputs label & probability

### Cluster Analysis (KMeans)
- Fit on full feature matrix (N_CLUSTERS, default 8 or configured)
- Silhouette score measures cohesion/separation
- PCA projection visualizes clusters in 2D

### Popularity Regression
- Models: LinearRegression, Ridge, RandomForestRegressor
- Metrics: RMSE (error in popularity points) and R² (variance explained)

## 🛠 Command-Line Tools

Train KNN
```bash
python train.py --force 
python train.py --evaluate
```

Get recommendations
```bash
python recommend.py --song "Song Name" --artist "Artist Name"  
```

Visualize similarity
```bash
python visualize.py --song "Song Name"  
```

## ⚙️ Configuration

Edit `config.py` to customize:
- N_NEIGHBORS, DISTANCE_METRIC for KNN
- N_CLUSTERS for KMeans
- POPULARITY_THRESHOLD for hit definition
- AUDIO_FEATURES list
- FUZZY_MATCH_CUTOFF for song matching
- VISUALIZATION_ENABLED, FIGURE_SIZE

## 📦 Data Source

Spotify Dataset on Kaggle: vatsalmavani/spotify-dataset
- Downloaded via kagglehub or manually placed under data/

## 📝 Requirements

- Python 3.8+
- numpy, pandas, scikit-learn, matplotlib
- pickle, difflib (stdlib), python-Levenshtein (optional)