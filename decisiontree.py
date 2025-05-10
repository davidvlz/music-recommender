"""
Decision‐tree model module for the Spotify Music Recommendation System.

Defines SpotifyDecisionTreeModel for training, saving, loading, and predicting
with a DecisionTreeClassifier, and a CLI entry‐point to train/evaluate.
"""

import os
import pickle
import config
from data_loader import get_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

class SpotifyDecisionTreeModel:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy'):
        """
        Initialize a decision‐tree classifier.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the tree to (X, y).
        """
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Predict labels for X. Requires fit() first.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def save(self, file_path=None):
        """
        Serialize this SpotifyDecisionTreeModel to disk.
        """
        if file_path is None:
            file_path = config.DECISIONTREE_MODEL_FILE
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path=None):
        """
        Load a saved SpotifyDecisionTreeModel from disk, or return None.
        If unpickling fails (e.g. module mismatch), print a warning and return None.
        """
        if file_path is None:
            file_path = config.DECISIONTREE_MODEL_FILE
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"⚠️  Warning: Failed to load Decision Tree model from {file_path}: {e}")
            return None

def get_decision_tree_model(X=None, y=None, retrain=False):
    """
    Load existing model or train a new one if absent or retrain=True.
    Returns a fitted SpotifyDecisionTreeModel.
    """
    model = SpotifyDecisionTreeModel.load()

    # If load failed (None) or user requested retrain, build a new model
    if model is None or retrain:
        if X is None or y is None:
            raise ValueError("X and y must be provided to train the model.")
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        model = SpotifyDecisionTreeModel()
        model.fit(X_train, y_train)
        model.save()
        print(f"✅ Trained new model and saved to {config.DECISIONTREE_MODEL_FILE}")

        # Evaluate
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Sample predictions
        sample_preds = model.predict(X[:5])
        print("Sample predictions for first 5 tracks:", sample_preds)
    else:
        print(f"ℹ️  Loaded existing model from {config.DECISIONTREE_MODEL_FILE}")

    return model

if __name__ == "__main__":
    # 1) Load data (df, X, y, scaler)
    data = get_data()
    if data is None:
        print("❌ Unable to load data; aborting.")
        exit(1)
    df, X, y, scaler = data

    # 2) Train (or retrain) and evaluate
    model = get_decision_tree_model(X, y, retrain=True)
