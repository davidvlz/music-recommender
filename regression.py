# regression.py

"""
Train and evaluate regression models to predict raw popularity.
"""

import os
import pickle
import config
from math import sqrt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_regressors(df, X):
    """
    Train and report metrics for multiple regressors on df['popularity'].
    Saves each trained model to models/<name>_popularity.pkl.
    """
    # Extract the continuous popularity target
    y = df['popularity'].values

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    regressors = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print("\nRegression model performance:")
    for name, model in regressors.items():
        # Fit
        model.fit(X_train, y_train)
        # Predict
        preds = model.predict(X_test)
        # Compute metrics
        mse = mean_squared_error(y_test, preds)
        rmse = sqrt(mse)
        r2   = r2_score(y_test, preds)
        print(f"  {name:<16} RMSE={rmse:.2f}   R²={r2:.3f}")

        # Save the trained model
        fname = f"{name.lower()}_popularity.pkl"
        path = os.path.join(config.MODELS_DIR, fname)
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    print("\nRegression models saved to the models directory.")
