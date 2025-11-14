"""
Hyperparameter tuning script for regularized linear models.

This script uses GridSearchCV to find the optimal hyperparameters for
Ridge, Lasso, and ElasticNet models using pre-processed training data.
The results from this script (e.g., best alpha values) can be
used in the main training pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Constants ---

X_train = pd.read_csv("archive/train/x_train.csv")
y_train = pd.read_csv("archive/train/y_train.csv").squeeze()

print("Loaded training data:", X_train.shape, "target:", y_train.shape)

# --- Parameter Grids ---

alpha_grid = np.logspace(-4, 3, 20)
l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

# --- Helper Function ---

def run_grid_search(model, param_grid, name):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model)
    ])

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print(f"\n🔹 {name} Results")
    print("--------------------------")
    print("Best Params:", search.best_params_)
    print(f"Best CV R²: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


ridge_params = {"model__alpha": alpha_grid}
best_ridge, ridge_best_params, ridge_best_score = run_grid_search(
    Ridge(), ridge_params, "Ridge"
)

lasso_params = {"model__alpha": alpha_grid}
best_lasso, lasso_best_params, lasso_best_score = run_grid_search(
    Lasso(max_iter=10000), lasso_params, "Lasso"
)

elastic_params = {
    "model__alpha": alpha_grid,
    "model__l1_ratio": l1_grid
}
best_enet, enet_best_params, enet_best_score = run_grid_search(
    ElasticNet(max_iter=10000, random_state=42),
    elastic_params,
    "ElasticNet"
)

print("\n===============================")
print("SUMMARY:")
print("===============================")
print(f"Ridge:      R² = {ridge_best_score:.4f}, alpha = {ridge_best_params['model__alpha']:.4g}")
print(f"Lasso:      R² = {lasso_best_score:.4f}, alpha = {lasso_best_params['model__alpha']:.4g}")
print(f"ElasticNet: R² = {enet_best_score:.4f}, alpha = {enet_best_params['model__alpha']:.4g}, l1_ratio = {enet_best_params['model__l1_ratio']}")