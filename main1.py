"""
Main training script for the Car Price Prediction model.

This script performs the following steps:
1.  Loads the car price dataset.
2.  Defines custom scikit-learn transformers for preprocessing and feature selection (TabularPreprocessor, BICForwardSelector).
3.  Splits the data into training, validation, and test sets.
4.  Defines pipelines for four models:
    - Linear Regression (with BIC Forward Selection)
    - Ridge Regression
    - Lasso Regression
    - ElasticNet Regression
5.  Trains each model on the training data.
6.  Evaluates each model on the train, val, and test sets, saving metrics to a CSV.
7.  Saves the trained model pipelines using joblib.
8.  Generates and saves diagnostic plots for the validation set.
9.  Generates and saves comparison plots for model metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import statsmodels.api as sm

# --- Constants ----

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "archive" / "CarPrice_Assignment.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "model_metrics_main1.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

TARGET_COLUMN = "price" # The variable to predict
DROP_COLUMNS = ["car_ID"] # Columns to remove before preprocessing
RANDOM_STATE = 42 # Ensure reproducibke splist and model training

# Dictionary mapping (model names to save paths)
MODEL_PATHS = {
    "linear_bic": MODELS_DIR / "linear_bic_pipeline.pkl",
    "ridge": MODELS_DIR / "ridge_pipeline.pkl",
    "lasso": MODELS_DIR / "lasso_pipeline.pkl",
    "elastic_net": MODELS_DIR / "elasticnet_pipeline.pkl",
}

# --- Custom Transformers ---

class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for preprocessing tabular data.
    It performs:
    1.  Dropping specified columns.
    2.  One-hot encoding of categorical features.
    3.  Standard scaling of numeric features.
    """

    def __init__(self, drop_columns: Optional[List[str]] = None, drop_first: bool = True):
        """
        Args:
            drop_columns: A list of column names to drop from the DataFrame.
            drop_first: Whether to drop the first category in one-hot encoding to avoid multicollinearity.
        """
        self.drop_columns = drop_columns or []
        self.drop_first = drop_first

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns the preprocessing steps from the training data.
        - Identifies categorical and numeric columns.
        - Fits a StandardScaler on the numeric columns.
        - Stores the complete list of feature columns after one-hot encoding.

        Args:
            X: Training data (pandas DataFrame).
            y: Ignored.
        """
        df = self._prepare_frame(X)

        # Identify categorical columns
        self.categorical_columns_ = df.select_dtypes(include="object").columns.tolist()

        # Get all feature names after encoding
        df_encoded = pd.get_dummies(df, columns=self.categorical_columns_, drop_first=self.drop_first)
        self.feature_columns_ = df_encoded.columns.tolist()

        # Identify numeric columns (excluding binary/dummy variables)
        self.numeric_columns_ = [
            col
            for col in self.feature_columns_
            if pd.api.types.is_numeric_dtype(df_encoded[col]) and df_encoded[col].nunique() > 2
        ]

        # Fit scaler on numeric columns
        if self.numeric_columns_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(df_encoded[self.numeric_columns_])
        else:
            self.scaler_ = None
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the learned preprocessing steps to new data.

        Args:
            X: Data to transform (pandas DataFrame).

        Returns:
            Processed DataFrame with scaled numeric features,
            one-hot encoded categorical features, and consistent columns.
        """
        df = self._prepare_frame(X)

        # Handle categorical columns that might be in `fit` but not `transform`
        missing = set(self.categorical_columns_) - set(df.columns)
        for col in missing:
            df[col] = pd.NA # Add missing columns with NA

        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=self.categorical_columns_, drop_first=self.drop_first)

        # Ensure columns match the 'fit' step
        df_encoded = df_encoded.reindex(columns=self.feature_columns_, fill_value=0)

        # Scale numeric columns
        if self.numeric_columns_:
            df_encoded.loc[:, self.numeric_columns_] = self.scaler_.transform(df_encoded[self.numeric_columns_])

        return df_encoded

    def _prepare_frame(self, X):
        """Helper to copy and drop specified columns."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Handle non-DataFrame inputs if necessary
            df = pd.DataFrame(X, columns=getattr(self, "feature_columns_", None))

        drop_cols = [c for c in self.drop_columns if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df


class BICForwardSelector(BaseEstimator, TransformerMixin):
    """
    A custom feature selector using Forward Stepwise Selection based on the
    Bayesian Information Criterion (BIC).

    Uses `statsmodels.OLS` to calculate BIC.
    """

    def __init__(self, max_features: Optional[int] = None, verbose: bool = True):
        """
        Args:
            max_features: The maximum number of features to select. If None, selection continues until BIC no longer improves.
            verbose: If True, prints the selection process.
        """
        self.max_features = max_features
        self.verbose = verbose

    def fit(self, X, y):
        """
        Selects the best subset of features based on BIC.

        Args:
            X: Training data (DataFrame or array).
            y: Target variable (Series or array).
        """
        df = self._ensure_frame(X)
        y_series = self._ensure_series(y)
        X_const = sm.add_constant(df, has_constant="add")

        remaining = [col for col in X_const.columns if col != "const"]
        selected = ["const"]

        # Calculate baseline BIC
        baseline = sm.OLS(y_series, X_const[selected]).fit()
        best_bic = baseline.bic
        self._log(f"[BIC] Start BIC={best_bic:.2f}")

        while remaining:
            candidate = None
            candidate_bic = best_bic

            # Iterate over all remaining features
            for feature in remaining:
                cols = selected + [feature]
                try:
                    # Fit model and get BIC
                    bic = sm.OLS(y_series, X_const[cols]).fit().bic
                except Exception:
                    # Catch potential errors.
                    continue

                # If this feature improves BIC, get it
                if bic < candidate_bic:
                    candidate_bic = bic
                    candidate = feature

            # Stop if no feature improved the BIC
            if candidate is None:
                self._log("[BIC] Stop: no better attribute.")
                break

            # Add the best feature to selected list
            selected.append(candidate)
            remaining.remove(candidate)
            best_bic = candidate_bic
            self._log(f"[BIC] + {candidate} -> BIC={best_bic:.2f}")

            # Stop if hit the feature limit
            if self.max_features and len(selected) - 1 >= self.max_features:
                self._log(f"[BIC] Reached max_features={self.max_features}.")
                break

        # Final selected features (excluding the constant)
        chosen = [feature for feature in selected if feature != "const"]
        if not chosen:
            # Fallback in case no features are selected
            chosen = list(df.columns)

        self.selected_features_ = chosen
        self.input_feature_names_ = list(df.columns)
        self.final_bic_ = best_bic
        return self

    def transform(self, X):
        """
        Reduces the DataFrame to only the selected features.

        Args:
            X: Data to transform.

        Returns:
            DataFrame with only the `selected_features_`.
        """
        df = self._ensure_frame(X)
        missing = set(self.selected_features_) - set(df.columns)
        if missing:
            raise ValueError(f"BIC selector expected columns missing: {sorted(missing)}")
        return df[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        """Returns the selected feature names."""
        return np.array(self.selected_features_)

    def _ensure_frame(self, X):
        """Helper to ensure X is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        columns = getattr(self, "input_feature_names_", None)
        return pd.DataFrame(X, columns=columns)

    def _ensure_series(self, y):
        """Helper to ensure y is a Series."""
        if isinstance(y, pd.Series):
            return y
        return pd.Series(y, name="target")

    def _log(self, message: str):
        """Prints a message if verbose is True."""
        if self.verbose:
            print(message)


# --- Helper Functions ---

def evaluate_split(name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series):
    """
    Calculates regression metrics for a given data split.

    Args:
        name: Name of the split.
        model: The trained scikit-learn Pipeline.
        X: Features for the split.
        y: True target values for the split.

    Returns:
        A dictionary with MAE, RMSE, and R2.
    """
    preds = model.predict(X)
    return {
        "split": name,
        "MAE": mean_absolute_error(y, preds),
        "RMSE": mean_squared_error(y, preds) ** 0.5,
        "R2": r2_score(y, preds),
    }


def build_pipelines():
    """
    Constructs all model pipelines used for training.

    Note: Hyperparameters are hardcoded, assuming
    they were pre-tuned in a separate process.

    Returns:
        A dictionary mapping model names to their corresponding
        scikit-learn Pipeline objects.
    """
    pipelines = {
        "linear_bic": Pipeline(
            steps=[
                ("preprocess", TabularPreprocessor(drop_columns=DROP_COLUMNS)),
                ("bic_selector", BICForwardSelector(verbose=True)),
                ("linear_model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("preprocess", TabularPreprocessor(drop_columns=DROP_COLUMNS)),
                ("linear_model", Ridge(alpha=33.6, random_state=RANDOM_STATE)),
            ]
        ),
        "lasso": Pipeline(
            steps=[
                ("preprocess", TabularPreprocessor(drop_columns=DROP_COLUMNS)),
                ("linear_model", Lasso(alpha=78.48, random_state=RANDOM_STATE, max_iter=10000)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("preprocess", TabularPreprocessor(drop_columns=DROP_COLUMNS)),
                (
                    "linear_model",
                    ElasticNet(alpha=1.129, l1_ratio=0.7, random_state=RANDOM_STATE, max_iter=10000),
                ),
            ]
        ),
    }
    return pipelines


# --- Plotting Functions ---

def plot_pred_vs_actual(y_true: pd.Series, y_pred: np.ndarray, model_name: str, out_dir: Path):
    """Saves a scatter plot of predicted vs. actual values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual y (validation)")
    plt.ylabel("Predicted y")
    plt.title(f"Predicted vs Actual - {model_name}")
    plt.tight_layout()
    out_path = out_dir / f"{model_name}_pred_vs_actual.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str, out_dir: Path):
    """Saves a scatter plot of residuals vs. predicted values."""
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted y")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted - {model_name}")
    plt.tight_layout()
    out_path = out_dir / f"{model_name}_residuals.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_residual_hist(y_true: pd.Series, y_pred: np.ndarray, model_name: str, out_dir: Path):
    """Saves a histogram of the residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual distribution - {model_name}")
    plt.tight_layout()
    out_path = out_dir / f"{model_name}_residual_hist.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, out_dir: Path):
    """
    Saves bar charts comparing MAE, RMSE, and R2 for all models
    on the validation set.

    Args:
        metrics_df: DataFrame with columns ['model', 'split', 'MAE', 'RMSE', 'R2'].
        out_dir: Directory to save the plots.
    """
    # Filter for validation metrics only
    val_df = metrics_df[metrics_df["split"] == "val"].copy()
    val_df = val_df.set_index("model")
    models_order = val_df.index.tolist()

    # Plot MAE
    plt.figure(figsize=(6, 4))
    plt.bar(models_order, val_df["MAE"])
    plt.ylabel("MAE")
    plt.title("Model comparison - MAE (validation)")
    plt.tight_layout()
    plt.ylim(val_df["MAE"].min() * 0.9, val_df["MAE"].max() * 1.1)
    plt.savefig(out_dir / "compare_MAE.png", dpi=200)
    plt.close()

    # Plot RMSE
    plt.figure(figsize=(6, 4))
    plt.bar(models_order, val_df["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Model comparison - RMSE (validation)")
    plt.tight_layout()
    plt.ylim(val_df["RMSE"].min() * 0.9, val_df["RMSE"].max() * 1.1)
    plt.savefig(out_dir / "compare_RMSE.png", dpi=200)
    plt.close()

    # Plot R2
    plt.figure(figsize=(6, 4))
    plt.bar(models_order, val_df["R2"])
    plt.ylabel("R²")
    plt.title("Model comparison - R² (validation)")
    plt.tight_layout()
    plt.ylim(max(val_df["R2"].min() - 0.05, 0), min(val_df["R2"].max() + 0.05, 1))
    plt.savefig(out_dir / "compare_R2.png", dpi=200)
    plt.close()


# --- Main Execution ---

def main():
    """
    Main function to run the entire training and evaluation pipeline.
    """
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Train and Evaluate Models
    pipelines = build_pipelines()
    metrics = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    val_predictions = {}

    for model_name, pipeline in pipelines.items():
        print(f"\n=== Training model: {model_name} ===")
        pipeline.fit(X_train, y_train)

        if model_name == "linear_bic":
            bic_step = pipeline.named_steps["bic_selector"]
            print(f"Selected {len(bic_step.selected_features_)} columns via BIC.")

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            split_metrics = evaluate_split(split_name, pipeline, X_split, y_split)
            split_metrics["model"] = model_name
            metrics.append(split_metrics)

        model_path = MODEL_PATHS[model_name]
        joblib.dump(pipeline, model_path)
        print(f"Saved {model_name} pipeline to {model_path.resolve()}")

        y_val_pred = pipeline.predict(X_val)
        val_predictions[model_name] = y_val_pred

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[["model", "split", "MAE", "RMSE", "R2"]]
    metrics_df.to_csv(METRICS_PATH, index=False)
    print("\n=== Metrics ===")
    print(metrics_df)
    print(f"\nSaved metrics to {METRICS_PATH.resolve()}")

    # Plotting
    for model_name, y_pred in val_predictions.items():
        plot_pred_vs_actual(y_val, y_pred, model_name, FIGURES_DIR)
        plot_residuals(y_val, y_pred, model_name, FIGURES_DIR)
        plot_residual_hist(y_val, y_pred, model_name, FIGURES_DIR)

    # Generate comparison plots
    plot_metric_bars(metrics_df, FIGURES_DIR)

    print(f"\nAll figures saved to '{FIGURES_DIR.resolve()}' and metrics saved to '{METRICS_PATH.resolve()}'.")
    

if __name__ == "__main__":
    main()
