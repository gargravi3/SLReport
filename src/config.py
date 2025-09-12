"""
Configuration constants for the hotel booking cancellation prediction project.

This module contains all the configuration parameters used across the project,
including data paths, model parameters, and evaluation settings.
"""

import os
from pathlib import Path

# Project root directory (assuming config.py is in src/)
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, METRIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_PATH = ROOT_DIR / "hotel_bookings.csv"
TARGET_COLUMN = "is_canceled"
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
POSITIVE_RATE = 0.37  # 37.0% cancellation rate
PR_AUC_BASELINE = 0.3704  # Baseline PR-AUC (equal to positive rate)

# Columns to drop (leakage control)
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# Data splitting
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 80/20 train/test split
CV_FOLDS = 5  # Number of cross-validation folds

# Class imbalance handling
PRIMARY_IMBALANCE_STRATEGY = "class_weight"  # Use class_weight='balanced' where supported
SECONDARY_IMBALANCE_STRATEGY = "smote"  # Optional: SMOTE for sensitivity checks

# Preprocessing
CATEGORICAL_THRESHOLD = 10  # Max unique values to consider a column categorical
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

# --------------------------------------------------------------------------- #
# New global guidance constants
# --------------------------------------------------------------------------- #
# Encoding strategy for high-cardinality categoricals
ENCODING_STRATEGY = "frequency"          # 'frequency' or 'target'
# Force float32 everywhere (models will cast)
DTYPE_FLOAT = "float32"
# Cross-validation strategy (StratifiedShuffleSplit‐style)
CV_SPLITS = 3
CV_TEST_SIZE = 0.2
# Default train-sizes for learning curves
LEARNING_CURVE_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Model configurations
# Decision Tree
DT_PARAMS = {
    "criterion": "gini",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED
}
DT_PARAM_GRID = {
    # Regularised grid per assignment guidance
    "max_depth": [6, 10, 14, 18],
    "min_samples_leaf": [50, 100, 200],
    "min_samples_split": [100, 200, 400],
    "max_features": ["sqrt", "log2", 0.5],
    "ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3]
}

# k-Nearest Neighbors
KNN_PARAMS = {
    # Algorithm settings aligned with guidance
    "algorithm": "brute",
    "metric": "euclidean",
    "n_jobs": -1
}
KNN_PARAM_GRID = {
    # Only vary k; keep other params fixed
    "n_neighbors": [3, 5, 11, 21],
    "weights": ["uniform", "distance"]
}

# Support Vector Machine
SVM_LINEAR_PARAMS = {
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter": 20000,
    "dual": False
}
SVM_LINEAR_PARAM_GRID = {
    "C": [0.1, 1, 10]
}

SVM_RBF_PARAMS = {
    "kernel": "rbf",
    "class_weight": "balanced",
    "probability": False,      # Faster training; calibrate later
    "cache_size": 2000,        # MB
    "random_state": RANDOM_SEED,
    "max_iter": 10000
}
SVM_RBF_PARAM_GRID = {
    "C": [0.5, 2, 8],
    # gamma numeric values will be injected in trainer based on d
    "gamma": ["scale"]
}

# Neural Network (SGD)
NN_PARAMS = {
    "solver": "sgd",
    "activation": "relu",
    "random_state": RANDOM_SEED,
    "max_iter": 15,            # epoch cap per guidance
    "early_stopping": True,
    "n_iter_no_change": 3,
    "batch_size": 1024,
    "tol": 1e-4,
    "momentum": 0.0,           # Added to enforce pure SGD
    "nesterovs_momentum": False # Added to enforce pure SGD
}
NN_PARAM_GRID = {
    "hidden_layer_sizes": [(512, 512), (256, 256, 128, 128)],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [0.001, 0.01]
}

# Calibration
CALIBRATION_METHOD = "sigmoid"  # Options: 'sigmoid' or 'isotonic'
CALIBRATION_CV = 5

# Evaluation
THRESHOLD_METRIC = "f1"  # Metric to optimize threshold for (f1, precision, recall)
MAIN_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
