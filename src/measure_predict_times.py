#!/usr/bin/env python
"""
Measure prediction (inference) wall-clock times for saved calibrated models.

This script:
1. Loads the dataset and creates the same train/test split as the main pipeline
2. For each saved model, measures predict_proba and predict times
3. Calculates estimated fit time by subtracting predict time from total runtime
4. Writes results to a CSV file and prints a summary table

Usage:
    python src/measure_predict_times.py [--dataset {hotel,accidents}]

Example:
    python src/measure_predict_times.py --dataset hotel
    python src/measure_predict_times.py --dataset accidents
"""

import os
import sys
import time
import argparse
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path if needed
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure prediction times for saved models"
    )
    
    parser.add_argument(
        "--dataset", type=str, choices=["hotel", "accidents"], default="hotel",
        help="Dataset to use (default: hotel)"
    )
    
    # Optional comma-separated filter so users can time only a subset of models
    parser.add_argument(
        "--only", type=str, default=None,
        help=(
            "Comma-separated list of model names to process "
            "(e.g., 'decision_tree_calibrated,nn_calibrated'). "
            "If omitted, all available models are processed."
        ),
    )
    
    return parser.parse_args()


def load_dataset(dataset_name: str):
    """
    Load the specified dataset and create train/test split.
    
    Args:
        dataset_name: Name of the dataset ('hotel' or 'accidents')
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_types)
    """
    if dataset_name == "accidents":
        # Import accidents-specific modules
        from src.data_accidents import preprocess_accidents
        X_train, X_test, y_train, y_test, feature_types = preprocess_accidents(do_eda=False)
    else:
        # Import hotel-specific modules
        from src.data import preprocess_dataset
        X_train, X_test, y_train, y_test, feature_types = preprocess_dataset(do_eda=False)
    
    return X_train, X_test, y_train, y_test, feature_types


def get_model_files(dataset_name: str) -> Dict[str, Path]:
    """
    Get paths to saved model files.
    
    Args:
        dataset_name: Name of the dataset ('hotel' or 'accidents')
        
    Returns:
        Dictionary mapping model names to file paths
    """
    from src.config import OUTPUT_DIR
    
    models_dir = OUTPUT_DIR / "models"
    
    # Define model files based on dataset
    if dataset_name == "hotel":
        model_names = [
            "decision_tree_calibrated",
            "knn_calibrated",
            "linear_svm_liblinear_calibrated",
            "rbf_svm_calibrated",
            "nn_calibrated"
        ]
    else:  # accidents
        model_names = [
            "decision_tree_calibrated",
            "knn_calibrated",
            "linear_svm_sgd_calibrated",
            "rbf_svm_calibrated",
            "nn_calibrated"
        ]
    
    # Map model names to file paths
    model_files = {}
    for name in model_names:
        file_path = models_dir / f"{name}.pkl"
        if file_path.exists():
            model_files[name] = file_path
        else:
            print(f"Warning: Model file not found: {file_path}")
    
    return model_files


def stratified_sample_n(
    X: pd.DataFrame,
    y: pd.Series,
    n: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Take a stratified sample of ``n`` rows from ``X`` and ``y``.

    This is a lightweight re-implementation to avoid importing the heavier
    accidents runner just for caps logic.
    """
    from sklearn.model_selection import train_test_split

    if n >= len(X):
        return X, y

    X_sampled, _, y_sampled, _ = train_test_split(
        X, y, train_size=n, random_state=seed, stratify=y
    )
    return X_sampled, y_sampled


def get_end_to_end_runtime(dataset_name: str, model_name: str) -> float:
    """
    Get end-to-end runtime from metrics CSV file.
    
    Args:
        dataset_name: Name of the dataset ('hotel' or 'accidents')
        model_name: Name of the model
        
    Returns:
        End-to-end runtime in seconds, or 0 if not found
    """
    from src.config import OUTPUT_DIR
    
    metrics_file = OUTPUT_DIR / "metrics" / f"metrics_{model_name}.csv"
    
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found: {metrics_file}")
        return 0.0
    
    try:
        metrics_df = pd.read_csv(metrics_file)
        if "runtime_seconds" in metrics_df.columns:
            return float(metrics_df["runtime_seconds"].iloc[0])
        else:
            print(f"Warning: 'runtime_seconds' column not found in {metrics_file}")
            return 0.0
    except Exception as e:
        print(f"Error reading metrics file {metrics_file}: {e}")
        return 0.0


def measure_predict_times(model, X_test) -> Tuple[float, float]:
    """
    Measure predict_proba and predict times for a model.
    
    Args:
        model: Trained model with predict_proba and predict methods
        X_test: Test feature matrix
        
    Returns:
        Tuple of (predict_proba_seconds, predict_seconds)
    """
    # Measure predict_proba time
    start_time = time.perf_counter()
    try:
        model.predict_proba(X_test)
        predict_proba_seconds = time.perf_counter() - start_time
    except (AttributeError, NotImplementedError) as e:
        print(f"Warning: predict_proba not available, using decision_function: {e}")
        try:
            # Try decision_function + sigmoid as fallback
            start_time = time.perf_counter()
            scores = model.decision_function(X_test)
            from scipy.special import expit
            expit(scores)  # sigmoid
            predict_proba_seconds = time.perf_counter() - start_time
        except (AttributeError, NotImplementedError) as e:
            print(f"Warning: decision_function not available either: {e}")
            predict_proba_seconds = 0.0
    
    # Measure predict time
    start_time = time.perf_counter()
    try:
        model.predict(X_test)
        predict_seconds = time.perf_counter() - start_time
    except (AttributeError, NotImplementedError) as e:
        print(f"Warning: predict not available: {e}")
        predict_seconds = 0.0
    
    return predict_proba_seconds, predict_seconds


def main():
    """Main function."""
    args = parse_args()
    
    # Set dataset name environment variable for config
    os.environ['DATASET_NAME'] = args.dataset
    
    # Now import config-dependent modules
    from src.config import OUTPUT_DIR, RANDOM_SEED
    from src.paths import ensure_dir_exists
    
    print(f"\nMeasuring prediction times for {args.dataset} dataset")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    X_train, X_test, y_train, y_test, feature_types = load_dataset(args.dataset)
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Get model files
    model_files = get_model_files(args.dataset)
    print(f"\nFound {len(model_files)} model files")

    # ------------------------------------------------------------------ #
    # Optional filtering via --only flag
    # ------------------------------------------------------------------ #
    if args.only:
        requested = [m.strip() for m in args.only.split(",") if m.strip()]
        print(f"\n--only flag detected. Requested models: {requested}")
        filtered = {k: v for k, v in model_files.items() if k in requested}
        missing = [m for m in requested if m not in filtered]
        for m in missing:
            print(f"Warning: requested model '{m}' not found among saved models.")
        model_files = filtered
    print(f"Models to be processed: {list(model_files.keys())}")
    
    # Measure prediction times for each model
    results = []
    
    # Accidents specific constant (matches run_accidents default)
    ACC_KNN_TEST_CAP = 25_000

    for model_name, model_file in model_files.items():
        print(f"\nProcessing model: {model_name}")
        
        try:
            # Load model
            print(f"  Loading model from {model_file}")
            model = joblib.load(model_file)
            
            # ------------------------------------------------------------ #
            # Apply accidents KNN test-set cap (≤ 25 000 rows)
            # ------------------------------------------------------------ #
            X_test_eff, y_test_eff = X_test, y_test
            if (
                args.dataset == "accidents"
                and model_name.startswith("knn")
                and len(X_test) > ACC_KNN_TEST_CAP
            ):
                print(
                    f"  Sampling test data for KNN: "
                    f"{len(X_test):,} → {ACC_KNN_TEST_CAP:,} rows"
                )
                X_test_eff, y_test_eff = stratified_sample_n(
                    X_test, y_test, ACC_KNN_TEST_CAP
                )
                print(
                    f"  Sampled test set: {len(X_test_eff):,} rows, "
                    f"positive rate: {y_test_eff.mean():.4f}"
                )

            # Measure prediction times
            print(f"  Measuring prediction times...")
            predict_proba_seconds, predict_seconds = measure_predict_times(
                model, X_test_eff
            )
            
            # Get end-to-end runtime
            end_to_end_seconds = get_end_to_end_runtime(args.dataset, model_name)
            
            # Calculate estimated fit time
            fit_seconds_est = max(end_to_end_seconds - predict_seconds, 0)
            
            # Add to results
            results.append({
                "model": model_name,
                "n_test": len(X_test_eff),
                "predict_proba_seconds": predict_proba_seconds,
                "predict_seconds": predict_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "fit_seconds_est": fit_seconds_est
            })
            
            print(f"  Predict proba: {predict_proba_seconds:.4f} seconds")
            print(f"  Predict: {predict_seconds:.4f} seconds "
                  f"(n_test={len(X_test_eff):,})")
            print(f"  End-to-end: {end_to_end_seconds:.4f} seconds")
            print(f"  Estimated fit: {fit_seconds_est:.4f} seconds")
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    metrics_dir = OUTPUT_DIR / "metrics"
    ensure_dir_exists(metrics_dir)
    output_file = metrics_dir / "fit_predict_times.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\nSummary:")
    summary_df = results_df[["model", "predict_seconds", "fit_seconds_est", "end_to_end_seconds"]]
    summary_df = summary_df.rename(columns={
        "predict_seconds": "predict_time",
        "fit_seconds_est": "fit_time_est",
        "end_to_end_seconds": "total_time"
    })
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
