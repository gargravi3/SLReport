#!/usr/bin/env python
"""
Main script to run the complete hotel booking cancellation prediction pipeline.

This script:
1. Loads and preprocesses the hotel bookings dataset
2. Runs all model pipelines (Decision Tree, KNN, SVM, Neural Network)
3. Generates comparison plots and metrics
4. Saves all results

Usage:
    python src/run_all.py [--models MODEL1,MODEL2,...] [--no-calibration] [--no-save] [--seed SEED]

Example:
    python src/run_all.py --models dt,knn,svm,nn
    python src/run_all.py --models dt,knn --no-calibration
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path if needed
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RANDOM_SEED, CV_TEST_SIZE
from src.paths import ensure_dir_exists, get_output_path, get_figure_path
from src.data import load_dataset, preprocess_dataset
from src.utils import (
    compare_models,
    plot_model_comparison,
    set_seed,
    use_non_interactive_backend,
)

# Import model pipelines
from src.models.train_decision_tree import run_decision_tree_pipeline
from src.models.train_knn import run_knn_pipeline
from src.models.train_svm import run_svm_pipeline
from src.models.train_nn import run_nn_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hotel booking cancellation prediction pipeline"
    )
    
    parser.add_argument(
        "--models", type=str, default="dt,knn,svm,nn",
        help="Comma-separated list of models to run (dt,knn,svm,nn)"
    )
    
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Disable probability calibration"
    )
    
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable saving of models, metrics, and plots"
    )
    
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )

    # NEW FLAGS ------------------------------------------------------------ #
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip exploratory data analysis during preprocessing"
    )
    parser.add_argument(
        "--skip-curves", action="store_true",
        help="Skip generation of learning/complexity curves inside model pipelines"
    )
    parser.add_argument(
        "--cv", type=int, default=None,
        help="Number of CV splits to use in GridSearch (default: config value). "
             "If omitted, each model will use its internal default."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs for sklearn (default: -1 = use all cores)"
    )
    
    return parser.parse_args()


def run_pipeline(args):
    """
    Run the complete machine learning pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    set_seed(args.seed)
    # Ensure matplotlib uses a headless-safe backend before any plotting
    use_non_interactive_backend()
    
    # Parse models to run
    models_to_run = [m.strip().lower() for m in args.models.split(",")]
    calibrate = not args.no_calibration
    save_results = not args.no_save
    skip_eda = args.skip_eda
    skip_curves = args.skip_curves
    
    print("\n" + "="*80)
    print(f"HOTEL BOOKING CANCELLATION PREDICTION PIPELINE")
    print("="*80)
    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"Calibration: {'Enabled' if calibrate else 'Disabled'}")
    print(f"Save results: {'Enabled' if save_results else 'Disabled'}")
    print(f"Skip EDA: {'Yes' if skip_eda else 'No'}")
    print(f"Skip Curves: {'Yes' if skip_curves else 'No'}")
    print(f"CV Splits: {args.cv if args.cv else 'Model default'}")
    print(f"n_jobs: {args.n_jobs}")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    start_time = time.time()
    
    # Load dataset
    df = load_dataset()
    
    # Preprocess dataset
    X_train, X_test, y_train, y_test, feature_types = preprocess_dataset(
        df, do_eda=not skip_eda
    )
    
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Target distribution: {y_train.mean():.2%} positive rate (train), "
          f"{y_test.mean():.2%} positive rate (test)")
    
    # Step 2: Run model pipelines
    results = {}
    model_names = []

    # Construct CV splitter object if user specified --cv
    cv_obj = None
    if args.cv:
        cv_obj = StratifiedShuffleSplit(
            n_splits=args.cv, test_size=CV_TEST_SIZE, random_state=args.seed
        )
    
    # Decision Tree
    if "dt" in models_to_run:
        print("\nStep 2.1: Running Decision Tree pipeline...")
        start_time = time.time()
        
        dt_results = run_decision_tree_pipeline(
            X_train, X_test, y_train, y_test, feature_types,
            calibrate=calibrate, save_results=save_results,
            skip_curves=skip_curves, cv=cv_obj, n_jobs=args.n_jobs
        )
        
        results["decision_tree"] = dt_results
        model_names.append("decision_tree_calibrated" if calibrate else "decision_tree")
        
        print(f"Decision Tree pipeline completed in {time.time() - start_time:.2f} seconds")
    
    # k-Nearest Neighbors
    if "knn" in models_to_run:
        print("\nStep 2.2: Running k-Nearest Neighbors pipeline...")
        start_time = time.time()
        
        knn_results = run_knn_pipeline(
            X_train, X_test, y_train, y_test, feature_types,
            calibrate=calibrate, save_results=save_results,
            skip_curves=skip_curves, cv=cv_obj, n_jobs=args.n_jobs
        )
        
        results["knn"] = knn_results
        model_names.append("knn_calibrated" if calibrate else "knn")
        
        print(f"k-Nearest Neighbors pipeline completed in {time.time() - start_time:.2f} seconds")
    
    # Support Vector Machine
    if "svm" in models_to_run:
        print("\nStep 2.3: Running Support Vector Machine pipeline...")
        start_time = time.time()
        
        svm_results = run_svm_pipeline(
            X_train, X_test, y_train, y_test, feature_types,
            calibrate=calibrate, save_results=save_results,
            skip_curves=skip_curves, cv=cv_obj, n_jobs=args.n_jobs
        )
        
        results["svm"] = svm_results
        model_names.extend(["linear_svm_calibrated", "rbf_svm_calibrated" if calibrate else "rbf_svm"])
        
        print(f"Support Vector Machine pipeline completed in {time.time() - start_time:.2f} seconds")
    
    # Neural Network
    if "nn" in models_to_run:
        print("\nStep 2.4: Running Neural Network pipeline...")
        start_time = time.time()
        
        nn_results = run_nn_pipeline(
            X_train, X_test, y_train, y_test, feature_types,
            calibrate=calibrate, save_results=save_results,
            skip_curves=skip_curves, cv=cv_obj, n_jobs=args.n_jobs
        )
        
        results["nn"] = nn_results
        model_names.append("nn_calibrated" if calibrate else "nn")
        
        print(f"Neural Network pipeline completed in {time.time() - start_time:.2f} seconds")
    
    # Step 3: Generate comparison plots and metrics
    if len(model_names) > 1:
        print("\nStep 3: Generating model comparisons...")
        
        # Compare models
        comparison_df = compare_models(model_names)
        
        # Plot comparison
        plot_model_comparison(model_names)
        
        # Save comparison to CSV
        if save_results:
            comparison_path = get_output_path("model_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Model comparison saved to: {comparison_path}")
        
        # Print comparison table
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
    
    print("\nPipeline completed successfully!")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run pipeline
    results = run_pipeline(args)
