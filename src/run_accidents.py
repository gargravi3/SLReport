#!/usr/bin/env python
"""
Main script to run the complete US Accidents severity prediction pipeline.

This script:
1. Loads and preprocesses the US Accidents dataset
2. Runs all model pipelines (Decision Tree, KNN, SVM, Neural Network)
3. Generates comparison plots and metrics
4. Saves all results to the accidents-specific output directory

Usage:
    python src/run_accidents.py [--models MODEL1,MODEL2,...] [--no-calibration] [--no-save] [--seed SEED]

Example:
    python src/run_accidents.py --models dt,knn,svm,nn
    python src/run_accidents.py --models dt,knn --no-calibration
"""

# Set dataset name environment variable BEFORE importing config
import os
os.environ['DATASET_NAME'] = 'accidents'

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Add project root to path if needed
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import config-dependent modules
from src.config import RANDOM_SEED, CV_TEST_SIZE, OUTPUT_DIR
from src.paths import ensure_dir_exists, get_output_path, get_figure_path
from src.data_accidents import load_accidents, preprocess_accidents
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


def stratified_sample_n(X: pd.DataFrame, y: np.ndarray, n: int, seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Take a stratified sample of n rows from X and y.
    
    Args:
        X: Feature DataFrame
        y: Target array
        n: Number of samples to take
        seed: Random seed
        
    Returns:
        Tuple of (X_sampled, y_sampled)
    """
    if n >= len(X):
        return X, y
    
    X_sampled, _, y_sampled, _ = train_test_split(
        X, y, 
        train_size=n,
        random_state=seed,
        stratify=y
    )
    
    return X_sampled, y_sampled


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run US Accidents severity prediction pipeline"
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
    
    # Add large-dataset control arguments
    parser.add_argument(
        "--cap-dt-train", type=int, default=1000000,
        help="Maximum number of training samples for Decision Tree (default: 1,000,000)"
    )
    
    parser.add_argument(
        "--cap-knn-train", type=int, default=250000,
        help="Maximum number of training samples for KNN (default: 250,000)"
    )
    
    parser.add_argument(
        "--cap-knn-test", type=int, default=25000,
        help="Maximum number of test samples for KNN prediction (default: 25,000)"
    )
    
    parser.add_argument(
        "--knn-test-batch-size", type=int, default=5000,
        help="Batch size for KNN test predictions (default: 5,000)"
    )
    
    parser.add_argument(
        "--cap-svm-linear-train", type=int, default=1000000,
        help="Maximum number of training samples for Linear SVM (default: 1,000,000)"
    )
    
    parser.add_argument(
        "--cap-svm-rbf-train", type=int, default=100000,
        help="Maximum number of training samples for RBF SVM (default: 100,000)"
    )
    
    return parser.parse_args()


def run_pipeline(args):
    """
    Run the complete machine learning pipeline for US Accidents.
    
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
    print(f"US ACCIDENTS SEVERITY PREDICTION PIPELINE")
    print("="*80)
    print(f"Dataset: US Accidents (binary severity classification)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"Calibration: {'Enabled' if calibrate else 'Disabled'}")
    print(f"Save results: {'Enabled' if save_results else 'Disabled'}")
    print(f"Skip EDA: {'Yes' if skip_eda else 'No'}")
    print(f"Skip Curves: {'Yes' if skip_curves else 'No'}")
    print(f"CV Splits: {args.cv if args.cv else 'Model default'}")
    print(f"n_jobs: {args.n_jobs}")
    print(f"Random seed: {args.seed}")
    print("\nTraining caps:")
    print(f"  Decision Tree: {args.cap_dt_train:,} samples")
    print(f"  KNN: {args.cap_knn_train:,} samples (train), {args.cap_knn_test:,} samples (test)")
    print(f"  Linear SVM: {args.cap_svm_linear_train:,} samples")
    print(f"  RBF SVM: {args.cap_svm_rbf_train:,} samples")
    print(f"  KNN batch size: {args.knn_test_batch_size:,} samples")
    print("="*80 + "\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing US Accidents data...")
    start_time = time.time()
    
    # Load dataset
    df = load_accidents()
    
    # Preprocess dataset using accidents-specific function
    X_train, X_test, y_train, y_test, feature_types = preprocess_accidents(
        df, do_eda=not skip_eda
    )
    
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Target distribution: {y_train.mean():.2%} severe accidents (train), "
          f"{y_test.mean():.2%} severe accidents (test)")
    
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
        
        # Apply training cap for Decision Tree
        X_train_dt, y_train_dt = X_train, y_train
        if len(X_train) > args.cap_dt_train:
            print(f"Sampling training data for Decision Tree: {len(X_train):,} → {args.cap_dt_train:,} samples")
            X_train_dt, y_train_dt = stratified_sample_n(X_train, y_train, args.cap_dt_train, args.seed)
            print(f"Sampled training set: {len(X_train_dt):,} samples, positive rate: {y_train_dt.mean():.4f}")
        
        dt_results = run_decision_tree_pipeline(
            X_train_dt, X_test, y_train_dt, y_test, feature_types,
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
        
        # Apply training cap for KNN
        X_train_knn, y_train_knn = X_train, y_train
        if len(X_train) > args.cap_knn_train:
            print(f"Sampling training data for KNN: {len(X_train):,} → {args.cap_knn_train:,} samples")
            X_train_knn, y_train_knn = stratified_sample_n(X_train, y_train, args.cap_knn_train, args.seed)
            print(f"Sampled training set: {len(X_train_knn):,} samples, positive rate: {y_train_knn.mean():.4f}")
        
        # Apply test cap for KNN
        X_test_knn, y_test_knn = X_test, y_test
        if len(X_test) > args.cap_knn_test:
            print(f"Sampling test data for KNN: {len(X_test):,} → {args.cap_knn_test:,} samples")
            X_test_knn, y_test_knn = stratified_sample_n(X_test, y_test, args.cap_knn_test, args.seed)
            print(f"Sampled test set: {len(X_test_knn):,} samples, positive rate: {y_test_knn.mean():.4f}")
        
        knn_results = run_knn_pipeline(
            X_train_knn, X_test_knn, y_train_knn, y_test_knn, feature_types,
            calibrate=calibrate, save_results=save_results,
            skip_curves=skip_curves, cv=cv_obj, n_jobs=args.n_jobs,
            test_batch_size=args.knn_test_batch_size
        )
        
        results["knn"] = knn_results
        model_names.append("knn_calibrated" if calibrate else "knn")
        
        print(f"k-Nearest Neighbors pipeline completed in {time.time() - start_time:.2f} seconds")
    
    # Support Vector Machine
    if "svm" in models_to_run:
        print("\nStep 2.3: Running Support Vector Machine pipeline...")
        start_time = time.time()
        
        # Apply training cap for Linear SVM
        X_train_svm_linear, y_train_svm_linear = X_train, y_train
        if len(X_train) > args.cap_svm_linear_train:
            print(f"Sampling training data for Linear SVM: {len(X_train):,} → {args.cap_svm_linear_train:,} samples")
            X_train_svm_linear, y_train_svm_linear = stratified_sample_n(
                X_train, y_train, args.cap_svm_linear_train, args.seed
            )
            print(f"Sampled training set: {len(X_train_svm_linear):,} samples, positive rate: {y_train_svm_linear.mean():.4f}")
        
        # Apply training cap for RBF SVM (note: train_svm.py also has internal capping)
        X_train_svm_rbf, y_train_svm_rbf = X_train, y_train
        if len(X_train) > args.cap_svm_rbf_train:
            print(f"Sampling training data for RBF SVM: {len(X_train):,} → {args.cap_svm_rbf_train:,} samples")
            X_train_svm_rbf, y_train_svm_rbf = stratified_sample_n(
                X_train, y_train, args.cap_svm_rbf_train, args.seed
            )
            print(f"Sampled training set: {len(X_train_svm_rbf):,} samples, positive rate: {y_train_svm_rbf.mean():.4f}")
        
        # Use the smaller of the two SVM datasets
        X_train_svm = X_train_svm_linear if len(X_train_svm_linear) <= len(X_train_svm_rbf) else X_train_svm_rbf
        y_train_svm = y_train_svm_linear if len(X_train_svm_linear) <= len(X_train_svm_rbf) else y_train_svm_rbf
        
        svm_results = run_svm_pipeline(
            X_train_svm, X_test, y_train_svm, y_test, feature_types,
            calibrate=calibrate, save_results=save_results,
            skip_curves=skip_curves, cv=cv_obj, n_jobs=args.n_jobs,
            use_sgd=True  # Use SGDClassifier for large datasets
        )
        
        results["svm"] = svm_results
        model_names.extend(["linear_svm_sgd_calibrated", "rbf_svm_calibrated" if calibrate else "rbf_svm"])
        
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
    
    print("\nUS Accidents pipeline completed successfully!")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run pipeline
    results = run_pipeline(args)
