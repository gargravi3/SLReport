"""
Decision Tree model training and evaluation for hotel booking cancellation prediction.

This module implements a complete pipeline for training, tuning, and evaluating
a Decision Tree classifier for predicting hotel booking cancellations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve, StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    RANDOM_SEED, DT_PARAMS, DT_PARAM_GRID, 
    CALIBRATION_METHOD, CALIBRATION_CV, CV_FOLDS,
    CV_SPLITS, CV_TEST_SIZE, LEARNING_CURVE_SIZES,
    DTYPE_FLOAT
)
from src.utils import (
    set_seed, calculate_class_weights, find_optimal_threshold,
    evaluate_classifier, plot_learning_curve, plot_complexity_curve,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_calibration_curve, plot_threshold_metrics, save_metrics_to_csv,
    plot_feature_importance, save_model,
    wall_clock_seconds, get_peak_memory_gb
)
from src.preprocess import (
    create_preprocessing_pipeline, apply_preprocessing, ToFloat32,
    get_feature_names_from_preprocessor
)


def train_decision_tree(X_train_df: pd.DataFrame, y_train: np.ndarray,
                        feature_types: Dict[str, List[str]],
                        param_grid: Optional[Dict[str, List[Any]]] = None,
                        cv: Optional[int] = None,
                        n_jobs: int = -1,
                        verbose: int = 1) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train a Decision Tree classifier with hyperparameter tuning.
    
    Args:
        X_train_df: Training feature DataFrame
        y_train: Training target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        param_grid: Hyperparameter grid for GridSearchCV (default: config.DT_PARAM_GRID)
        cv: Number of cross-validation folds or CV splitter
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_pipeline, best_params, cv_results)
    """
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Use default parameter grid if not provided
    if param_grid is None:
        param_grid = {f'clf__{k}': v for k, v in DT_PARAM_GRID.items()}
    
    # Create preprocessing pipeline
    preprocessor, _, _ = create_preprocessing_pipeline(
        feature_types, scale_numeric=True
    )
    
    # Create pipeline with preprocessor and classifier
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('to_float32', ToFloat32()),
        ('clf', DecisionTreeClassifier(**DT_PARAMS))
    ])
    
    # Set up cross-validation strategy
    if cv is None:
        cv = StratifiedShuffleSplit(
            n_splits=CV_SPLITS, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED
        )
    
    print(f"Training Decision Tree classifier with GridSearchCV...")
    print(f"Parameter grid: {param_grid}")
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    grid_search.fit(X_train_df, y_train)
    
    # Get best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (F1): {grid_search.best_score_:.4f}")
    
    # Get CV results for analysis
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    return best_pipeline, best_params, cv_results


def calibrate_decision_tree(pipeline: Pipeline, 
                           X_train_df: pd.DataFrame, 
                           y_train: np.ndarray,
                           method: str = CALIBRATION_METHOD,
                           cv: int = CALIBRATION_CV) -> CalibratedClassifierCV:
    """
    Calibrate a Decision Tree classifier's probability estimates.
    
    Args:
        pipeline: Trained pipeline with Decision Tree classifier
        X_train_df: Training feature DataFrame
        y_train: Training target vector
        method: Calibration method ('sigmoid' or 'isotonic')
        cv: Number of cross-validation folds for calibration
        
    Returns:
        Calibrated classifier
    """
    print(f"Calibrating Decision Tree probabilities using {method} method...")
    
    # Create calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=cv  # No n_jobs parameter
    )
    
    # Fit calibrated model
    calibrated_model.fit(X_train_df, y_train)
    
    return calibrated_model


def plot_decision_tree(pipeline: Pipeline, 
                      feature_names: List[str],
                      class_names: List[str] = ['Not Canceled', 'Canceled'],
                      max_depth: int = 3,
                      save: bool = True) -> plt.Figure:
    """
    Plot the Decision Tree structure.
    
    Args:
        pipeline: Trained pipeline with Decision Tree classifier
        feature_names: List of feature names
        class_names: List of class names
        max_depth: Maximum depth to plot (None for full tree)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Extract the tree from the pipeline
    model = pipeline.named_steps['clf']
    
    plt.figure(figsize=(20, 10))
    
    # Plot tree structure
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth
    )
    
    plt.title(f'Decision Tree Structure (max_depth={max_depth if max_depth else "None"})')
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"decision_tree_structure.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_pruning_path(X_train_df: pd.DataFrame, y_train: np.ndarray,
                     X_val_df: pd.DataFrame, y_val: np.ndarray,
                     feature_types: Dict[str, List[str]],
                     save: bool = True) -> plt.Figure:
    """
    Plot the pruning path for a Decision Tree.
    
    Args:
        X_train_df: Training feature DataFrame
        y_train: Training target vector
        X_val_df: Validation feature DataFrame
        y_val: Validation target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Create preprocessing pipeline
    preprocessor, _, _ = create_preprocessing_pipeline(
        feature_types, scale_numeric=True
    )
    
    # Preprocess the data
    X_train = preprocessor.fit_transform(X_train_df).astype(DTYPE_FLOAT)
    X_val = preprocessor.transform(X_val_df).astype(DTYPE_FLOAT)
    
    # Create a decision tree classifier
    clf = DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced')
    
    # Fit the tree
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Ensure alphas are non-negative and unique (guard against -0.0 / tiny negatives)
    ccp_alphas = np.asarray(ccp_alphas, dtype=float)
    ccp_alphas = np.clip(ccp_alphas, 0.0, None)
    ccp_alphas = np.unique(ccp_alphas)
    
    # Create trees with different alphas and evaluate them
    clfs = []
    train_scores = []
    val_scores = []
    
    # Filter out alphas that are too close to each other
    ccp_alphas = ccp_alphas[::10]  # Take every 10th value to reduce computation
    
    # Ensure we have at least 10 points
    if len(ccp_alphas) < 10:
        ccp_alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 10)]
    
    # Add 0 if not present
    if ccp_alphas[0] != 0:
        ccp_alphas = np.insert(ccp_alphas, 0, 0)
    
    print(f"Evaluating {len(ccp_alphas)} alpha values for pruning path...")
    
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=RANDOM_SEED, ccp_alpha=ccp_alpha, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # Record number of nodes and performance
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        val_scores.append(clf.score(X_val, y_val))
    
    # Get number of nodes for each tree
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth_counts = [clf.tree_.max_depth for clf in clfs]
    
    # Plot node count vs alpha
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Alpha vs nodes
    ax[0, 0].plot(ccp_alphas, node_counts, marker='o')
    ax[0, 0].set_xlabel('Alpha')
    ax[0, 0].set_ylabel('Number of nodes')
    ax[0, 0].set_title('Number of nodes vs Alpha')
    ax[0, 0].grid(True)
    
    # Plot 2: Alpha vs depth
    ax[0, 1].plot(ccp_alphas, depth_counts, marker='o')
    ax[0, 1].set_xlabel('Alpha')
    ax[0, 1].set_ylabel('Max depth')
    ax[0, 1].set_title('Tree depth vs Alpha')
    ax[0, 1].grid(True)
    
    # Plot 3: Alpha vs accuracy
    ax[1, 0].plot(ccp_alphas, train_scores, marker='o', label='train')
    ax[1, 0].plot(ccp_alphas, val_scores, marker='o', label='validation')
    ax[1, 0].set_xlabel('Alpha')
    ax[1, 0].set_ylabel('Accuracy')
    ax[1, 0].set_title('Accuracy vs Alpha')
    ax[1, 0].legend()
    ax[1, 0].grid(True)
    
    # Plot 4: Nodes vs accuracy
    ax[1, 1].plot(node_counts, train_scores, marker='o', label='train')
    ax[1, 1].plot(node_counts, val_scores, marker='o', label='validation')
    ax[1, 1].set_xlabel('Number of nodes')
    ax[1, 1].set_ylabel('Accuracy')
    ax[1, 1].set_title('Accuracy vs Number of nodes')
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path("decision_tree_pruning_path.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def train_and_evaluate_decision_tree(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                                    y_train: np.ndarray, y_test: np.ndarray,
                                    feature_types: Dict[str, List[str]],
                                    param_grid: Optional[Dict[str, List[Any]]] = None,
                                    calibrate: bool = True,
                                    save_results: bool = True,
                                    skip_curves: bool = False,
                                    cv: Optional[int] = None,
                                    n_jobs: int = -1) -> Dict[str, Any]:
    """
    Train, evaluate, and generate plots for a Decision Tree model.
    
    Args:
        X_train_df: Training feature DataFrame
        X_test_df: Test feature DataFrame
        y_train: Training target vector
        y_test: Test target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        param_grid: Hyperparameter grid for GridSearchCV
        calibrate: Whether to calibrate the model's probabilities
        save_results: Whether to save models, metrics, and plots
        skip_curves: Whether to skip generating learning and complexity curves
        cv: Number of cross-validation folds or CV splitter
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with results
    """
    # Start timing
    start_time = time.time()
    
    # Train the model
    pipeline, best_params, cv_results = train_decision_tree(
        X_train_df, y_train, feature_types, param_grid=param_grid, cv=cv, n_jobs=n_jobs
    )
    
    # Extract feature names for plotting
    preprocessor = pipeline.named_steps['preprocess']
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, feature_types['categorical'], feature_types['numeric']
    )
    
    # Split training data for calibration and pruning path
    from sklearn.model_selection import train_test_split
    X_cal_train_df, X_cal_val_df, y_cal_train, y_cal_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Plot pruning path if not skipping curves
    if not skip_curves:
        plot_pruning_path(X_cal_train_df, y_cal_train, X_cal_val_df, y_cal_val, feature_types)
        
        # Plot tree structure
        plot_decision_tree(pipeline, feature_names)
        
        # Generate learning curve
        plot_learning_curve(
            pipeline, X_train_df, y_train,
            title="Decision Tree Learning Curve",
            model_name="decision_tree",
            train_sizes=LEARNING_CURVE_SIZES,
            n_jobs=n_jobs
        )
        
        # Generate complexity curves for important hyperparameters
        for param_name, param_range in [
            ('clf__max_depth', [2, 4, 6, 8, 10, 15, 18]),
            ('clf__min_samples_split', [50, 100, 200, 400]),
            ('clf__min_samples_leaf', [1, 2, 5, 10, 20, 50, 100, 200]),
            ('clf__ccp_alpha', [0.0, 1e-4, 5e-4, 1e-3, 1e-2])
        ]:
            plot_complexity_curve(
                Pipeline([
                    ('preprocess', preprocessor),
                    ('to_float32', ToFloat32()),
                    ('clf', DecisionTreeClassifier(**DT_PARAMS))
                ]),
                X_train_df, y_train,
                param_name=param_name,
                param_range=param_range,
                title=f"Decision Tree Complexity Curve - {param_name.replace('clf__', '')}",
                model_name=f"decision_tree_{param_name.replace('clf__', '')}",
                cv=3,  # Use a smaller CV for complexity curves
                n_jobs=n_jobs
            )
    
    # Calibrate the model if requested
    if calibrate:
        calibrated_model = calibrate_decision_tree(pipeline, X_train_df, y_train)
        final_model = calibrated_model
        model_name = "decision_tree_calibrated"
    else:
        final_model = pipeline
        model_name = "decision_tree"
    
    # Get predictions on test set
    y_prob = final_model.predict_proba(X_test_df)[:, 1]
    
    # Find optimal threshold
    # Use the held-out calibration/validation split (y_cal_val) rather than the
    # entire training set to select the threshold.  This avoids optimistic bias
    # and reduces the chance of UndefinedMetricWarning when the tuned threshold
    # leads to no positive predictions on unseen data.
    threshold = find_optimal_threshold(
        y_cal_val, final_model.predict_proba(X_cal_val_df)[:, 1]
    )
    print(f"Optimal threshold: {threshold:.4f}")
    
    # Apply threshold to get binary predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob, model_name=model_name)
    
    # Plot PR curve
    plot_pr_curve(y_test, y_prob, model_name=model_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name=model_name)
    
    # Plot calibration curve
    plot_calibration_curve(y_test, y_prob, model_name=model_name)
    
    # Plot threshold metrics
    plot_threshold_metrics(y_test, y_prob, model_name=model_name)
    
    # Plot feature importance if not skipping curves
    if not skip_curves:
        model = pipeline.named_steps['clf']
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model.feature_importances_, feature_names, 
                model_name=model_name
            )
    
    # Calculate runtime and peak memory
    runtime = wall_clock_seconds(start_time)
    peak_memory_gb = get_peak_memory_gb()
    
    # Evaluate the model
    metrics = evaluate_classifier(final_model, X_test_df, y_test, threshold=threshold)
    
    # Add runtime and memory metrics
    metrics['runtime_seconds'] = runtime
    metrics['peak_memory_gb'] = peak_memory_gb
    
    print("\nTest set metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nRuntime: {runtime:.2f} seconds")
    print(f"Peak memory: {peak_memory_gb:.2f} GB")
    
    # Save results if requested
    if save_results:
        # Save model
        model_path = save_model(final_model, model_name)
        print(f"Model saved to: {model_path}")
        
        # Save metrics
        metrics_path = save_metrics_to_csv(metrics, model_name)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save CV results
        from src.paths import get_metric_path
        cv_results_path = get_metric_path(f"cv_results_{model_name}", extension=".csv")
        cv_results.to_csv(cv_results_path, index=False)
        print(f"CV results saved to: {cv_results_path}")
    
    # Return results
    return {
        'model': final_model,
        'best_params': best_params,
        'metrics': metrics,
        'threshold': threshold,
        'cv_results': cv_results,
        'runtime_seconds': runtime,
        'peak_memory_gb': peak_memory_gb,
        'feature_names': feature_names
    }


def run_decision_tree_pipeline(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                              y_train: np.ndarray, y_test: np.ndarray,
                              feature_types: Dict[str, List[str]],
                              calibrate: bool = True,
                              save_results: bool = True,
                              skip_curves: bool = False,
                              cv: Optional[int] = None,
                              n_jobs: int = -1) -> Dict[str, Any]:
    """
    Run the complete Decision Tree pipeline from preprocessing to evaluation.
    
    Args:
        X_train_df: Training feature DataFrame
        X_test_df: Test feature DataFrame
        y_train: Training target vector
        y_test: Test target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        calibrate: Whether to calibrate the model's probabilities
        save_results: Whether to save models, metrics, and plots
        skip_curves: Whether to skip generating learning and complexity curves
        cv: Number of cross-validation folds or CV splitter
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("DECISION TREE PIPELINE")
    print("="*80)
    
    # Train and evaluate the model
    results = train_and_evaluate_decision_tree(
        X_train_df, X_test_df, y_train, y_test, feature_types,
        calibrate=calibrate, save_results=save_results,
        skip_curves=skip_curves, cv=cv, n_jobs=n_jobs
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.data import load_dataset, preprocess_dataset
    
    # Load and preprocess the dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test, feature_types = preprocess_dataset(df, do_eda=True)
    
    # Run the Decision Tree pipeline
    results = run_decision_tree_pipeline(
        X_train, X_test, y_train, y_test, feature_types,
        calibrate=True, save_results=True
    )
    
    print("\nDecision Tree pipeline completed!")
