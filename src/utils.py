"""
Utility functions for the hotel booking cancellation prediction project.

This module provides utility functions for:
- Setting random seeds for reproducibility
- Plotting learning curves, ROC curves, PR curves, etc.
- Computing and optimizing thresholds
- Computing class weights
- Other helper functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
# --------------------------------------------------------------------------- #
# Std-lib imports for resource logging (added per assignment guidance)
# --------------------------------------------------------------------------- #
import warnings
import time
import resource
import platform
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    auc, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from pathlib import Path
import random

from src.config import (
    RANDOM_SEED, TARGET_COLUMN, POSITIVE_CLASS, NEGATIVE_CLASS,
    POSITIVE_RATE, PR_AUC_BASELINE, FIGURE_DIR, THRESHOLD_METRIC
)

# --------------------------------------------------------------------------- #
# Warning filters
# --------------------------------------------------------------------------- #
# Silence the very common FutureWarning emitted by scikit-learn when a Pipeline
# is cloned (e.g., during cross-validation) and therefore not yet fitted.  This
# warning is slated to become an error in sklearn 1.8, but until then it clutters
# output without providing actionable information in our workflow.
warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
    module="sklearn.pipeline",
)


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #

def use_non_interactive_backend() -> None:
    """
    Force matplotlib to use a non-interactive backend for headless environments.

    This prevents errors when the default backend requires a display
    (e.g., TkAgg) which is unavailable on many servers/CI runners.
    Call this once early in the program (before any plotting) or import
    this module prior to other matplotlib usage.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)


# --------------------------------------------------------------------------- #
# Runtime / memory helpers (new)
# --------------------------------------------------------------------------- #

def wall_clock_seconds(start_time: float) -> float:
    """
    Return wall-clock time elapsed since ``start_time`` in seconds.

    Usage
    -----
    >>> t0 = time.time()
    >>> ...  # code
    >>> elapsed = wall_clock_seconds(t0)
    """
    return time.time() - start_time


def get_peak_memory_gb() -> float:
    """
    Return peak resident set size (RSS) of the current process in **gigabytes**.

    Notes
    -----
    * On Linux `ru_maxrss` is reported in **kilobytes**.
    * On macOS / BSD it is reported in **bytes**.
    The function normalises to gigabytes so callers don't worry about platform
    differences.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system = platform.system()

    if system == "Darwin":  # macOS reports bytes
        bytes_used = usage
    else:  # Assume Linux/Unix – ru_maxrss in KiB
        bytes_used = usage * 1024

    return bytes_used / (1024 ** 3)  # GiB



def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility across multiple libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class labels to weights
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def get_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Convert class weights to sample weights for models that don't support class_weight.
    
    Args:
        y: Target labels
        
    Returns:
        Array of sample weights corresponding to each instance
    """
    class_weights = calculate_class_weights(y)
    return np.array([class_weights[label] for label in y])


def save_model(model: BaseEstimator, model_name: str, directory: str = None) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        model_name: Name to give the saved model
        directory: Directory to save the model in (default: config.MODEL_DIR)
        
    Returns:
        Path to the saved model
    """
    from src.paths import get_model_path
    
    path = get_model_path(model_name)
    joblib.dump(model, path)
    return str(path)


def load_model(model_name: str, directory: str = None) -> BaseEstimator:
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model to load
        directory: Directory to load the model from (default: config.MODEL_DIR)
        
    Returns:
        Loaded model
    """
    from src.paths import get_model_path
    
    path = get_model_path(model_name)
    return joblib.load(path)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                           metric: str = THRESHOLD_METRIC) -> float:
    """
    Find the optimal threshold for binary classification based on a given metric.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Optimal threshold value
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx]


def evaluate_classifier(model: BaseEstimator, X: np.ndarray, y_true: np.ndarray, 
                        threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Evaluate a binary classifier and return various metrics.
    
    Args:
        model: Trained classifier with predict_proba method
        X: Feature matrix
        y_true: True binary labels
        threshold: Classification threshold (if None, use default 0.5)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predicted probabilities for the positive class
    y_prob = model.predict_proba(X)[:, 1]
    
    # Apply threshold if provided
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        # zero_division=0 silences warnings when a class is missing in
        # predictions (common with very high/low thresholds).  It returns 0
        # instead of raising or emitting UndefinedMetricWarning.
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob),
        'threshold': threshold if threshold is not None else 0.5
    }
    
    return metrics


def plot_learning_curve(model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                        cv: int = 5, n_jobs: int = -1, 
                        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
                        scoring: str = "average_precision",
                        title: str = "Learning Curve", 
                        model_name: str = "model", 
                        save: bool = True) -> plt.Figure:
    """
    Plot learning curve for a model.
    
    Args:
        model: Estimator to use
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel
        train_sizes: Array of training sizes to try
        scoring: Scoring metric to use for learning curve (default: 'average_precision')
        title: Plot title
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring
    )
    
    # Calculate mean and std for training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Plot standard deviation bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    # Add labels and title
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"learning_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_complexity_curve(model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                          param_name: str, param_range: List[Any], 
                          cv: int = 5, n_jobs: int = -1,
                          title: str = "Complexity Curve", 
                          model_name: str = "model", 
                          save: bool = True) -> plt.Figure:
    """
    Plot model complexity curve for a specific hyperparameter.
    
    Args:
        model: Estimator to use
        X: Feature matrix
        y: Target vector
        param_name: Name of the parameter to vary
        param_range: Range of parameter values to try
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel
        title: Plot title
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate validation curve
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=n_jobs, scoring='f1'
    )
    
    # Calculate mean and std for training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Use numerical x positions so non-numeric params (e.g., None, strings) don't break plotting
    x_vals = np.arange(len(param_range))
    
    # Plot validation curve using numeric x positions
    plt.plot(x_vals, train_mean, 'o-', color='r', label='Training score')
    plt.plot(x_vals, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Plot standard deviation bands
    plt.fill_between(x_vals, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(x_vals, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    # Add labels and title
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Set x-tick labels to the original parameter values for readability
    plt.xticks(x_vals, [str(v) for v in param_range], rotation=45)
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"complexity_curve_{param_name}_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   model_name: str = "model", 
                   save: bool = True) -> plt.Figure:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(8, 8))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"roc_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  model_name: str = "model", 
                  save: bool = True) -> plt.Figure:
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(8, 8))
    
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Plot PR curve
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Plot baseline (no-skill classifier)
    plt.plot([0, 1], [POSITIVE_RATE, POSITIVE_RATE], 'k--', lw=2, 
             label=f'Baseline (AUC = {PR_AUC_BASELINE:.3f})')
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"pr_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str = "model", 
                          save: bool = True) -> plt.Figure:
    """
    Plot confusion matrix for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Add class labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks + 0.5, ['Not Canceled', 'Canceled'])
    plt.yticks(tick_marks + 0.5, ['Not Canceled', 'Canceled'])
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"confusion_matrix_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                           model_name: str = "model", 
                           n_bins: int = 10,
                           save: bool = True) -> plt.Figure:
    """
    Plot calibration curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model (for saving)
        n_bins: Number of bins for calibration curve
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Plot calibration curve
    plt.plot(prob_pred, prob_true, 's-', label=model_name)
    
    # Plot perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Add labels and title
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Calculate Brier score
    brier = brier_score_loss(y_true, y_prob)
    plt.text(0.05, 0.95, f'Brier Score: {brier:.3f}', 
             horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"calibration_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_importance(feature_importances: np.ndarray, feature_names: List[str], 
                            model_name: str = "model", 
                            top_n: int = 20,
                            save: bool = True) -> plt.Figure:
    """
    Plot feature importance for a model.
    
    Args:
        feature_importances: Array of feature importance scores
        feature_names: List of feature names
        model_name: Name of the model (for saving)
        top_n: Number of top features to show
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # Sort by importance and take top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    
    # Add labels and title
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"feature_importance_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                           model_name: str = "model", 
                           save: bool = True) -> plt.Figure:
    """
    Plot precision, recall, and F1 score across different thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for the positive class
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision_scores.append(precision_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    # Find optimal threshold for F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot metrics
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    
    # Mark optimal threshold
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    
    # Add labels and title
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision, Recall, and F1 Score vs. Threshold - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"threshold_metrics_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def save_metrics_to_csv(metrics: Dict[str, float], model_name: str) -> str:
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        
    Returns:
        Path to the saved CSV file
    """
    from src.paths import get_metric_path
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save to CSV
    path = get_metric_path(f"metrics_{model_name}")
    metrics_df.to_csv(path, index=False)
    
    return str(path)


def compare_models(model_names: List[str]) -> pd.DataFrame:
    """
    Compare metrics across multiple models.
    
    Args:
        model_names: List of model names to compare
        
    Returns:
        DataFrame with comparison metrics
    """
    from src.paths import get_metric_path
    
    # Load metrics for each model
    all_metrics = []
    for model_name in model_names:
        path = get_metric_path(f"metrics_{model_name}")
        if os.path.exists(path):
            metrics = pd.read_csv(path)
            metrics['model'] = model_name
            all_metrics.append(metrics)
    
    # Combine metrics
    if all_metrics:
        return pd.concat(all_metrics, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_model_comparison(model_names: List[str], metrics: List[str] = None,
                          save: bool = True) -> plt.Figure:
    """
    Plot comparison of metrics across multiple models.
    
    Args:
        model_names: List of model names to compare
        metrics: List of metrics to compare (default: all available)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Get comparison metrics
    comparison_df = compare_models(model_names)
    
    if comparison_df.empty:
        print("No metrics found for the specified models.")
        return None
    
    # Select metrics to compare
    if metrics is None:
        # Exclude 'model' and any non-numeric columns
        metrics = [col for col in comparison_df.columns 
                  if col != 'model' and np.issubdtype(comparison_df[col].dtype, np.number)]
    
    # Melt DataFrame for easier plotting
    melted_df = pd.melt(comparison_df, id_vars=['model'], value_vars=metrics,
                        var_name='Metric', value_name='Value')
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='model', data=melted_df)
    
    # Add labels and title
    plt.title('Model Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path("model_comparison.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()
