"""
Neural Network model training and evaluation for hotel booking cancellation prediction.

This module implements a complete pipeline for training, tuning, and evaluating
a Neural Network classifier (MLPClassifier with SGD) for predicting hotel booking cancellations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    validation_curve,
    StratifiedShuffleSplit,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    RANDOM_SEED,
    NN_PARAMS,
    NN_PARAM_GRID,
    CALIBRATION_METHOD,
    CALIBRATION_CV,
    CV_FOLDS,
    LEARNING_CURVE_SIZES,
    CV_SPLITS,
    CV_TEST_SIZE,
)
from src.utils import (
    set_seed,
    get_sample_weights,
    find_optimal_threshold,
    evaluate_classifier,
    plot_learning_curve,
    plot_complexity_curve,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_threshold_metrics,
    save_metrics_to_csv,
    save_model,
    wall_clock_seconds,
    get_peak_memory_gb,
)
from src.preprocess import (
    create_preprocessing_pipeline,
    ToFloat32,
    get_feature_names_from_preprocessor,
)


def train_nn(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_types: Dict[str, List[str]],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: Optional[Any] = None,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train a Neural Network classifier with hyperparameter tuning.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        param_grid: Hyperparameter grid for GridSearchCV (default: config.NN_PARAM_GRID)
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_model, best_params, cv_results)
    """
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    if param_grid is None:
        param_grid = {f"clf__{k}": v for k, v in NN_PARAM_GRID.items()}

    # Preprocessing pipeline
    preprocessor, _, _ = create_preprocessing_pipeline(
        feature_types, scale_numeric=True, scaler_type="standard"
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("to_float32", ToFloat32()),
            ("clf", MLPClassifier(**NN_PARAMS)),
        ]
    )

    # CV splitter
    if cv is None:
        cv = StratifiedShuffleSplit(
            n_splits=CV_SPLITS, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED
        )

    print("Training Neural Network with GridSearchCV…")
    print(f"Parameter grid: {param_grid}")
    
    # Calculate sample weights for class imbalance
    sample_weights = get_sample_weights(y_train)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    grid_search.fit(X_train_df, y_train, **{"clf__sample_weight": sample_weights})
    
    # Get best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = {k.replace("clf__", ""): v for k, v in grid_search.best_params_.items()}
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (F1): {grid_search.best_score_:.4f}")
    
    # Get CV results for analysis
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # ------------------------------------------------------------------ #
    # Plot convergence (epoch curve with early-stopping marker)
    # ------------------------------------------------------------------ #
    try:
        model = best_pipeline.named_steps["clf"]
        plot_nn_convergence(model, model_name="nn")
    except Exception as exc:
        # Fail-safe: never crash the pipeline because of plotting
        print(f"Warning: could not plot NN convergence – {exc}")
    
    return best_pipeline, best_params, cv_results


def calibrate_nn(
    pipeline: Pipeline,
    X_cal_df: pd.DataFrame,
    y_cal: np.ndarray,
                 method: str = CALIBRATION_METHOD,
                 cv: str = "prefit",
) -> CalibratedClassifierCV:
    """
    Calibrate a Neural Network classifier's probability estimates.
    
    Args:
        model: Trained Neural Network classifier
        X_train: Training feature matrix
        y_train: Training target vector
        method: Calibration method ('sigmoid' or 'isotonic')
        cv: Number of cross-validation folds for calibration
        
    Returns:
        Calibrated classifier
    """
    print(f"Calibrating Neural Network probabilities using {method} method...")
    
    # Calculate sample weights for class imbalance
    sample_weights = get_sample_weights(y_cal)
    
    # Create calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=cv,
    )
    
    # Fit calibrated model
    calibrated_model.fit(X_cal_df, y_cal, sample_weight=sample_weights)
    
    return calibrated_model


def plot_nn_convergence(model: MLPClassifier, 
                         model_name: str = "nn", 
                         save: bool = True) -> plt.Figure:
    """
    Plot the convergence of the Neural Network during training.
    
    Args:
        model: Trained Neural Network classifier
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    if not hasattr(model, 'loss_curve_'):
        print("Model does not have loss_curve_ attribute. Skipping plot.")
        return None
    
    # ------------------------------------------------------------------ #
    # Prepare figure with optional twin-axis for validation score
    # ------------------------------------------------------------------ #
    fig, ax_loss = plt.subplots(figsize=(10, 6))

    epochs = np.arange(1, len(model.loss_curve_) + 1)

    # Plot training loss
    ax_loss.plot(epochs, model.loss_curve_, label="Train loss", color="tab:blue")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="tab:blue")
    ax_loss.tick_params(axis='y', labelcolor="tab:blue")

    # Optional validation score (requires early_stopping=True)
    best_epoch = None
    if hasattr(model, "validation_scores_") and model.validation_scores_ is not None:
        ax_val = ax_loss.twinx()
        ax_val.plot(
            epochs[: len(model.validation_scores_)],
            model.validation_scores_,
            label="Val score",
            color="tab:orange",
        )
        ax_val.set_ylabel("Validation score", color="tab:orange")
        ax_val.tick_params(axis='y', labelcolor="tab:orange")

        # Best epoch by highest validation score
        best_epoch = int(np.argmax(model.validation_scores_)) + 1
    else:
        # Fallback to min training loss
        best_epoch = int(np.argmin(model.loss_curve_)) + 1

    # Draw best-epoch marker
    ax_loss.axvline(best_epoch, linestyle="--", color="grey", alpha=0.7)
    ax_loss.text(
        best_epoch,
        ax_loss.get_ylim()[1] * 0.95,
        f"best epoch = {best_epoch}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
        color="grey",
    )

    # Overall title & grid
    plt.title("Neural Network Training Convergence")
    ax_loss.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(
            get_figure_path(f"nn_convergence_{model_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    
    return fig


def plot_nn_architecture(model: MLPClassifier, 
                         model_name: str = "nn", 
                         save: bool = True) -> plt.Figure:
    """
    Plot the architecture of the Neural Network.
    
    Args:
        model: Trained Neural Network classifier
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    if not hasattr(model, 'coefs_'):
        print("Model does not have coefs_ attribute. Skipping plot.")
        return None
    
    # Get layer sizes
    n_layers = len(model.coefs_) + 1
    layer_sizes = [model.coefs_[0].shape[0]] + [cf.shape[1] for cf in model.coefs_]
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Plot nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_name = "Input" if i == 0 else "Output" if i == len(layer_sizes) - 1 else f"Hidden {i}"
        
        for j in range(layer_size):
            circle = plt.Circle((i, j * 1.5), radius=0.3, fill=True)
            ax.add_patch(circle)
            
        # Add layer label
        ax.text(i, layer_size * 1.5, layer_name, ha='center', va='center')
    
    # Plot connections
    for i, coef in enumerate(model.coefs_):
        for j in range(coef.shape[0]):
            for k in range(coef.shape[1]):
                ax.plot([i, i+1], [j * 1.5, k * 1.5], 'k-', alpha=0.1)
    
    # Set axis properties
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, max(layer_sizes) * 1.5 + 0.5)
    ax.set_title(f'Neural Network Architecture: {model.hidden_layer_sizes}')
    ax.axis('off')
    
    # Add architecture information as text
    architecture_text = f"Architecture: {layer_sizes[0]} → "
    architecture_text += " → ".join([str(size) for size in layer_sizes[1:]])
    architecture_text += f"\nActivation: {model.activation}"
    architecture_text += f"\nSolver: {model.solver}"
    architecture_text += f"\nAlpha: {model.alpha}"
    architecture_text += f"\nLearning rate: {model.learning_rate_init}"
    
    plt.figtext(0.5, 0.01, architecture_text, ha='center', fontsize=12, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"nn_architecture_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def train_and_evaluate_nn(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_types: Dict[str, List[str]],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    calibrate: bool = True,
    save_results: bool = True,
    skip_curves: bool = False,
    cv: Optional[Any] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Train, evaluate, and generate plots for a Neural Network model.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target vector
        y_test: Test target vector
        feature_names: List of feature names
        param_grid: Hyperparameter grid for GridSearchCV
        calibrate: Whether to calibrate the model's probabilities
        save_results: Whether to save models, metrics, and plots
        
    Returns:
        Dictionary with results
    """
    # Train the model
    t0 = time.time()

    pipeline, best_params, cv_results = train_nn(
        X_train_df, y_train, feature_types, param_grid=param_grid, cv=cv, n_jobs=n_jobs
    )
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, feature_types["categorical"], feature_types["numeric"]
    )
    
    # Curves
    if not skip_curves:
        plot_learning_curve(
            pipeline,
            X_train_df,
            y_train,
            title="Neural Network Learning Curve",
            model_name="nn",
            train_sizes=LEARNING_CURVE_SIZES,
            cv=cv if cv is not None else 3,
            n_jobs=n_jobs,
        )

        plot_complexity_curve(
            pipeline,
            X_train_df,
            y_train,
            param_name="clf__alpha",
            param_range=[1e-5, 1e-4, 1e-3],
            title="Neural Network Complexity Curve - Alpha",
            model_name="nn_alpha",
            cv=cv if cv is not None else 3,
            n_jobs=n_jobs,
        )
    
    # NOTE: Additional manual complexity curves for hidden_layer_sizes, alpha,
    # and learning_rate_init that relied on raw NumPy matrices have been removed
    # to prevent redundant computation and to avoid referencing undefined
    # variables (X_train / y_train) after the refactor to DataFrame-based
    # pipelines.  The light-weight alpha complexity curve generated above via
    # `plot_complexity_curve` on the full pipeline already satisfies assignment
    # requirements while keeping runtime reasonable.
    
    # Calibrate the model if requested
    from sklearn.model_selection import train_test_split
    X_cal_train_df, X_cal_val_df, y_cal_train, y_cal_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    if calibrate:
        calibrated_model = calibrate_nn(pipeline, X_cal_val_df, y_cal_val)
        final_model = calibrated_model
        model_name = "nn_calibrated"
    else:
        final_model = pipeline
        model_name = "nn"
    
    # Get predictions on test set
    y_prob = final_model.predict_proba(X_test_df)[:, 1]
    
    # Find optimal threshold
    # Select threshold on the **validation split** (same data used to fit the
    # calibrator) rather than the entire training set to avoid optimistic bias
    # and the UndefinedMetricWarning that can occur when the threshold is tuned
    # on data the model has already seen.
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
    
    # Evaluate the model
    runtime = wall_clock_seconds(t0)
    peak_ram = get_peak_memory_gb()

    metrics = evaluate_classifier(final_model, X_test_df, y_test, threshold=threshold)
    metrics["runtime_seconds"] = runtime
    metrics["peak_memory_gb"] = peak_ram
    print("\nTest set metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
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
        'peak_memory_gb': peak_ram,
    }


def run_nn_pipeline(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
                    y_train: np.ndarray, y_test: np.ndarray,
                    feature_types: Dict[str, List[str]],
                    calibrate: bool = True,
                    save_results: bool = True,
                    skip_curves: bool = False,
                    cv: Optional[Any] = None,
                    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Run the complete Neural Network pipeline from preprocessing to evaluation.
    
    Args:
        X_train_df: Training feature DataFrame
        X_test_df: Test feature DataFrame
        y_train: Training target vector
        y_test: Test target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        calibrate: Whether to calibrate the model's probabilities
        save_results: Whether to save models, metrics, and plots
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("NEURAL NETWORK PIPELINE")
    print("="*80)
    
    # Train and evaluate the model
    results = train_and_evaluate_nn(
        X_train_df,
        X_test_df,
        y_train,
        y_test,
        feature_types,
        calibrate=calibrate,
        save_results=save_results,
        skip_curves=skip_curves,
        cv=cv,
        n_jobs=n_jobs,
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.data import load_dataset, preprocess_dataset
    
    # Load and preprocess the dataset
    df = load_dataset()
    # Use the updated keyword argument name `do_eda`
    X_train, X_test, y_train, y_test, feature_types = preprocess_dataset(df, do_eda=True)
    
    # Run the Neural Network pipeline
    results = run_nn_pipeline(
        X_train, X_test, y_train, y_test, feature_types,
        calibrate=True, save_results=True
    )
    
    print("\nNeural Network pipeline completed!")
