"""
Support Vector Machine model training and evaluation for hotel booking cancellation prediction.

This module implements a complete pipeline for training, tuning, and evaluating
SVM classifiers (both Linear and RBF kernels) for predicting hotel booking cancellations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    validation_curve,
    StratifiedShuffleSplit,
    train_test_split,
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
    SVM_LINEAR_PARAMS,
    SVM_LINEAR_PARAM_GRID,
    SVM_RBF_PARAMS,
    SVM_RBF_PARAM_GRID,
    CALIBRATION_METHOD,
    CALIBRATION_CV,
    CV_FOLDS,
    CV_SPLITS,
    CV_TEST_SIZE,
    LEARNING_CURVE_SIZES,
    DTYPE_FLOAT,
)
from src.utils import (
    set_seed, calculate_class_weights, find_optimal_threshold,
    evaluate_classifier, plot_learning_curve, plot_complexity_curve,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_calibration_curve, plot_threshold_metrics, save_metrics_to_csv,
    save_model, wall_clock_seconds, get_peak_memory_gb
)
from src.preprocess import (
    create_preprocessing_pipeline, ToFloat32, get_feature_names_from_preprocessor
)


def train_linear_svm(X_train_df: pd.DataFrame, y_train: np.ndarray,
                     feature_types: Dict[str, List[str]],
                     param_grid: Optional[Dict[str, List[Any]]] = None,
                     cv: Optional[Any] = None,
                     n_jobs: int = -1,
                     verbose: int = 1) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train a Linear SVM classifier with hyperparameter tuning.
    
    Args:
        X_train_df: Training feature DataFrame
        y_train: Training target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        param_grid: Hyperparameter grid for GridSearchCV (default: config.SVM_LINEAR_PARAM_GRID)
        cv: Cross-validation strategy (default: StratifiedShuffleSplit)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_pipeline, best_params, cv_results)
    """
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Use default parameter grid if not provided
    if param_grid is None:
        param_grid = {f'clf__{k}': v for k, v in SVM_LINEAR_PARAM_GRID.items()}
    
    # Create preprocessing pipeline
    preprocessor, _, _ = create_preprocessing_pipeline(
        feature_types, scale_numeric=True, scaler_type='standard'
    )
    
    # Create pipeline with preprocessor and classifier
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('to_float32', ToFloat32()),
        ('clf', LinearSVC(**SVM_LINEAR_PARAMS))
    ])
    
    # Set up cross-validation strategy
    if cv is None:
        cv = StratifiedShuffleSplit(
            n_splits=CV_SPLITS, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED
        )
    
    print(f"Training Linear SVM classifier with GridSearchCV...")
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


def train_linear_svm_sgd(X_train_df: pd.DataFrame, y_train: np.ndarray,
                         feature_types: Dict[str, List[str]],
                         param_grid: Optional[Dict[str, List[Any]]] = None,
                         cv: Optional[Any] = None,
                         n_jobs: int = -1,
                         verbose: int = 1) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train a Linear SVM classifier using SGDClassifier for large datasets.
    
    Args:
        X_train_df: Training feature DataFrame
        y_train: Training target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        param_grid: Hyperparameter grid for GridSearchCV
        cv: Cross-validation strategy (default: StratifiedShuffleSplit)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_pipeline, best_params, cv_results)
    """
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Default SGD parameter grid
    if param_grid is None:
        param_grid = {
            'clf__alpha': [1e-5, 1e-4, 1e-3]
        }
    
    # Create preprocessing pipeline
    preprocessor, _, _ = create_preprocessing_pipeline(
        feature_types, scale_numeric=True, scaler_type='standard'
    )
    
    # Create pipeline with preprocessor and SGDClassifier
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('to_float32', ToFloat32()),
        ('clf', SGDClassifier(
            loss='hinge',
            class_weight='balanced',
            random_state=RANDOM_SEED,
            max_iter=20000,
            tol=1e-4,
            n_jobs=1  # SGD doesn't support n_jobs, but we'll use GridSearchCV parallelism
        ))
    ])
    
    # Set up cross-validation strategy
    if cv is None:
        cv = StratifiedShuffleSplit(
            n_splits=CV_SPLITS, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED
        )
    
    print(f"Training Linear SVM (SGD) classifier with GridSearchCV...")
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


def train_rbf_svm(X_train_df: pd.DataFrame, y_train: np.ndarray,
                  feature_types: Dict[str, List[str]],
                  param_grid: Optional[Dict[str, List[Any]]] = None,
                  cv: Optional[Any] = None,
                  n_jobs: int = -1,
                  verbose: int = 1) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train an RBF kernel SVM classifier with hyperparameter tuning.
    
    Args:
        X_train_df: Training feature DataFrame
        y_train: Training target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        param_grid: Hyperparameter grid for GridSearchCV (default: config.SVM_RBF_PARAM_GRID)
        cv: Cross-validation strategy (default: StratifiedShuffleSplit)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_pipeline, best_params, cv_results)
    """
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Create preprocessing pipeline
    preprocessor, _, _ = create_preprocessing_pipeline(
        feature_types, scale_numeric=True, scaler_type='standard'
    )
    
    # Create pipeline with preprocessor and classifier
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('to_float32', ToFloat32()),
        ('clf', SVC(**SVM_RBF_PARAMS))
    ])
    
    # Calculate feature dimension for gamma values
    d = len(feature_types['numeric']) + len(feature_types['categorical'])
    
    # Use default parameter grid if not provided, with dynamic gamma values
    if param_grid is None:
        gamma_list = ['scale', 1.0/max(d, 1), 2.0/max(d, 1)]
        param_grid = {
            'clf__C': SVM_RBF_PARAM_GRID['C'],
            'clf__gamma': gamma_list
        }
    
    # Set up cross-validation strategy
    if cv is None:
        cv = StratifiedShuffleSplit(
            n_splits=CV_SPLITS, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED
        )
    
    print(f"Training RBF SVM classifier with GridSearchCV...")
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


def calibrate_svm(pipeline: Pipeline, 
                  X_cal_df: pd.DataFrame, 
                  y_cal: np.ndarray,
                  method: str = CALIBRATION_METHOD,
                  cv: str = 'prefit') -> CalibratedClassifierCV:
    """
    Calibrate an SVM classifier's probability estimates.
    
    Args:
        pipeline: Trained pipeline with SVM classifier
        X_cal_df: Calibration feature DataFrame
        y_cal: Calibration target vector
        method: Calibration method ('sigmoid' or 'isotonic')
        cv: Cross-validation strategy (default: 'prefit')
        
    Returns:
        Calibrated classifier
    """
    print(f"Calibrating SVM probabilities using {method} method...")
    
    # Create calibrated classifier
    # When using cv='prefit', the base_estimator is already fitted
    # and calibration samples are used to fit the calibration
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=cv,
    )
    
    # Fit calibrated model
    calibrated_model.fit(X_cal_df, y_cal)
    
    return calibrated_model


def plot_support_vector_fraction(pipeline: Pipeline, 
                                 model_name: str = "svm_rbf",
                                 save: bool = True) -> plt.Figure:
    """
    Plot the fraction of support vectors for an SVC model.
    
    Args:
        pipeline: Trained pipeline with SVC model
        model_name: Name of the model (for saving)
        save: Whether to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Extract the SVC model from the pipeline
    model = pipeline.named_steps['clf']
    
    if not hasattr(model, 'support_vectors_'):
        print("Model does not have support vectors attribute. Skipping plot.")
        return None
    
    n_support = len(model.support_vectors_)
    
    # Count support vectors by class
    if hasattr(model, 'n_support_'):
        n_support_by_class = model.n_support_
    else:
        n_support_by_class = ["Unknown"]
    
    # Create a figure with text information
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    
    info_text = (
        f"Support Vector Information\n"
        f"-------------------------\n"
        f"Number of support vectors: {n_support}\n\n"
    )
    
    for i, n in enumerate(n_support_by_class):
        if n != "Unknown":
            info_text += f"Class {i} support vectors: {n}\n"
    
    plt.text(0.1, 0.5, info_text, fontsize=12, va='center')
    
    # Save figure if requested
    if save:
        from src.paths import get_figure_path
        plt.savefig(get_figure_path(f"support_vector_fraction_{model_name}.png"), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def train_and_evaluate_linear_svm(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                                  y_train: np.ndarray, y_test: np.ndarray,
                                  feature_types: Dict[str, List[str]],
                                  param_grid: Optional[Dict[str, List[Any]]] = None,
                                  calibrate: bool = True,
                                  save_results: bool = True,
                                  skip_curves: bool = False,
                                  cv: Optional[Any] = None,
                                  n_jobs: int = -1,
                                  use_sgd: bool = False) -> Dict[str, Any]:
    """
    Train, evaluate, and generate plots for a Linear SVM model.
    
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
        cv: Cross-validation strategy
        n_jobs: Number of parallel jobs
        use_sgd: Whether to use SGDClassifier for large datasets (default: False)
        
    Returns:
        Dictionary with results
    """
    # Start timing
    start_time = time.time()
    
    # Train the model based on the selected implementation
    if use_sgd:
        print("Using SGDClassifier for large dataset linear SVM...")
        pipeline, best_params, cv_results = train_linear_svm_sgd(
            X_train_df, y_train, feature_types, param_grid=param_grid, cv=cv, n_jobs=n_jobs
        )
        model_type = "sgd"
    else:
        pipeline, best_params, cv_results = train_linear_svm(
            X_train_df, y_train, feature_types, param_grid=param_grid, cv=cv, n_jobs=n_jobs
        )
        model_type = "liblinear"
    
    # Extract feature names for plotting
    preprocessor = pipeline.named_steps['preprocess']
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, feature_types['categorical'], feature_types['numeric']
    )
    
    # Generate learning curve and complexity curves if not skipping
    if not skip_curves:
        # Create temporary pipeline for learning curve
        if use_sgd:
            temp_pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('to_float32', ToFloat32()),
                ('clf', SGDClassifier(
                    loss='hinge',
                    class_weight='balanced',
                    random_state=RANDOM_SEED,
                    max_iter=20000
                )),
            ])
            
            # Generate complexity curve for alpha (SGD)
            plot_complexity_curve(
                Pipeline([
                    ('preprocess', preprocessor),
                    ('to_float32', ToFloat32()),
                    ('clf', SGDClassifier(
                        loss='hinge',
                        class_weight='balanced',
                        random_state=RANDOM_SEED,
                        max_iter=20000
                    )),
                ]),
                X_train_df, y_train,
                param_name='clf__alpha',
                param_range=[1e-5, 1e-4, 1e-3],
                title="Linear SVM (SGD) Complexity Curve - Alpha",
                model_name="linear_svm_sgd_alpha",
                cv=3,  # Use a smaller CV for complexity curves
                n_jobs=n_jobs
            )
        else:
            # Temporary pipeline (no calibration) for learning-curve computation
            temp_pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('to_float32', ToFloat32()),
                ('clf', LinearSVC(**SVM_LINEAR_PARAMS)),
            ])
            
            # Generate complexity curve for C
            plot_complexity_curve(
                Pipeline([
                    ('preprocess', preprocessor),
                    ('to_float32', ToFloat32()),
                    ('clf', LinearSVC(class_weight='balanced', random_state=RANDOM_SEED)),
                ]),
                X_train_df, y_train,
                param_name='clf__C',
                param_range=[0.1, 1.0, 10.0],
                title="Linear SVM Complexity Curve - C",
                model_name="linear_svm_C",
                cv=3,  # Use a smaller CV for complexity curves
                n_jobs=n_jobs
            )
        
        # Learning curve is the same for both implementations
        plot_learning_curve(
            temp_pipeline, X_train_df, y_train,
            title=f"Linear SVM ({model_type}) Learning Curve",
            model_name=f"linear_svm_{model_type}",
            train_sizes=LEARNING_CURVE_SIZES,
            cv=3,  # Use a smaller CV for learning curve
            n_jobs=n_jobs
        )
    
    # Split training data for calibration
    X_cal_train_df, X_cal_val_df, y_cal_train, y_cal_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Calibrate the model (LinearSVC and SGDClassifier don't have predict_proba, so we always need to calibrate)
    calibrated_model = calibrate_svm(pipeline, X_cal_val_df, y_cal_val)
    final_model = calibrated_model
    model_name = f"linear_svm_{model_type}_calibrated"
    
    # Get predictions on test set
    y_prob = final_model.predict_proba(X_test_df)[:, 1]
    
    # Find optimal threshold
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
    
    # Calculate runtime and peak memory
    runtime = wall_clock_seconds(start_time)
    peak_memory_gb = get_peak_memory_gb()
    
    # Evaluate the model
    metrics = evaluate_classifier(
        final_model,
        X_test_df,
        y_test,
        threshold=threshold,
        y_prob=y_prob,  # pass pre-computed probabilities to avoid recomputation
    )
    
    # Add runtime and memory metrics
    metrics['runtime_seconds'] = runtime
    metrics['peak_memory_gb'] = peak_memory_gb
    
    print("\nTest set metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
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


def train_and_evaluate_rbf_svm(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                               y_train: np.ndarray, y_test: np.ndarray,
                               feature_types: Dict[str, List[str]],
                               param_grid: Optional[Dict[str, List[Any]]] = None,
                               calibrate: bool = True,
                               save_results: bool = True,
                               skip_curves: bool = False,
                               cv: Optional[Any] = None,
                               n_jobs: int = -1) -> Dict[str, Any]:
    """
    Train, evaluate, and generate plots for an RBF kernel SVM model.
    
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
        cv: Cross-validation strategy
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with results
    """
    # Start timing
    start_time = time.time()
    
    # Cap training size to keep runtime within budget
    X_train_sample_df = X_train_df
    y_train_sample = y_train
    
    if len(X_train_df) > 100000:
        print(f"Training set size ({len(X_train_df)}) exceeds 100,000 rows. Downsampling to 100,000 rows for RBF SVM training.")
        # Draw a stratified sample
        X_train_sample_df, _, y_train_sample, _ = train_test_split(
            X_train_df, y_train, 
            train_size=100000, 
            random_state=RANDOM_SEED, 
            stratify=y_train
        )
        print(f"Sampled training set: {len(X_train_sample_df)} rows, positive rate: {y_train_sample.mean():.4f}")
    
    # Train the model on the sampled data
    pipeline, best_params, cv_results = train_rbf_svm(
        X_train_sample_df, y_train_sample, feature_types, param_grid=param_grid, cv=cv, n_jobs=n_jobs
    )
    
    # Extract feature names for plotting
    preprocessor = pipeline.named_steps['preprocess']
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, feature_types['categorical'], feature_types['numeric']
    )
    
    # Plot support vector fraction
    plot_support_vector_fraction(pipeline, model_name="rbf_svm")
    
    # Generate learning curve and complexity curves if not skipping
    if not skip_curves:
        # Create a temporary pipeline for learning curve
        temp_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('to_float32', ToFloat32()),
            ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED))
        ])
        
        # Use the sampled data for learning curves
        plot_learning_curve(
            temp_pipeline, X_train_sample_df, y_train_sample,
            title="RBF SVM Learning Curve",
            model_name="rbf_svm",
            train_sizes=LEARNING_CURVE_SIZES,
            cv=3,  # Use a smaller CV for learning curve
            n_jobs=n_jobs
        )
        
        # Generate complexity curve for C using sampled data
        plot_complexity_curve(
            Pipeline([
                ('preprocess', preprocessor),
                ('to_float32', ToFloat32()),
                ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED))
            ]),
            X_train_sample_df, y_train_sample,
            param_name='clf__C',
            param_range=[0.5, 2.0, 8.0],
            title="RBF SVM Complexity Curve - C",
            model_name="rbf_svm_C",
            cv=3,  # Use a smaller CV for complexity curves
            n_jobs=n_jobs
        )
    
    # Split sampled training data for calibration
    X_cal_train_df, X_cal_val_df, y_cal_train, y_cal_val = train_test_split(
        X_train_sample_df, y_train_sample, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_sample
    )
    
    # Calibrate the model if requested
    if calibrate:
        calibrated_model = calibrate_svm(pipeline, X_cal_val_df, y_cal_val)
        final_model = calibrated_model
        model_name = "rbf_svm_calibrated"
    else:
        # Since SVM_RBF_PARAMS has probability=False, we need to calibrate to get probabilities
        calibrated_model = calibrate_svm(pipeline, X_cal_val_df, y_cal_val)
        final_model = calibrated_model
        model_name = "rbf_svm"
    
    # Get predictions on test set
    y_prob = final_model.predict_proba(X_test_df)[:, 1]
    
    # Find optimal threshold
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
    
    # Calculate runtime and peak memory
    runtime = wall_clock_seconds(start_time)
    peak_memory_gb = get_peak_memory_gb()
    
    # Evaluate the model
    metrics = evaluate_classifier(
        final_model,
        X_test_df,
        y_test,
        threshold=threshold,
        y_prob=y_prob,  # pass pre-computed probabilities to avoid recomputation
    )
    
    # Add runtime and memory metrics
    metrics['runtime_seconds'] = runtime
    metrics['peak_memory_gb'] = peak_memory_gb
    
    print("\nTest set metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
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


def run_svm_pipeline(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                     y_train: np.ndarray, y_test: np.ndarray,
                     feature_types: Dict[str, List[str]],
                     calibrate: bool = True,
                     save_results: bool = True,
                     skip_curves: bool = False,
                     cv: Optional[Any] = None,
                     n_jobs: int = -1,
                     use_sgd: bool = False) -> Dict[str, Any]:
    """
    Run the complete SVM pipeline from preprocessing to evaluation.
    
    Args:
        X_train_df: Training feature DataFrame
        X_test_df: Test feature DataFrame
        y_train: Training target vector
        y_test: Test target vector
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        calibrate: Whether to calibrate the model's probabilities
        save_results: Whether to save models, metrics, and plots
        skip_curves: Whether to skip generating learning and complexity curves
        cv: Cross-validation strategy
        n_jobs: Number of parallel jobs
        use_sgd: Whether to use SGDClassifier for large datasets (default: False)
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("SUPPORT VECTOR MACHINE PIPELINE")
    print("="*80)
    
    # Train and evaluate Linear SVM
    print("\n" + "-"*80)
    print(f"LINEAR SVM {'(SGD)' if use_sgd else '(liblinear)'}")
    print("-"*80)
    linear_results = train_and_evaluate_linear_svm(
        X_train_df, X_test_df, y_train, y_test, feature_types,
        calibrate=True, save_results=save_results,  # Always calibrate Linear SVM
        skip_curves=skip_curves, cv=cv, n_jobs=n_jobs, use_sgd=use_sgd
    )
    
    # Train and evaluate RBF SVM
    print("\n" + "-"*80)
    print("RBF KERNEL SVM")
    print("-"*80)
    rbf_results = train_and_evaluate_rbf_svm(
        X_train_df, X_test_df, y_train, y_test, feature_types,
        calibrate=calibrate, save_results=save_results,
        skip_curves=skip_curves, cv=cv, n_jobs=n_jobs
    )
    
    # Combine results
    results = {
        'linear_svm': linear_results,
        'rbf_svm': rbf_results
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.data import load_dataset, preprocess_dataset
    
    # Load and preprocess the dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test, feature_types = preprocess_dataset(df, do_eda=True)
    
    # Run the SVM pipeline
    results = run_svm_pipeline(
        X_train, X_test, y_train, y_test, feature_types,
        calibrate=True, save_results=True
    )
    
    print("\nSVM pipeline completed!")
