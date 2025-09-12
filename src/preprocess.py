"""
Preprocessing utilities for the hotel booking cancellation prediction project.

This module provides functions for:
- Creating preprocessing pipelines for categorical and numeric features
- Handling missing values through imputation
- Encoding categorical features
- Scaling numeric features (optional)
- Creating a complete preprocessing pipeline using ColumnTransformer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Project configuration
# --------------------------------------------------------------------------- #
from src.config import (
    CATEGORICAL_IMPUTE_STRATEGY,
    NUMERIC_IMPUTE_STRATEGY,
    ENCODING_STRATEGY,
    DTYPE_FLOAT,
)

# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #

from sklearn.base import BaseEstimator, TransformerMixin


class TargetFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Simple target or frequency encoder for categorical columns.

    Parameters
    ----------
    strategy : {'frequency', 'target'}
        Encoding strategy.  *frequency* maps each category to its
        relative frequency in the training data.  *target* maps each
        category to the mean of the target (label must be supplied to
        `fit`).
    """

    def __init__(self, strategy: str = "frequency"):
        if strategy not in {"frequency", "target"}:
            raise ValueError("strategy must be 'frequency' or 'target'")
        self.strategy = strategy
        self.category_mapping_: Dict[str, Dict[Any, float]] = {}
        self._fitted = False

    # ------------------------------------------------------------------ #
    # scikit-learn API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(X)

        for col in X.columns:
            if self.strategy == "frequency":
                freqs = X[col].value_counts(normalize=True)
            else:  # target
                if y is None:
                    raise ValueError("y must be provided for target encoding")
                df = pd.DataFrame({"cat": X[col], "y": y})
                freqs = df.groupby("cat")["y"].mean()
            self.category_mapping_[col] = freqs.to_dict()

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("The encoder has not been fitted.")
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(X)

        encoded_cols = []
        for col in X.columns:
            mapping = self.category_mapping_.get(col, {})
            encoded = X[col].map(mapping).fillna(0.0).astype(DTYPE_FLOAT)
            encoded_cols.append(encoded)
        return np.vstack([col.values for col in encoded_cols]).T

    # ------------------------------------------------------------------ #
    def get_feature_names_out(self, input_features=None):
        # Each categorical column stays as a single encoded column
        return np.asarray(input_features if input_features is not None else [])


class ToFloat32(BaseEstimator, TransformerMixin):
    """Cast array to np.float32."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(DTYPE_FLOAT, copy=False)


def create_categorical_pipeline(ordinal: bool = False) -> Pipeline:
    """
    Create a preprocessing pipeline for categorical features.
    
    Args:
        ordinal: Whether to use ordinal encoding instead of one-hot encoding
        
    Returns:
        Scikit-learn pipeline for categorical preprocessing
    """
    steps = []
    
    # Add imputation step
    steps.append(
        ('imputer', SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY, fill_value='missing'))
    )
    
    # Add encoding step
    if ordinal:
        steps.append(
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        )
    else:
        # Replace One-hot with target/frequency encoder per guidance
        steps.append(
            ('encoder', TargetFrequencyEncoder(strategy=ENCODING_STRATEGY))
        )
    
    # Cast to float32 to satisfy guidance
    steps.append(("to_float32", ToFloat32()))
    return Pipeline(steps)


def create_numeric_pipeline(scale: bool = True, scaler_type: str = 'standard') -> Pipeline:
    """
    Create a preprocessing pipeline for numeric features.
    
    Args:
        scale: Whether to include scaling in the pipeline
        scaler_type: Type of scaler to use ('standard' or 'minmax')
        
    Returns:
        Scikit-learn pipeline for numeric preprocessing
    """
    steps = []
    
    # Add imputation step
    steps.append(
        ('imputer', SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY))
    )
    
    # Add scaling step if requested
    if scale:
        if scaler_type == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'standard' or 'minmax'.")
    
    return Pipeline(steps)


def create_preprocessor(categorical_features: List[str], 
                        numeric_features: List[str],
                        scale_numeric: bool = True,
                        scaler_type: str = 'standard',
                        ordinal_encoding: bool = False) -> ColumnTransformer:
    """
    Create a complete preprocessing pipeline using ColumnTransformer.
    
    Args:
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        scale_numeric: Whether to scale numeric features
        scaler_type: Type of scaler to use for numeric features ('standard' or 'minmax')
        ordinal_encoding: Whether to use ordinal encoding for categorical features
        
    Returns:
        ColumnTransformer for preprocessing both categorical and numeric features
    """
    transformers = []
    
    # Add categorical transformer if there are categorical features
    if categorical_features:
        transformers.append(
            ('cat', create_categorical_pipeline(ordinal=ordinal_encoding), categorical_features)
        )
    
    # Add numeric transformer if there are numeric features
    if numeric_features:
        transformers.append(
            ('num', create_numeric_pipeline(scale=scale_numeric, scaler_type=scaler_type), numeric_features)
        )
    
    return ColumnTransformer(transformers, remainder='drop')


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer, 
                                        categorical_features: List[str],
                                        numeric_features: List[str]) -> List[str]:
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        categorical_features: Original categorical feature names
        numeric_features: Original numeric feature names
        
    Returns:
        List of feature names after preprocessing
    """
    feature_names = []
    
    # Get feature names for each transformer
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            # Robustly obtain output names for categorical transformer.
            try:
                # Newer sklearn transformers implement this directly.
                cat_features = transformer.get_feature_names_out(features)
            except Exception:
                # Either attribute missing or failed.  Try encoder inside a
                # sub-pipeline (our case: TargetFrequencyEncoder or similar).
                cat_features = None

                # If it's a Pipeline, look for the 'encoder' step.
                if hasattr(transformer, "named_steps"):
                    enc = transformer.named_steps.get("encoder", None)
                    if enc is not None and hasattr(enc, "get_feature_names_out"):
                        try:
                            cat_features = enc.get_feature_names_out(features)
                        except Exception:
                            cat_features = None

                # Final fallback – just return the original column names.
                if cat_features is None:
                    cat_features = [str(f) for f in features]
            
            feature_names.extend(cat_features)
        
        elif name == 'num':
            # For numeric features, keep the original names
            feature_names.extend(features)
    
    return feature_names


def create_preprocessing_pipeline(feature_types: Dict[str, List[str]], 
                                  scale_numeric: bool = True,
                                  scaler_type: str = 'standard',
                                  ordinal_encoding: bool = False) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a preprocessing pipeline based on detected feature types.
    
    Args:
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        scale_numeric: Whether to scale numeric features
        scaler_type: Type of scaler to use for numeric features ('standard' or 'minmax')
        ordinal_encoding: Whether to use ordinal encoding for categorical features
        
    Returns:
        Tuple of (preprocessor, categorical_features, numeric_features)
    """
    categorical_features = feature_types['categorical']
    numeric_features = feature_types['numeric']
    
    print(f"Creating preprocessing pipeline with:")
    print(f"  - {len(categorical_features)} categorical features")
    print(f"  - {len(numeric_features)} numeric features")
    print(f"  - Numeric scaling: {scale_numeric} (type: {scaler_type if scale_numeric else 'N/A'})")
    # Reflect chosen encoder strategy in the log
    print(
        "  - Categorical encoding: "
        f"{'ordinal' if ordinal_encoding else ENCODING_STRATEGY}"
    )
    
    preprocessor = create_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scale_numeric=scale_numeric,
        scaler_type=scaler_type,
        ordinal_encoding=ordinal_encoding
    )
    
    return preprocessor, categorical_features, numeric_features


def get_preprocessing_info(preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """
    Get information about the preprocessing pipeline.
    
    Args:
        preprocessor: ColumnTransformer preprocessing pipeline
        
    Returns:
        Dictionary with preprocessing information
    """
    info = {
        'transformers': [],
        'n_features_in': getattr(preprocessor, 'n_features_in_', 'unknown'),
        'n_features_out': 0
    }
    
    # Get information for each transformer
    for name, transformer, features in preprocessor.transformers_:
        transformer_info = {
            'name': name,
            'features': features,
            'n_features_in': len(features),
            'steps': []
        }
        
        # Get steps for pipeline transformers
        if hasattr(transformer, 'steps'):
            for step_name, step in transformer.steps:
                step_info = {
                    'name': step_name,
                    'type': type(step).__name__
                }
                
                # Add specific information for different step types
                if isinstance(step, SimpleImputer):
                    step_info['strategy'] = step.strategy
                elif isinstance(step, (StandardScaler, MinMaxScaler)):
                    step_info['scale'] = True
                    step_info['type'] = type(step).__name__
                elif isinstance(step, OneHotEncoder):
                    step_info['sparse'] = step.sparse_output
                    if hasattr(step, 'n_features_out_'):
                        step_info['n_features_out'] = step.n_features_out_
                elif isinstance(step, OrdinalEncoder):
                    step_info['encoding'] = 'ordinal'
                
                transformer_info['steps'].append(step_info)
        
        info['transformers'].append(transformer_info)
        
        # Calculate total output features if available
        if hasattr(transformer, 'n_features_out_'):
            info['n_features_out'] += transformer.n_features_out_
    
    return info


def apply_preprocessing(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        feature_types: Dict[str, List[str]],
                        scale_numeric: bool = True,
                        scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Apply preprocessing to training and test data.
    
    Args:
        X_train: Training feature DataFrame
        X_test: Test feature DataFrame
        feature_types: Dictionary with 'categorical' and 'numeric' keys containing lists of feature names
        scale_numeric: Whether to scale numeric features
        scaler_type: Type of scaler to use for numeric features ('standard' or 'minmax')
        
    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessor, feature_names)
    """
    # Create preprocessor
    preprocessor, categorical_features, numeric_features = create_preprocessing_pipeline(
        feature_types, scale_numeric, scaler_type
    )
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, categorical_features, numeric_features
    )
    
    print(f"Preprocessing applied:")
    print(f"  - Input shape: {X_train.shape}")
    print(f"  - Output shape: {X_train_processed.shape}")
    print(f"  - Features after preprocessing: {len(feature_names)}")
    
    return X_train_processed, X_test_processed, preprocessor, feature_names
