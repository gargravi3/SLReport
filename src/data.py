"""
Data loading and preprocessing utilities for the hotel booking cancellation prediction project.

This module provides functions for:
- Loading the hotel bookings dataset
- Preprocessing the data (dropping columns, handling missing values)
- Detecting feature types (categorical vs. numeric)
- Splitting the data into train and test sets
- Exploratory data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.config import (
    DATASET_PATH, TARGET_COLUMN, LEAKAGE_COLUMNS,
    RANDOM_SEED, TEST_SIZE, CATEGORICAL_THRESHOLD
)
from src.paths import get_figure_path


def load_dataset(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the hotel bookings dataset.
    
    Args:
        path: Path to the dataset CSV file (default: config.DATASET_PATH)
        
    Returns:
        Loaded DataFrame
    """
    if path is None:
        path = DATASET_PATH
    
    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    return df


def drop_leakage_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Drop columns that would cause data leakage.
    
    Args:
        df: Input DataFrame
        columns: List of columns to drop (default: config.LEAKAGE_COLUMNS)
        
    Returns:
        DataFrame with leakage columns removed
    """
    if columns is None:
        columns = LEAKAGE_COLUMNS
    
    print(f"Dropping leakage columns: {columns}")
    return df.drop(columns=columns, errors='ignore')


def detect_feature_types(df: pd.DataFrame, 
                         categorical_threshold: int = CATEGORICAL_THRESHOLD) -> Dict[str, List[str]]:
    """
    Automatically detect feature types in the dataset.
    
    Args:
        df: Input DataFrame
        categorical_threshold: Maximum number of unique values to consider a column categorical
        
    Returns:
        Dictionary with 'categorical' and 'numeric' keys containing lists of column names
    """
    categorical_features = []
    numeric_features = []
    
    # Exclude the target column
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    
    # Force specific columns to be treated as categorical regardless of dtype/unique counts
    forced_categorical = {'agent', 'company', 'country'}
    for col in forced_categorical:
        if col in feature_columns:
            categorical_features.append(col)
    
    # Create a set of already processed columns for faster lookup
    processed_columns = set(categorical_features)
    
    for column in feature_columns:
        # Skip columns that have already been processed
        if column in processed_columns:
            continue
            
        # Check if column is already categorical or object type
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            categorical_features.append(column)
        # Check if column is numeric but has few unique values (potential categorical)
        elif pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() <= categorical_threshold:
            categorical_features.append(column)
        # Otherwise, consider it numeric
        elif pd.api.types.is_numeric_dtype(df[column]):
            numeric_features.append(column)
        # Default to categorical for other types
        else:
            categorical_features.append(column)
    
    print(f"Detected {len(categorical_features)} categorical features and {len(numeric_features)} numeric features")
    
    return {
        'categorical': categorical_features,
        'numeric': numeric_features
    }


def split_data(df: pd.DataFrame, 
               target_column: str = TARGET_COLUMN,
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and test sets with stratification.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    # Cast target to int32 as required by assignment guidance
    y = df[target_column].astype("int32")
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({100 * (1 - test_size):.0f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({100 * test_size:.0f}%)")
    print(f"Target distribution in train: {y_train.mean():.4f} positive rate")
    print(f"Target distribution in test: {y_test.mean():.4f} positive rate")
    
    return X_train, X_test, y_train, y_test


def analyze_missing_values(df: pd.DataFrame, save_plot: bool = True) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.
    
    Args:
        df: Input DataFrame
        save_plot: Whether to save the missing values plot
        
    Returns:
        DataFrame with missing value statistics
    """
    # Calculate missing value statistics
    missing = pd.DataFrame({
        'count': df.isnull().sum(),
        'percentage': 100 * df.isnull().sum() / len(df)
    }).sort_values('percentage', ascending=False)
    
    # Filter out columns with no missing values
    missing = missing[missing['count'] > 0]
    
    if len(missing) == 0:
        print("No missing values found in the dataset")
        return missing
    
    print(f"Found {len(missing)} columns with missing values")
    
    # Plot missing values
    if save_plot and not missing.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing.index, y='percentage', data=missing)
        plt.title('Missing Values by Column')
        plt.xlabel('Column')
        plt.ylabel('Missing Percentage')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(get_figure_path('missing_values.png'), dpi=300, bbox_inches='tight')
    
    return missing


def analyze_target_distribution(df: pd.DataFrame, 
                                target_column: str = TARGET_COLUMN,
                                save_plot: bool = True) -> None:
    """
    Analyze the distribution of the target variable.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        save_plot: Whether to save the target distribution plot
    """
    # Count target values
    target_counts = df[target_column].value_counts()
    target_percentages = 100 * target_counts / len(df)
    
    print(f"Target distribution:")
    for value, count in target_counts.items():
        percentage = target_percentages[value]
        print(f"  Class {value}: {count} samples ({percentage:.2f}%)")
    
    # Plot target distribution
    if save_plot:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=df)
        plt.title(f'Distribution of {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Count')
        
        # Add count and percentage labels
        for i, (count, percentage) in enumerate(zip(target_counts, target_percentages)):
            plt.text(i, count + 100, f"{count}\n({percentage:.1f}%)", 
                     ha='center', va='bottom')
        
        # Save the plot
        plt.savefig(get_figure_path('target_distribution.png'), dpi=300, bbox_inches='tight')


def analyze_categorical_features(df: pd.DataFrame, 
                                 categorical_features: List[str],
                                 target_column: str = TARGET_COLUMN,
                                 max_features: int = 10,
                                 save_plots: bool = True) -> None:
    """
    Analyze categorical features and their relationship with the target.
    
    Args:
        df: Input DataFrame
        categorical_features: List of categorical feature names
        target_column: Name of the target column
        max_features: Maximum number of features to plot
        save_plots: Whether to save the plots
    """
    # Limit the number of features to plot
    if len(categorical_features) > max_features:
        print(f"Too many categorical features ({len(categorical_features)}). "
              f"Plotting only the first {max_features}.")
        features_to_plot = categorical_features[:max_features]
    else:
        features_to_plot = categorical_features
    
    # Plot each categorical feature
    for feature in features_to_plot:
        # Count unique values
        value_counts = df[feature].value_counts()
        n_unique = len(value_counts)
        
        print(f"Feature '{feature}' has {n_unique} unique values")
        
        # Skip features with too many unique values
        if n_unique > 20:
            print(f"  Skipping plot for '{feature}' (too many unique values)")
            continue
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot count by target
        ax = sns.countplot(x=feature, hue=target_column, data=df)
        plt.title(f'Distribution of {feature} by {target_column}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        
        # Save the plot
        if save_plots:
            plt.tight_layout()
            plt.savefig(get_figure_path(f'cat_feature_{feature}.png'), dpi=300, bbox_inches='tight')
        
        plt.close()


def analyze_numeric_features(df: pd.DataFrame, 
                             numeric_features: List[str],
                             target_column: str = TARGET_COLUMN,
                             max_features: int = 10,
                             save_plots: bool = True) -> None:
    """
    Analyze numeric features and their relationship with the target.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
        target_column: Name of the target column
        max_features: Maximum number of features to plot
        save_plots: Whether to save the plots
    """
    # Limit the number of features to plot
    if len(numeric_features) > max_features:
        print(f"Too many numeric features ({len(numeric_features)}). "
              f"Plotting only the first {max_features}.")
        features_to_plot = numeric_features[:max_features]
    else:
        features_to_plot = numeric_features
    
    # Plot each numeric feature
    for feature in features_to_plot:
        # Calculate statistics
        feature_stats = df[feature].describe()
        print(f"Feature '{feature}' statistics:")
        print(feature_stats)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        sns.histplot(data=df, x=feature, hue=target_column, kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {feature} by {target_column}')
        
        # Box plot
        sns.boxplot(data=df, x=target_column, y=feature, ax=axes[1])
        axes[1].set_title(f'Box plot of {feature} by {target_column}')
        
        # Save the plot
        if save_plots:
            plt.tight_layout()
            plt.savefig(get_figure_path(f'num_feature_{feature}.png'), dpi=300, bbox_inches='tight')
        
        plt.close()


def analyze_correlations(df: pd.DataFrame, 
                         numeric_features: List[str],
                         target_column: str = TARGET_COLUMN,
                         save_plot: bool = True) -> pd.Series:
    """
    Analyze correlations between numeric features and the target.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
        target_column: Name of the target column
        save_plot: Whether to save the correlation plot
        
    Returns:
        Series with correlations to the target
    """
    # Select numeric features and target
    features_with_target = numeric_features + [target_column]
    df_numeric = df[features_with_target]
    
    # Calculate correlations with target
    correlations = df_numeric.corr()[target_column].sort_values(ascending=False)
    correlations = correlations.drop(target_column)
    
    print("Top correlations with target:")
    print(correlations.head(10))
    
    # Plot correlations
    if save_plot:
        plt.figure(figsize=(12, 8))
        
        # Correlation matrix heatmap
        corr_matrix = df_numeric.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(get_figure_path('correlation_matrix.png'), dpi=300, bbox_inches='tight')
        
        # Bar plot of correlations with target
        plt.figure(figsize=(12, 8))
        correlations.plot(kind='barh')
        plt.title(f'Correlations with {target_column}')
        plt.xlabel('Correlation')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(get_figure_path('target_correlations.png'), dpi=300, bbox_inches='tight')
    
    return correlations


def perform_eda(df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: Input DataFrame
        save_plots: Whether to save the plots
        
    Returns:
        Dictionary with EDA results
    """
    print("Starting exploratory data analysis...")
    
    # Basic dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Detect feature types
    feature_types = detect_feature_types(df)
    categorical_features = feature_types['categorical']
    numeric_features = feature_types['numeric']
    
    # Analyze target distribution
    analyze_target_distribution(df, save_plot=save_plots)
    
    # Analyze missing values
    missing_values = analyze_missing_values(df, save_plot=save_plots)
    
    # Analyze categorical features
    analyze_categorical_features(df, categorical_features, save_plots=save_plots)
    
    # Analyze numeric features
    analyze_numeric_features(df, numeric_features, save_plots=save_plots)
    
    # Analyze correlations
    correlations = analyze_correlations(df, numeric_features, save_plot=save_plots)
    
    # Return EDA results
    return {
        'feature_types': feature_types,
        'missing_values': missing_values,
        'correlations': correlations
    }


def preprocess_dataset(df: pd.DataFrame = None, do_eda: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Load, preprocess, and split the dataset in one function.
    
    Args:
        df: Input DataFrame (if None, load from default path)
        do_eda: Whether to perform exploratory data analysis
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_types)
    """
    # Load dataset if not provided
    if df is None:
        df = load_dataset()
    
    # Drop leakage columns
    df = drop_leakage_columns(df)
    
    # Perform EDA if requested
    if do_eda:
        eda_results = perform_eda(df)
        feature_types = eda_results['feature_types']
    else:
        feature_types = detect_feature_types(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    return X_train, X_test, y_train, y_test, feature_types


if __name__ == "__main__":
    # Example usage
    df = load_dataset()
    X_train, X_test, y_train, y_test, feature_types = preprocess_dataset(df)
    
    print("\nPreprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Categorical features: {len(feature_types['categorical'])}")
    print(f"Numeric features: {len(feature_types['numeric'])}")
