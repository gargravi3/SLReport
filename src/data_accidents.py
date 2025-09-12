"""
Data loading and preprocessing utilities for the US Accidents dataset.

This module provides functions for:
- Loading the US Accidents dataset
- Creating a binary severity target
- Preprocessing the data (dropping columns, handling missing values)
- Detecting feature types (categorical vs. numeric)
- Splitting the data into train and test sets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

from src.config import (
    DATASET_PATH, CATEGORICAL_THRESHOLD, 
    RANDOM_SEED, TEST_SIZE
)
from src.data import split_data


def load_accidents(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the US Accidents dataset.
    
    Args:
        path: Path to the dataset CSV file (default: config.DATASET_PATH)
        
    Returns:
        Loaded DataFrame with binary severity target
    """
    if path is None:
        path = DATASET_PATH
    
    print(f"Loading US Accidents dataset from {path}")
    
    # Define columns to keep
    keep_cols = [
        'ID', 'Source', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng',
        'End_Lat', 'End_Lng', 'Distance(mi)', 'Description', 'Street', 'City',
        'County', 'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code',
        'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
        'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
        'Precipitation(in)', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
        'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
        'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
        'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
        'Astronomical_Twilight', 'End_Time'
    ]
    
    # Define dtype mapping for memory efficiency
    dtype_map = {
        # Numeric columns as float32
        'Start_Lat': 'float32',
        'Start_Lng': 'float32',
        'End_Lat': 'float32',
        'End_Lng': 'float32',
        'Distance(mi)': 'float32',
        'Temperature(F)': 'float32',
        'Wind_Chill(F)': 'float32',
        'Humidity(%)': 'float32',
        'Pressure(in)': 'float32',
        'Visibility(mi)': 'float32',
        'Wind_Speed(mph)': 'float32',
        'Precipitation(in)': 'float32',
        
        # Boolean columns
        'Amenity': 'bool',
        'Bump': 'bool',
        'Crossing': 'bool',
        'Give_Way': 'bool',
        'Junction': 'bool',
        'No_Exit': 'bool',
        'Railway': 'bool',
        'Roundabout': 'bool',
        'Station': 'bool',
        'Stop': 'bool',
        'Traffic_Calming': 'bool',
        'Traffic_Signal': 'bool',
        'Turning_Loop': 'bool',
        
        # Integer columns
        'Severity': 'int8'
    }
    
    # Read CSV file with memory optimizations
    df = pd.read_csv(
        path,
        usecols=keep_cols,
        parse_dates=['Start_Time'],
        dtype=dtype_map
    )
    
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Create binary severity target
    df['is_severe'] = (df['Severity'] >= 3).astype('int32')
    
    # Drop original Severity column
    df = df.drop(columns=['Severity'])
    
    # Derive time features from Start_Time
    df['start_hour'] = df['Start_Time'].dt.hour.astype('int8')
    df['start_dow'] = df['Start_Time'].dt.dayofweek.astype('int8')
    df['start_month'] = df['Start_Time'].dt.month.astype('int8')
    
    # Drop Start_Time after extracting features
    df = df.drop(columns=['Start_Time'])
    
    # Drop leakage columns and other unnecessary columns
    leakage_columns = ['End_Time', 'End_Lat', 'End_Lng', 'Weather_Timestamp', 
                       'Description', 'ID', 'Source']
    
    # Only drop columns that exist in the DataFrame
    leakage_columns = [col for col in leakage_columns if col in df.columns]
    
    if leakage_columns:
        print(f"Dropping leakage and unnecessary columns: {leakage_columns}")
        df = df.drop(columns=leakage_columns)
    
    # Convert high-cardinality categoricals to category dtype
    category_cols = [
        'City', 'Street', 'Zipcode', 'County', 'State', 'Weather_Condition',
        'Wind_Direction', 'Timezone', 'Airport_Code', 'Country',
        'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 
        'Astronomical_Twilight'
    ]
    
    # Only convert columns that exist in the DataFrame
    category_cols = [col for col in category_cols if col in df.columns]
    
    if category_cols:
        print(f"Converting {len(category_cols)} columns to category dtype")
        for col in category_cols:
            df[col] = df[col].astype('category')
    
    # Print memory usage summary
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Print dtypes summary
    print("\nDataFrame dtypes summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Print prevalence of severe accidents
    prevalence = df['is_severe'].mean()
    print(f"\nPrevalence of severe accidents (Severity >= 3): {prevalence:.4f} ({prevalence*100:.2f}%)")
    
    print(f"Final dataset shape after preprocessing: {df.shape}")
    
    return df


def detect_feature_types_accidents(df: pd.DataFrame, 
                                  target_column: str = 'is_severe',
                                  categorical_threshold: int = CATEGORICAL_THRESHOLD) -> Dict[str, List[str]]:
    """
    Automatically detect feature types in the dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column to exclude
        categorical_threshold: Maximum number of unique values to consider a column categorical
        
    Returns:
        Dictionary with 'categorical' and 'numeric' keys containing lists of column names
    """
    categorical_features = []
    numeric_features = []
    
    # Exclude the target column
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Force specific columns to be treated as categorical regardless of dtype/unique counts
    # For US Accidents, high-cardinality columns that should use frequency encoding
    forced_categorical = {'City', 'Street', 'Zipcode', 'County', 'State', 
                          'Weather_Condition', 'Wind_Direction', 'Timezone', 'Airport_Code'}
    
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


def preprocess_accidents(df: pd.DataFrame = None, do_eda: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Load, preprocess, and split the US Accidents dataset in one function.
    
    Args:
        df: Input DataFrame (if None, load from default path)
        do_eda: Whether to perform exploratory data analysis (not implemented for accidents)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_types)
    """
    # Load dataset if not provided
    if df is None:
        df = load_accidents()
    
    # Detect feature types
    feature_types = detect_feature_types_accidents(df)
    
    # If do_eda is True, we would perform EDA here
    # For now, just print a message
    if do_eda:
        print("Note: EDA for accidents dataset not implemented in this version.")
    
    # Split data using the function from src.data
    # The target column is 'is_severe'
    X_train, X_test, y_train, y_test = split_data(df, target_column='is_severe', 
                                                 test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    return X_train, X_test, y_train, y_test, feature_types


if __name__ == "__main__":
    # Example usage
    df = load_accidents()
    X_train, X_test, y_train, y_test, feature_types = preprocess_accidents(df)
    
    print("\nPreprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Categorical features: {len(feature_types['categorical'])}")
    print(f"Numeric features: {len(feature_types['numeric'])}")
