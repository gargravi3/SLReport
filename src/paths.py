"""
Path utilities for the hotel booking cancellation prediction project.

This module provides utility functions for managing file paths and
ensuring directories exist before they're used.
"""

import os
from pathlib import Path
from typing import Union, Optional

from src.config import (
    ROOT_DIR, DATA_DIR, OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, METRIC_DIR
)

def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path as string or Path object
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_data_path(filename: Optional[str] = None) -> Path:
    """
    Get path to a data file or the data directory.
    
    Args:
        filename: Optional name of the file within the data directory
        
    Returns:
        Path to the file or directory
    """
    ensure_dir_exists(DATA_DIR)
    if filename:
        return DATA_DIR / filename
    return DATA_DIR

def get_figure_path(filename: str, create_dir: bool = True) -> Path:
    """
    Get path to a figure file.
    
    Args:
        filename: Name of the figure file
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to the figure file
    """
    if create_dir:
        ensure_dir_exists(FIGURE_DIR)
    return FIGURE_DIR / filename

def get_model_path(model_name: str, extension: str = ".pkl", create_dir: bool = True) -> Path:
    """
    Get path to a model file.
    
    Args:
        model_name: Name of the model
        extension: File extension (default: .pkl)
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to the model file
    """
    if create_dir:
        ensure_dir_exists(MODEL_DIR)
    
    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = f".{extension}"
        
    return MODEL_DIR / f"{model_name}{extension}"

def get_metric_path(metric_name: str, extension: str = ".csv", create_dir: bool = True) -> Path:
    """
    Get path to a metric file.
    
    Args:
        metric_name: Name of the metric file
        extension: File extension (default: .csv)
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to the metric file
    """
    if create_dir:
        ensure_dir_exists(METRIC_DIR)
    
    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = f".{extension}"
        
    return METRIC_DIR / f"{metric_name}{extension}"

def get_output_path(filename: str, create_dir: bool = True) -> Path:
    """
    Get path to a general output file.
    
    Args:
        filename: Name of the output file
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to the output file
    """
    if create_dir:
        ensure_dir_exists(OUTPUT_DIR)
    return OUTPUT_DIR / filename

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    return ROOT_DIR
