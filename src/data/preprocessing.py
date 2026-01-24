"""
Data loading and preprocessing functions for tunnel squeezing classification.

This module provides functions for loading, validating, preprocessing, and splitting
the tunnel squeezing dataset.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load tunnel squeezing data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing tunnel data.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe with tunnel squeezing data.
        
    Examples
    --------
    >>> df = load_data('tunnel.csv')
    >>> print(df.shape)
    """
    df = pd.read_csv(filepath)
    # Filter out rows with K(MPa) <= 0 as per notebooks
    df = df[df['K(MPa)'] > 0].copy()
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate input dataframe for required columns and data integrity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
        
    Returns
    -------
    bool
        True if data is valid, raises ValueError otherwise.
        
    Raises
    ------
    ValueError
        If required columns are missing or data contains issues.
        
    Examples
    --------
    >>> df = load_data('tunnel.csv')
    >>> validate_data(df)
    True
    """
    required_columns = ['D (m)', 'H(m)', 'Q', 'K(MPa)', 'Class']
    
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values in required columns
    if df[required_columns].isnull().any().any():
        raise ValueError("Data contains null values in required columns")
    
    # Check if Class column has valid values (1, 2, 3)
    valid_classes = {1, 2, 3}
    if not set(df['Class'].unique()).issubset(valid_classes):
        raise ValueError(f"Class column should only contain values: {valid_classes}")
    
    return True


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess features and extract target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature dataframe (X) and target series (y).
        
    Examples
    --------
    >>> df = load_data('tunnel.csv')
    >>> X, y = preprocess_features(df)
    >>> print(X.shape, y.shape)
    """
    feature_columns = ['D (m)', 'H(m)', 'Q', 'K(MPa)']
    
    X = df[feature_columns].copy()
    y = df['Class'].copy()
    
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets with stratification.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series
        Target series.
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test split datasets.
        
    Examples
    --------
    >>> X, y = preprocess_features(df)
    >>> X_train, X_test, y_train, y_test = split_data(X, y)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    random_state : int, optional
        Random seed for reproducibility (default: 42).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Balanced X_train and y_train as numpy arrays.
        
    Examples
    --------
    >>> X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    >>> print(pd.Series(y_train_balanced).value_counts())
    """
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced
