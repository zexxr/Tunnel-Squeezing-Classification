"""
Model evaluation and metrics functions.

This module provides functions for calculating various evaluation metrics
and generating visualizations for model performance.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy, balanced_accuracy, precision, recall, and f1_score.
        
    Examples
    --------
    >>> metrics = calculate_metrics(y_test, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate detailed classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    target_names : List[str], optional
        List of target class names.
        
    Returns
    -------
    str
        Formatted classification report string.
        
    Examples
    --------
    >>> report = generate_classification_report(y_test, y_pred, 
    ...                                         ['Non-squeezing', 'Minor', 'Severe'])
    >>> print(report)
    """
    if target_names is None:
        target_names = ['Non-squeezing', 'Minor Squeezing', 'Severe Squeezing']
    
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix for classification results.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : List[str], optional
        List of class labels for display.
    normalize : bool, optional
        Whether to normalize the confusion matrix (default: False).
    figsize : tuple, optional
        Figure size (default: (8, 6)).
    cmap : str, optional
        Colormap for the heatmap (default: 'Blues').
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object with the confusion matrix plot.
        
    Examples
    --------
    >>> fig = plot_confusion_matrix(y_test, y_pred, 
    ...                             labels=['Non-squeezing', 'Minor', 'Severe'])
    >>> plt.show()
    """
    if labels is None:
        labels = ['Non-squeezing', 'Minor', 'Severe']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    plt.tight_layout()
    
    return fig


def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate balanced accuracy score.
    
    Balanced accuracy is the average of recall obtained on each class.
    It is useful for imbalanced datasets.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
        
    Returns
    -------
    float
        Balanced accuracy score.
        
    Examples
    --------
    >>> balanced_acc = calculate_balanced_accuracy(y_test, y_pred)
    >>> print(f"Balanced Accuracy: {balanced_acc:.3f}")
    """
    return balanced_accuracy_score(y_true, y_pred)
