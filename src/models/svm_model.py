"""
SVM (Support Vector Machine) model training and prediction functions.

This module provides functions for training, optimizing, and making predictions
with SVM models for tunnel squeezing classification.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Train an SVM model with specified parameters.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : Dict[str, Any], optional
        SVM parameters. If None, uses default parameters.
        
    Returns
    -------
    Pipeline
        Trained SVM pipeline (StandardScaler -> SVC).
        
    Examples
    --------
    >>> model = train_svm(X_train, y_train, {'C': 1.0, 'kernel': 'rbf'})
    """
    if params is None:
        params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
    
    # Create pipeline with scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**params, probability=True, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def optimize_svm_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Optimize SVM hyperparameters using GridSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    use_smote : bool, optional
        Whether to use SMOTE for class balancing (default: True).
        
    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        Best trained model pipeline and best parameters.
        
    Examples
    --------
    >>> best_model, best_params = optimize_svm_hyperparameters(X_train, y_train)
    >>> print(best_params)
    """
    # Define parameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear', 'poly']
    }
    
    # Create pipeline
    if use_smote:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
    
    # Setup GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


def predict_svm(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using trained SVM model.
    
    Parameters
    ----------
    model : Pipeline
        Trained SVM pipeline.
    X : pd.DataFrame
        Features to predict on.
        
    Returns
    -------
    np.ndarray
        Predicted class labels.
        
    Examples
    --------
    >>> predictions = predict_svm(model, X_test)
    """
    return model.predict(X)


def predict_with_confidence(
    model: Pipeline,
    X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores (probabilities).
    
    Parameters
    ----------
    model : Pipeline
        Trained SVM pipeline with probability=True.
    X : pd.DataFrame
        Features to predict on.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Predicted class labels and probability scores for each class.
        
    Examples
    --------
    >>> predictions, probabilities = predict_with_confidence(model, X_test)
    >>> print(f"Prediction: {predictions[0]}, Confidence: {probabilities[0].max():.2f}")
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities
