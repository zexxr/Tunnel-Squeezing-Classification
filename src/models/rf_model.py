"""
Random Forest model training and prediction functions.

This module provides functions for training, optimizing, and making predictions
with Random Forest models for tunnel squeezing classification.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Train a Random Forest model with specified parameters.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : Dict[str, Any], optional
        Random Forest parameters. If None, uses default parameters.
        
    Returns
    -------
    Pipeline
        Trained Random Forest pipeline (StandardScaler -> RandomForestClassifier).
        
    Examples
    --------
    >>> model = train_random_forest(X_train, y_train, {'n_estimators': 100, 'max_depth': 10})
    """
    if params is None:
        params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
    
    # Create pipeline with scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(**params, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def optimize_rf_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Optimize Random Forest hyperparameters using GridSearchCV.
    
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
    >>> best_model, best_params = optimize_rf_hyperparameters(X_train, y_train)
    >>> print(best_params)
    """
    # Define parameter grid
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }
    
    # Create pipeline
    if use_smote:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42))
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
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


def predict_rf(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using trained Random Forest model.
    
    Parameters
    ----------
    model : Pipeline
        Trained Random Forest pipeline.
    X : pd.DataFrame
        Features to predict on.
        
    Returns
    -------
    np.ndarray
        Predicted class labels.
        
    Examples
    --------
    >>> predictions = predict_rf(model, X_test)
    """
    return model.predict(X)


def get_feature_importance(
    model: Pipeline,
    feature_names: list
) -> pd.DataFrame:
    """
    Get feature importance from trained Random Forest model.
    
    Parameters
    ----------
    model : Pipeline
        Trained Random Forest pipeline.
    feature_names : list
        List of feature names.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores, sorted by importance.
        
    Examples
    --------
    >>> importance_df = get_feature_importance(model, ['D (m)', 'H(m)', 'Q', 'K(MPa)'])
    >>> print(importance_df)
    """
    # Extract the Random Forest classifier from the pipeline
    rf_classifier = model.named_steps['rf']
    
    # Get feature importances
    importances = rf_classifier.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df
