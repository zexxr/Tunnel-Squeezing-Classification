"""
Utility helper functions.

This module provides general utility functions for model persistence,
logging, and other common operations.
"""

import logging
from typing import Any
import joblib
from pathlib import Path


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk using joblib.
    
    Parameters
    ----------
    model : Any
        Trained model object (sklearn pipeline or estimator).
    filepath : str
        Path where the model should be saved.
        
    Examples
    --------
    >>> save_model(trained_model, 'models/rf_model.pkl')
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    logging.info(f"Model saved successfully to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk using joblib.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model file.
        
    Returns
    -------
    Any
        Loaded model object.
        
    Examples
    --------
    >>> model = load_model('models/rf_model.pkl')
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    logging.info(f"Model loaded successfully from {filepath}")
    
    return model


def setup_logging(
    level: int = logging.INFO,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Configure logging for the application.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).
    format_string : str, optional
        Format string for log messages.
        
    Examples
    --------
    >>> setup_logging(level=logging.DEBUG)
    >>> logging.info("This is an info message")
    """
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    logging.info("Logging configured successfully")
