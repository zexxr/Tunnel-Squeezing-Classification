# Tunnel Squeezing Classification - Modular Code Structure

This repository has been refactored to follow best practices with organized Python modules.

## Directory Structure

```
Tunnel-Squeezing-Classification/
├── src/                           # Modular Python package
│   ├── __init__.py
│   ├── data/                      # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── preprocessing.py       # Data loading, validation, SMOTE
│   ├── models/                    # Machine learning models
│   │   ├── __init__.py
│   │   ├── svm_model.py          # SVM training and prediction
│   │   └── rf_model.py           # Random Forest training and prediction
│   ├── evaluation/                # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py            # Metrics and visualization
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── helpers.py            # Model persistence, logging
│
├── Tunnel_Squeezing_Classification.ipynb  # Baseline SVM notebook
├── Tunnel_Squeezing_SVM.ipynb            # Advanced SVM with SMOTE
├── Tunnel_Squeezing_RandomForest.ipynb   # Random Forest with SMOTE
├── app.py                                # Streamlit web application
├── tunnel.csv                            # Dataset
└── requirements.txt                      # Python dependencies
```

## Module Overview

### `src/data/preprocessing.py`
Data loading and preprocessing functions:
- `load_data(filepath)` - Load and filter tunnel data
- `validate_data(df)` - Validate data integrity
- `preprocess_features(df)` - Extract features and target
- `split_data(X, y, test_size, random_state)` - Train/test split
- `apply_smote(X_train, y_train)` - Apply SMOTE balancing

### `src/models/svm_model.py`
SVM model functions:
- `train_svm(X_train, y_train, params)` - Train SVM model
- `optimize_svm_hyperparameters(X_train, y_train)` - GridSearchCV optimization
- `predict_svm(model, X)` - Make predictions
- `predict_with_confidence(model, X)` - Predictions with probabilities

### `src/models/rf_model.py`
Random Forest model functions:
- `train_random_forest(X_train, y_train, params)` - Train RF model
- `optimize_rf_hyperparameters(X_train, y_train)` - Hyperparameter tuning
- `predict_rf(model, X)` - Make predictions
- `get_feature_importance(model, feature_names)` - Extract feature importance

### `src/evaluation/metrics.py`
Evaluation and metrics functions:
- `calculate_metrics(y_true, y_pred)` - Comprehensive metrics
- `generate_classification_report(y_true, y_pred, target_names)` - Detailed report
- `plot_confusion_matrix(y_true, y_pred, labels)` - Confusion matrix plot
- `calculate_balanced_accuracy(y_true, y_pred)` - Balanced accuracy

### `src/utils/helpers.py`
Utility functions:
- `save_model(model, filepath)` - Save model to disk
- `load_model(filepath)` - Load model from disk
- `setup_logging()` - Configure logging

## Usage Examples

### In Jupyter Notebooks

```python
import sys
sys.path.insert(0, ".")

from src.data.preprocessing import load_data, preprocess_features, split_data
from src.models.svm_model import train_svm, optimize_svm_hyperparameters
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix
from src.utils.helpers import save_model

# Load and prepare data
df = load_data('tunnel.csv')
X, y = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train model
model, best_params = optimize_svm_hyperparameters(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = calculate_metrics(y_test, y_pred)
print(metrics)

# Save model
save_model(model, 'my_model.pkl')
```

### In Python Scripts

```python
from src.data.preprocessing import load_data, preprocess_features
from src.utils.helpers import load_model
from src.models.rf_model import predict_rf

# Load model and data
model = load_model('rf_tunnel_squeezing.pkl')
df = load_data('tunnel.csv')
X, y = preprocess_features(df)

# Make predictions
predictions = predict_rf(model, X)
```

### In Streamlit App

```python
from src.utils.helpers import load_model
from src.models.svm_model import predict_with_confidence

# Load model
model = load_model('svm_tunnel_squeezing_enhanced.pkl')

# Predict with confidence
predictions, probabilities = predict_with_confidence(model, input_features)
```

## Key Features

✅ **Type Hints**: All functions have comprehensive type hints  
✅ **Docstrings**: NumPy/Google-style docstrings with examples  
✅ **Modular**: Clear separation of concerns  
✅ **Reusable**: Functions can be imported and used anywhere  
✅ **Tested**: All modules validated and working  
✅ **Backward Compatible**: Works with existing saved models  

## Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Use Jupyter notebooks:
```bash
jupyter notebook
```

## Development

All functions include:
- Type hints for better IDE support
- Comprehensive docstrings
- Examples in docstrings
- Error handling and validation

Example function structure:
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description.
    
    Parameters
    ----------
    param1 : type1
        Description of param1.
    param2 : type2
        Description of param2.
        
    Returns
    -------
    return_type
        Description of return value.
        
    Examples
    --------
    >>> result = function_name(value1, value2)
    """
    # Implementation
    pass
```
