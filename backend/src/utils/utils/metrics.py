import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def calculate_metrics(y_true, y_pred, problem_type=None):
    """
    Compute standard ML metrics for regression or classification.
    Auto-detects problem type if not specified.
    Args:
        y_true: Ground truth labels/values (numpy array)
        y_pred: Predicted labels/values (numpy array)
        problem_type: 'classification' or 'regression' (optional)
    Returns:
        dict: Metrics
    """
    metrics = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Auto-detect problem type
    if problem_type is None:
        if len(np.unique(y_true)) <= 10 and y_true.dtype in [np.int32, np.int64, np.uint8]:
            problem_type = 'classification'
        else:
            problem_type = 'regression'

    if problem_type == 'classification':
        # If probabilities, convert to class labels
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred_label = np.argmax(y_pred, axis=-1)
        else:
            y_pred_label = (y_pred > 0.5).astype(int) if y_pred.ndim == 1 or y_pred.shape[-1] == 1 else y_pred
        y_true_label = y_true.flatten()
        metrics['accuracy'] = accuracy_score(y_true_label, y_pred_label)
        metrics['precision'] = precision_score(y_true_label, y_pred_label, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true_label, y_pred_label, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true_label, y_pred_label, average='macro', zero_division=0)
    else:
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics['r2'] = r2_score(y_true, y_pred)
        # Directional accuracy (for time series)
        if y_true.ndim == 1:
            true_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)
            direction_true = true_diff > 0
            direction_pred = pred_diff > 0
            metrics['directional_accuracy'] = np.mean(direction_true == direction_pred)
        else:
            metrics['directional_accuracy'] = np.nan
    return metrics 