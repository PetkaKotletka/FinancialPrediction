import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


def calculate_metrics(y_true, y_pred, task_type='regression'):
    """Calculate relevant metrics based on task type"""
    metrics = {}

    if task_type == 'regression':
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        # TODO: Add more regression metrics
    else:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        # TODO: Add precision, recall, F1

    # TODO: Add trading-specific metrics (Sharpe, max drawdown)

    return metrics
