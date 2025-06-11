import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support


def calculate_metrics(y_true, y_pred, task_type='regression'):
    """Calculate comprehensive metrics for model evaluation"""
    metrics = {}

    if task_type == 'regression':
        # Basic regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        mean_abs_return = np.mean(np.abs(y_true))
        if mean_abs_return > 0:
            metrics['mae_pct'] = (metrics['mae'] / mean_abs_return) * 100
        else:
            metrics['mae_pct'] = 0.0
        metrics['r2'] = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        # Directional accuracy (valuable for trading)
        y_true_direction = (y_true > 0).astype(int)
        y_pred_direction = (y_pred > 0).astype(int)
        metrics['directional_accuracy'] = np.mean(y_true_direction == y_pred_direction)

    else:  # classification
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

    return metrics
