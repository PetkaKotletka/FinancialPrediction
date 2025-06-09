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
