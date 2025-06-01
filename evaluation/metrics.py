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

        # Trading metrics
        metrics.update(calculate_trading_metrics(y_true, y_pred))

    else:  # classification
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

    return metrics


def calculate_trading_metrics(y_true, y_pred):
    """Calculate trading-specific metrics"""
    # Simple trading strategy: go long when prediction > 0
    position = (y_pred > 0).astype(int)
    returns = position * y_true

    # Sharpe ratio (annualized, assuming daily returns)
    if returns.std() > 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0.0

    # Hit rate: percentage of profitable trades
    trades = position != 0
    if trades.sum() > 0:
        hit_rate = (returns[trades] > 0).mean()
    else:
        hit_rate = 0.0

    return {
        'sharpe_ratio': sharpe,
        'hit_rate': hit_rate,
        'avg_return_when_long': returns[position == 1].mean() if (position == 1).sum() > 0 else 0.0,
        'n_trades': trades.sum()
    }
