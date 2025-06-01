import numpy as np
import pandas as pd
from .metrics import calculate_metrics


class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test, test_data=None):
        """Comprehensive model evaluation"""
        # Get predictions
        y_pred = model.predict(X_test)

        # Overall metrics
        metrics = calculate_metrics(y_test, y_pred, model.target_config['type'])

        # Regime-based evaluation if test_data provided
        if test_data is not None:
            metrics['regime_analysis'] = self._evaluate_by_regime(
                y_test, y_pred, test_data, model.target_config['type']
            )

        # Sample predictions
        metrics['sample_predictions'] = {
            'y_true': y_test[:5].tolist(),
            'y_pred': y_pred[:5].tolist()
        }

        return metrics

    def _evaluate_by_regime(self, y_true, y_pred, test_data, task_type):
        """Evaluate performance across different market regimes"""
        results = {}

        # Volatility regime analysis
        if 'volatility_regime' in test_data.columns:
            vol_regimes = {}
            for regime in test_data['volatility_regime'].unique():
                mask = test_data['volatility_regime'] == regime
                if mask.sum() > 0:
                    regime_metrics = calculate_metrics(
                        y_true[mask],
                        y_pred[mask],
                        task_type
                    )
                    vol_regimes[regime] = {
                        'n_samples': mask.sum(),
                        'key_metric': regime_metrics.get('directional_accuracy',
                                                         regime_metrics.get('accuracy', 0))
                    }
            results['volatility_regimes'] = vol_regimes

        # Sector analysis
        if 'sector' in test_data.columns:
            sectors = {}
            for sector in test_data['sector'].unique():
                mask = test_data['sector'] == sector
                if mask.sum() > 0:
                    sector_metrics = calculate_metrics(
                        y_true[mask],
                        y_pred[mask],
                        task_type
                    )
                    sectors[sector] = {
                        'n_samples': mask.sum(),
                        'key_metric': sector_metrics.get('directional_accuracy',
                                                         sector_metrics.get('accuracy', 0))
                    }
            results['sectors'] = sectors

        return results
