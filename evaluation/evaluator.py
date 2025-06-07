import numpy as np
import pandas as pd
from .metrics import calculate_metrics
from data import DataSplitter


class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test, test_data=None):
        """Comprehensive model evaluation with date-aware handling"""
        # Get predictions
        y_pred = model.predict(X_test)

        # Handle case where model couldn't predict some dates
        if len(y_pred) != len(y_test):
            print(f"Warning: Model predicted {len(y_pred)} samples but expected {len(y_test)}")
            # Truncate to minimum length for now
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
            if test_data is not None:
                test_data = test_data.iloc[:min_len]

        # Overall metrics
        metrics = calculate_metrics(y_test, y_pred, model.target_config['type'])

        # Add sample count info
        metrics['n_predictions'] = len(y_pred)
        metrics['n_expected'] = len(y_test)

        # Regime-based evaluation if test_data provided
        if test_data is not None and len(test_data) == len(y_pred):
            metrics['regime_analysis'] = self._evaluate_by_regime(
                y_test, y_pred, test_data, model.target_config['type']
            )

        return metrics

    def evaluate_models_two_tier(self, data_config, models_dict, data):
        """Two-tier evaluation: full vs common dates"""

        splitter = DataSplitter(data_config)
        split_dates = splitter.get_split_dates()

        results = {}

        # Tier 1: Each model on its maximum test set
        print("Tier 1: Individual model evaluation (maximum test sets)")
        for model_name, model_info in models_dict.items():
            model = model_info['model']

            # Get model's available dates in test period
            available_dates = model.get_available_dates(data, split_dates['test_start'], split_dates['test_end'])

            if len(available_dates) == 0:
                print(f"  {model_name}: No available test dates")
                results[model_name] = {'tier1': None, 'tier2': None}
                continue

            # Prepare data for available dates only
            test_data_subset = data[data.index.isin(available_dates)]
            X, y = model.prepare_data(test_data_subset)

            # Evaluate
            metrics = self.evaluate_model(model, X, y, test_data_subset)
            results[model_name] = {'tier1': metrics, 'available_dates': len(available_dates)}

            print(f"  {model_name}: {len(available_dates)} test dates, "
                  f"key_metric: {metrics.get('directional_accuracy', metrics.get('accuracy', 'N/A')):.4f}")

        # Tier 2: All models on common dates only
        print("\nTier 2: Fair comparison (common test dates only)")

        # Find common dates
        all_available_dates = []
        for model_name, model_info in models_dict.items():
            model = model_info['model']
            available_dates = model.get_available_dates(data, split_dates['test_start'], split_dates['test_end'])
            all_available_dates.append(set(available_dates))

        if all_available_dates:
            common_dates = set.intersection(*all_available_dates)
            common_dates = pd.DatetimeIndex(sorted(common_dates))

            print(f"Common test dates: {len(common_dates)}")

            if len(common_dates) > 0:
                for model_name, model_info in models_dict.items():
                    model = model_info['model']

                    # Prepare data for common dates only
                    common_test_data = data[data.index.isin(common_dates)]
                    X, y = model.prepare_data(common_test_data)

                    # Evaluate
                    metrics = self.evaluate_model(model, X, y, common_test_data)
                    results[model_name]['tier2'] = metrics

                    print(f"  {model_name}: key_metric: {metrics.get('directional_accuracy', metrics.get('accuracy', 'N/A')):.4f}")
            else:
                print("No common dates found across all models")

        return results

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
