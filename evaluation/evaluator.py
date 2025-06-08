import numpy as np
import pandas as pd

from .metrics import calculate_metrics
from data import DataSplitter
from backtesting import Backtester, BacktestConfig


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

    def evaluate_all_models(self, data_config, models_dict, data_dict):
        """Evaluate all models on same test data"""
        splitter = DataSplitter(data_config)
        split_dates = splitter.get_split_dates()

        results = {}

        # Get test data with history for sequence models
        test_data_with_history = {}
        for ticker, df in data_dict.items():
            # Include 30 days before test period for windowing
            expanded_start = (pd.to_datetime(split_dates['test_start']) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            expanded_df = splitter.filter_data_by_dates(df, expanded_start, split_dates['test_end'])
            test_data_with_history[ticker] = expanded_df

        print("Model evaluation (all models use same test period):")

        for model_name, model_info in models_dict.items():
            model = model_info['model']

            # Prepare test data
            X_test, y_test = model.prepare_data(test_data_with_history)

            if len(X_test) == 0:
                print(f"  {model_name}: No test data available")
                results[model_name] = None
                continue

            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            results[model_name] = metrics

            key_metric = metrics.get('directional_accuracy', metrics.get('accuracy', 'N/A'))
            print(f"  {model_name}: {len(X_test)} test samples, key_metric: {key_metric:.4f}")

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

    def backtest_model(self, model, test_data: pd.DataFrame, strategy=None):
        """Run backtest for a model"""

        # Detect model type from target config
        model_type = model.target_config['type']

        # Prepare predictions
        X_test, y_test = model.prepare_data(test_data)
        predictions = model.predict(X_test)

        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'date': test_data.index[:len(predictions)],
            'ticker': test_data['ticker'].iloc[:len(predictions)],
            'predicted_return': predictions
        })

        # Get actual returns
        actual_df = pd.DataFrame({
            'date': test_data.index[:len(predictions)],
            'ticker': test_data['ticker'].iloc[:len(predictions)],
            'actual_return': y_test
        })

        print("Predictions:")
        print(pred_df.head())
        print(actual_df.head())

        # Run backtest with appropriate model type
        config = BacktestConfig()
        backtester = Backtester(config)
        results = backtester.run_backtest(pred_df, actual_df, model_type=model_type, strategy=strategy)

        return results
