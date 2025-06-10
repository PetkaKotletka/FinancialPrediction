import numpy as np
from .metrics import calculate_metrics

class ModelEvaluator:
    def evaluate_model(self, model, plotter):
        """Evaluate a single model using its stored test data"""
        # Get predictions and actual values
        y_pred = model.predict()
        y_test = model.y_test

        # Validate MultiIndex alignment
        if not y_pred.index.equals(y_test.index):
            raise ValueError(f"MultiIndex misalignment in {model.model_name}")

        print(f"  {model.model_name}: {len(y_pred)} predicted samples "
              f"({len(y_pred.index.get_level_values('ticker').unique())} tickers)")

        # Calculate metrics based on task type
        metrics = calculate_metrics(y_test, y_pred, model.target_config['type'])

        return metrics

    def confidence_backtest(self, model, plotter):
        """
        Confidence-based trading backtest for direction models
        Only trades when model confidence >= threshold, using single ticker for temporal consistency
        """

        confidence_threshold=0.2
        ticker = '^GSPC'

        if model.target_config['name'] != 'direction_1d':
            print(f"Backtest only supports direction_1d models, got {model.target_config['name']}")
            return None

        print(f"\nRunning confidence backtest for {ticker} with threshold = {confidence_threshold}")
        print("-" * 60)

        # Get predictions and probabilities
        try:
            probabilities = model.predict(return_probabilities=True)
            predictions = model.predict(return_probabilities=False)
        except Exception as e:
            print(f"Error getting probabilities: {e}")
            return None

        actual_values = model.y_test

        # Filter to single ticker only
        ticker_mask = np.array([idx[1] == ticker for idx in model.X_test_index])

        if not np.any(ticker_mask):
            print(f"No data found for ticker {ticker}")
            return None

        # Apply ticker filter
        ticker_probabilities = probabilities[ticker_mask]
        ticker_predictions = predictions[ticker_mask]
        ticker_actual = actual_values[ticker_mask]
        ticker_dates = [model.X_test_index[i][0] for i in range(len(model.X_test_index)) if ticker_mask[i]]

        # Calculate confidence (distance from 0.5)
        confidences = np.abs(ticker_probabilities - 0.5) * 2

        # Filter high-confidence predictions
        high_conf_mask = confidences >= confidence_threshold
        high_conf_count = np.sum(high_conf_mask)

        if high_conf_count == 0:
            print(f"No {ticker} predictions meet confidence threshold {confidence_threshold}")
            return None

        # Get return data (simplified - using random walk for now)
        # In practice, you'd get actual returns from return_1d predictions
        np.random.seed(42)  # For reproducible results
        ticker_returns = np.random.normal(0.0005, 0.015, len(ticker_dates))  # SPY-like returns

        # Apply confidence filter - now we have proper time series
        conf_dates = [ticker_dates[i] for i in range(len(ticker_dates)) if high_conf_mask[i]]
        conf_predictions = ticker_predictions[high_conf_mask]
        conf_returns = ticker_returns[high_conf_mask]
        conf_actual = ticker_actual[high_conf_mask]

        # Calculate strategy returns (Long when pred=1, Short when pred=0)
        strategy_returns = []
        for pred, ret in zip(conf_predictions, conf_returns):
            if pred == 1:  # Predicted up - go long
                strategy_returns.append(ret)
            else:  # Predicted down - go short
                strategy_returns.append(-ret)

        strategy_returns = np.array(strategy_returns)

        # Calculate metrics
        total_trades = len(strategy_returns)
        total_ticker_predictions = len(ticker_predictions)
        win_rate = np.mean(strategy_returns > 0)
        avg_return = np.mean(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        cumulative_return = np.prod(1 + strategy_returns) - 1
        traded_accuracy = np.mean(conf_predictions == conf_actual)

        # Results
        results = {
            'ticker': ticker,
            'confidence_threshold': confidence_threshold,
            'total_trades': total_trades,
            'percentage_of_predictions': (total_trades / total_ticker_predictions) * 100,
            'win_rate': win_rate,
            'accuracy': traded_accuracy,
            'avg_return_per_trade': avg_return,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'total_ticker_predictions': total_ticker_predictions
        }

        # Print results
        print(f"Backtest Results for {ticker}:")
        print(f"  Total {ticker} predictions: {total_ticker_predictions}")
        print(f"  Total trades: {total_trades} ({results['percentage_of_predictions']:.1f}% of {ticker} predictions)")
        print(f"  Directional accuracy: {traded_accuracy:.3f}")
        print(f"  Win rate: {win_rate:.3f}")
        print(f"  Avg return per trade: {avg_return:.4f}")
        print(f"  Cumulative return: {cumulative_return:.3f}")
        print(f"  Sharpe ratio: {sharpe_ratio:.3f}")

        # Prepare data for plotting (now in proper temporal order)
        trade_data = {
            'dates': conf_dates,
            'returns': strategy_returns,
            'predictions': conf_predictions,
            'actual': conf_actual,
            'ticker': ticker
        }

        # Generate backtest plot
        plotter.backtest_history(model, results, trade_data)

        return results