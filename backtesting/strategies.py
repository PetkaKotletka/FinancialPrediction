import pandas as pd
import numpy as np

class TradingStrategy:
    """Base class for trading strategies"""

    def generate_signals(self, predictions: pd.DataFrame, current_positions: dict) -> pd.DataFrame:
        raise NotImplementedError

class DirectionalStrategy(TradingStrategy):
    """Trade based on binary direction predictions (for classification models)"""

    def __init__(self, hold_period=5):
        self.hold_period = hold_period  # Hold for N days after entry

    def generate_signals(self, daily_data: pd.DataFrame, current_positions: dict) -> pd.DataFrame:
        signals = []

        for _, row in daily_data.iterrows():
            ticker = row['ticker']

            # For classification models, predicted_return will be 0 or 1
            prediction = row['predicted_return']

            if ticker not in current_positions and prediction == 1:  # Predicted up
                signals.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'predicted_direction': 'up'
                })
            elif ticker in current_positions:
                # Check if we've held long enough
                position = current_positions[ticker]
                days_held = (row['date'] - position['entry_date']).days

                if prediction == 0 or days_held >= self.hold_period:  # Predicted down or time limit
                    signals.append({
                        'ticker': ticker,
                        'action': 'sell',
                        'predicted_direction': 'down' if prediction == 0 else 'time_exit'
                    })

        return pd.DataFrame(signals)

class MagnitudeStrategy(TradingStrategy):
    """Trade based on magnitude of predicted returns (for regression models)"""

    def __init__(self, entry_percentile=80, exit_percentile=20):
        self.entry_percentile = entry_percentile
        self.exit_percentile = exit_percentile
        self.thresholds = None

    def set_thresholds(self, historical_predictions: pd.Series):
        """Calculate thresholds based on historical predictions"""
        self.thresholds = {
            'buy': np.percentile(historical_predictions, self.entry_percentile),
            'sell': np.percentile(historical_predictions, self.exit_percentile)
        }

    def generate_signals(self, daily_data: pd.DataFrame, current_positions: dict) -> pd.DataFrame:
        if self.thresholds is None:
            raise ValueError("Must call set_thresholds before generating signals")

        signals = []

        for _, row in daily_data.iterrows():
            ticker = row['ticker']
            pred_return = row['predicted_return']

            if ticker not in current_positions and pred_return > self.thresholds['buy']:
                signals.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'predicted_return': pred_return,
                    'signal_strength': 'strong'
                })
            elif ticker in current_positions and pred_return < self.thresholds['sell']:
                signals.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'predicted_return': pred_return,
                    'signal_strength': 'weak'
                })

        return pd.DataFrame(signals)

class ThresholdStrategy(TradingStrategy):
    """Buy when predicted return > threshold, sell when < -threshold"""

    def __init__(self, buy_threshold=0.001, sell_threshold=-0.001):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals(self, daily_data: pd.DataFrame, current_positions: dict) -> pd.DataFrame:
        signals = []

        for _, row in daily_data.iterrows():
            ticker = row['ticker']
            pred_return = row['predicted_return']

            if ticker not in current_positions and pred_return > self.buy_threshold:
                signals.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'predicted_return': pred_return,
                    'actual_return': row['actual_return']
                })
            elif ticker in current_positions and pred_return < self.sell_threshold:
                signals.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'predicted_return': pred_return,
                    'actual_return': row['actual_return']
                })

        return pd.DataFrame(signals)