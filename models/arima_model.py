from .base_model import BaseModel
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, config: dict, target_config: dict):
        super().__init__(model_name, config, target_config)
        self.arima_config = self.models_config['arima']
        self.window_size = 252  # 1 year of trading days
        self.retrain_frequency = 21  # Retrain every month (21 trading days)
        self.ticker_data = {}
        self.last_retrain_idx = {}

    def get_model_type(self):
        """ARIMA uses time series data"""
        return 'arima'

    def build_model(self):
        """ARIMA builds models during training, not beforehand"""
        pass

    def train(self) -> dict:
        """Train ARIMA models using rolling window approach"""
        results = {}

        for ticker in self.ticker_data.keys():
            ticker_prices = self.ticker_data[ticker]['train_prices']

            if len(ticker_prices) < self.window_size:
                print(f"Warning: Not enough data for {ticker}, using available {len(ticker_prices)} points")
                continue

            try:
                # Train initial model on first window
                initial_window = ticker_prices[:self.window_size]
                arima_model = self._fit_arima_robust(initial_window)

                # Store initial model and training data
                self.ticker_data[ticker]['model'] = arima_model
                self.ticker_data[ticker]['current_data'] = initial_window.copy()
                self.last_retrain_idx[ticker] = self.window_size

                results[f'{ticker}_trained'] = True

            except Exception as e:
                print(f"Failed to train ARIMA for {ticker}: {e}")
                self.ticker_data[ticker]['model'] = None
                results[f'{ticker}_trained'] = False

        return results

    def predict(self) -> pd.Series:
        all_predictions = []
        multiindex_data = []

        for ticker in self.ticker_data.keys():
            ticker_predictions = self._predict_ticker(ticker) if self.ticker_data[ticker]['model'] else [0.0] * len(self.ticker_data[ticker]['test_targets'])
            test_dates = self.ticker_data[ticker]['test_dates']

            all_predictions.extend(ticker_predictions)
            multiindex_data.extend([(date, ticker) for date in test_dates])

        multiindex = pd.MultiIndex.from_tuples(multiindex_data, names=['date', 'ticker'])
        return pd.Series(all_predictions, index=multiindex, name='predictions').sort_index()

    def _predict_ticker(self, ticker):
        """Make predictions for a specific ticker"""
        model = self.ticker_data[ticker]['model']
        current_data = self.ticker_data[ticker]['current_data']
        test_prices = self.ticker_data[ticker]['test_prices']
        train_prices = self.ticker_data[ticker]['train_prices']

        predictions = []
        data_for_prediction = current_data.copy()

        for i, actual_price in enumerate(test_prices):
            # Check if we need to retrain
            if i > 0 and i % self.retrain_frequency == 0:
                try:
                    # Use most recent 252 days including some test data
                    total_available = len(train_prices) + i  # How much data we have so far
                    if total_available >= self.window_size:
                        if i < self.window_size:
                            # Still have enough training data
                            new_window = train_prices[-self.window_size:]
                        else:
                            # Need to include some test data
                            train_portion = train_prices[-(self.window_size - i):]
                            test_portion = test_prices[:i]
                            new_window = np.concatenate([train_portion, test_portion])

                        model = self._fit_arima_robust(new_window)
                        data_for_prediction = new_window.copy()
                except Exception:
                    pass

            # Make prediction
            try:
                # Determine number of steps based on target
                if self.target_column == 'return_5d':
                    steps = 5
                else:
                    steps = 1

                forecast = model.forecast(steps=steps)

                if steps == 1:
                    next_price = forecast[0]
                else:
                    next_price = forecast[-1]  # Use the 5-step ahead forecast

                # Convert price prediction to return prediction
                current_price = data_for_prediction[-1]
                pred_return = (next_price / current_price) - 1

                # Handle classification targets
                if self.target_config['type'] == 'classification':
                    pred_return = 1 if pred_return > 0 else 0

                predictions.append(pred_return)

            except Exception:
                predictions.append(0.0)

            # Update data for next prediction
            data_for_prediction = np.append(data_for_prediction, actual_price)

        return predictions

    def _fit_arima_robust(self, data):
        """Fit ARIMA with robust parameter handling"""
        try:
            # Try original parameters first
            model = ARIMA(
                data,
                order=(self.arima_config['p'], self.arima_config['d'], self.arima_config['q'])
            )
            fitted = model.fit(method='lbfgs', maxiter=50)
            return fitted
        except:
            try:
                # Fallback to simpler parameters
                model = ARIMA(data, order=(1, 1, 0))  # Simple AR(1) with differencing
                fitted = model.fit(method='lbfgs', maxiter=50)
                return fitted
            except:
                # Last resort: random walk
                model = ARIMA(data, order=(0, 1, 0))
                fitted = model.fit()
                return fitted