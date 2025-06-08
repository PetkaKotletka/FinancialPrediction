from .base_model import BaseModel
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, data_config: dict, models_config: dict, target_config: dict):
        super().__init__(model_name, data_config, models_config, target_config)
        self.arima_config = models_config['arima']
        self.fitted_values = None
        self.training_index = None
        self.models = {}
        self.ticker_data = {}
        self.current_ticker = None

    def get_model_type(self):
        """ARIMA uses time series data"""
        return 'arima'

    def build_model(self):
        """ARIMA doesn't pre-build, it fits directly on data"""
        if self.target_column != 'return_1d':
            raise ValueError(
                f"ARIMA can only be used for 1-day return prediction. "
                f"Target '{self.target_column}' is not supported. "
                f"Please use ARIMA only with 'return_1d' target."
            )
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Train separate ARIMA model for each ticker in training data"""
        results = {}

        for ticker in self.ticker_data.keys():  # For each ticker we found during prepare_data
            ticker_prices = self.ticker_data[ticker]['prices']
            ticker_returns = self.ticker_data[ticker]['returns']

            try:
                # Train ARIMA on this ticker's price series
                arima_model = ARIMA(
                    ticker_prices,
                    order=(self.arima_config['p'],
                           self.arima_config['d'],
                           self.arima_config['q'])
                ).fit()

                self.models[ticker] = {
                    'model': arima_model,
                    'training_prices': ticker_prices
                }

                # Forecast next price for validation
                if len(ticker_prices) > 0:
                    next_price = arima_model.forecast(steps=1)[0]
                    pred_return = (next_price / ticker_prices[-1]) - 1
                    results[f'{ticker}_val_sample'] = pred_return

            except Exception as e:
                print(f"Failed to train ARIMA for {ticker}: {e}")
                # Store a dummy model that predicts zero
                self.models[ticker] = None

        return {'trained_tickers': len(self.models)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Route predictions to appropriate ticker's ARIMA model"""

        if self.current_ticker not in self.models or self.models[self.current_ticker] is None:
            # No model for this ticker, return zeros
            return np.zeros(len(X))

        ticker_model = self.models[self.current_ticker]['model']
        training_prices = self.models[self.current_ticker]['training_prices']

        predictions = []
        current_prices = training_prices.copy()
        test_prices = X.flatten()

        for i in range(len(test_prices)):
            try:
                # Refit ARIMA with updated price series (rolling approach)
                updated_model = ARIMA(
                    current_prices,
                    order=(self.arima_config['p'],
                           self.arima_config['d'],
                           self.arima_config['q'])
                ).fit()

                # Forecast next price
                next_price = updated_model.forecast(steps=1)[0]
                pred_return = (next_price / current_prices[-1]) - 1
                predictions.append(pred_return)

            except:
                predictions.append(0.0)

            # Update with actual observed price
            current_prices = np.append(current_prices, test_prices[i])

        return np.array(predictions)