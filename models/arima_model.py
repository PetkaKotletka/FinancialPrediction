from .base_model import BaseModel
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, models_config: dict, target_config: dict):
        super().__init__(model_name, models_config, target_config)
        self.arima_config = models_config['arima']
        self.fitted_values = None
        self.training_index = None

    def get_model_type(self):
        """ARIMA uses time series data"""
        return 'arima'

    def prepare_data(self, df: pd.DataFrame):
        """Extract price series for ARIMA"""
        # For ARIMA, we need continuous price series for each ticker
        # We'll return the price series and corresponding target values

        # Get unique tickers
        tickers = df['ticker'].unique()

        # For now, use the first ticker (SPY or AAPL typically)
        # In production, might want to fit separate ARIMA per ticker
        ticker_data = df[df['ticker'] == tickers[0]].copy()

        # Remove any NaN values
        ticker_data = ticker_data[['Close', self.target_column]].dropna()

        # X is the price series, y is the target
        X = ticker_data[['Close']].values
        y = ticker_data[self.target_column].values

        # Store the index for later use
        self.training_index = ticker_data.index

        return X, y

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
        """Fit ARIMA model on price series"""
        # Extract price series
        price_series = X_train.flatten()

        # Fit ARIMA model
        model = ARIMA(
            price_series,
            order=(self.arima_config['p'],
                   self.arima_config['d'],
                   self.arima_config['q'])
        )

        self.model = model.fit()

        start_idx = max(self.arima_config['p'], self.arima_config['d'])

        # In-sample 1-day returns from fitted values
        train_pred_prices = self.model.fittedvalues[start_idx:]
        train_actual_prices = price_series[start_idx-1:-1]
        train_pred = (train_pred_prices / train_actual_prices) - 1
        train_actual = y_train[start_idx:]

        train_rmse = np.sqrt(np.mean((train_actual - train_pred) ** 2))

        # Validation metrics
        if X_val is not None:
            val_steps = len(X_val)
            forecast = self.model.forecast(steps=val_steps)
            val_prices = X_val.flatten()

            # 1-day returns: each forecast compared to previous actual price
            val_pred = []
            last_price = price_series[-1]
            for i, forecast_price in enumerate(forecast):
                val_pred.append((forecast_price / last_price) - 1)
                last_price = val_prices[i] if i < len(val_prices) - 1 else last_price

            val_pred = np.array(val_pred)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        else:
            val_rmse = train_rmse

        return {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast using ARIMA - 1-day returns only"""
        n_periods = len(X)
        forecast = self.model.forecast(steps=n_periods)

        # For 1-day returns, we need price(t+1) / price(t) - 1
        # Build the full price series: last training price + all forecasts
        last_training_price = self.model.data.endog[-1]
        price_series = np.concatenate([[last_training_price], forecast])

        # Calculate 1-day returns
        returns = (price_series[1:] / price_series[:-1]) - 1

        return returns