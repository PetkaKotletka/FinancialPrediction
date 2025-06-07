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

    def prepare_data(self, df: pd.DataFrame):
        """Extract and organize price series by ticker"""

        # Reset for new data preparation
        self.ticker_data = {}

        # Group by ticker and extract price/return series
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy()
            clean_data = ticker_df[['Close', self.target_column]].dropna()

            if len(clean_data) > 0:
                self.ticker_data[ticker] = {
                    'prices': clean_data['Close'].values,
                    'returns': clean_data[self.target_column].values
                }

        # For interface compatibility, return combined data
        # But store current ticker for prediction routing
        if len(self.ticker_data) == 1:
            # Single ticker case (like in scatter plot)
            self.current_ticker = list(self.ticker_data.keys())[0]
            ticker_data = self.ticker_data[self.current_ticker]
            return ticker_data['prices'].reshape(-1, 1), ticker_data['returns']
        else:
            # Multi-ticker case (like in training)
            all_prices = []
            all_returns = []
            for ticker_data in self.ticker_data.values():
                all_prices.extend(ticker_data['prices'])
                all_returns.extend(ticker_data['returns'])
            return np.array(all_prices).reshape(-1, 1), np.array(all_returns)

    def prepare_data_for_dates(self, df: pd.DataFrame, start_date: str, end_date: str) -> tuple:
        """Prepare ARIMA data for specific date range"""
        filtered_df = self.data_splitter.filter_data_by_dates(df, start_date, end_date)

        # Use original prepare_data logic on filtered data
        return self.prepare_data(filtered_df)

    def get_available_dates(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Return dates ARIMA can predict for"""
        filtered_df = self.data_splitter.filter_data_by_dates(df, start_date, end_date)

        if self.current_ticker:
            # Single ticker mode
            ticker_data = filtered_df[filtered_df['ticker'] == self.current_ticker]
            clean_data = ticker_data[['Close', self.target_column]].dropna()
            return clean_data.index
        else:
            # Multi-ticker mode - return union of all available dates
            all_dates = []
            for ticker in filtered_df['ticker'].unique():
                ticker_data = filtered_df[filtered_df['ticker'] == ticker]
                clean_data = ticker_data[['Close', self.target_column]].dropna()
                all_dates.extend(clean_data.index)

            return pd.DatetimeIndex(sorted(set(all_dates)))

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