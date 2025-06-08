import pandas as pd
import numpy as np
from typing import Dict


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config

    def clean_data(self, stock_data: Dict[str, pd.DataFrame], fred_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Merge and clean all data sources - return dict of DataFrames per ticker"""
        processed_data = {}

        for ticker, df in stock_data.items():
            # Merge with FRED data
            df = df.merge(fred_data, how='left', left_index=True, right_index=True)

            # Forward fill missing economic data
            economic_cols = ['VIXCLS', 'GDP', 'T10YIE']
            df[economic_cols] = df[economic_cols].ffill()

            # Add basic features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            processed_data[ticker] = df

        return processed_data

    def create_targets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create prediction targets for each ticker"""
        for ticker, df in data_dict.items():
            # 1-day return
            df['return_1d'] = df['Close'].pct_change(1).shift(-1)

            # 1-day direction
            df['direction_1d'] = (df['return_1d'] > 0).astype(int)

            # 5-day return
            df['return_5d'] = df['Close'].pct_change(5).shift(-5)

            data_dict[ticker] = df

        return data_dict

    def create_regime_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add market regime features to each ticker's data"""
        for ticker, df in data_dict.items():
            # Volatility regime
            df['volatility_regime'] = pd.cut(
                df['VIXCLS'],
                bins=[0, 15, 25, 100],
                labels=['Low', 'Medium', 'High']
            )

            # Economic regime
            df['gdp_growth'] = df['GDP'].diff()
            df['economic_regime'] = np.nan
            df.loc[df['gdp_growth'] > 0, 'economic_regime'] = 1  # Expansion
            df.loc[df['gdp_growth'] < 0, 'economic_regime'] = 0  # Contraction
            df.loc[df.index[0], 'economic_regime'] = 1  # Start with expansion
            df['economic_regime'] = df['economic_regime'].ffill()

            data_dict[ticker] = df

        return data_dict
