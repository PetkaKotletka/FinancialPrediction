import pandas as pd
import numpy as np
from typing import Dict


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config

    def clean_data(self, stock_data: Dict[str, pd.DataFrame], fred_data: pd.DataFrame) -> pd.DataFrame:
        """Merge and clean all data sources"""
        all_data = []

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

            all_data.append(df)

        return pd.concat(all_data, axis=0).sort_index()

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction targets"""
        # 1-day return
        df['return_1d'] = df.groupby('ticker')['Close'].pct_change(1).shift(-1)

        # 1-day direction
        df['direction_1d'] = (df['return_1d'] > 0).astype(int)

        # 5-day return
        df['return_5d'] = df.groupby('ticker')['Close'].pct_change(5).shift(-5)

        return df

    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        # Volatility regime
        df['volatility_regime'] = pd.cut(
            df['VIXCLS'],
            bins=[0, 15, 25, 100],
            labels=['Low', 'Medium', 'High']
        )

        # Economic regime
        df['gdp_growth'] = df['GDP'].pct_change()
        df['economic_regime'] = (df['gdp_growth'] > 0).astype(int)

        return df
