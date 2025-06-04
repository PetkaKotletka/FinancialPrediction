import pandas as pd
import pandas_ta as ta
from typing import List


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.ta_config = config['features']['technical_indicators']

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for each ticker"""
        features = []

        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()

            # RSI
            ticker_data['RSI'] = ta.rsi(ticker_data['Close'], length=self.ta_config['rsi_period'])

            # MACD
            macd = ta.macd(
                ticker_data['Close'],
                fast=self.ta_config['macd_fast'],
                slow=self.ta_config['macd_slow'],
                signal=self.ta_config['macd_signal']
            )
            ticker_data['MACD'] = macd[f'MACD_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']
            ticker_data['MACD_signal'] = macd[f'MACDs_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']
            ticker_data['MACD_hist'] = macd[f'MACDh_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']

            # Bollinger Bands
            bb = ta.bbands(
                ticker_data['Close'],
                length=self.ta_config['bb_period'],
                std=self.ta_config['bb_std']
            )
            ticker_data['BB_upper'] = bb[f'BBU_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']
            ticker_data['BB_middle'] = bb[f'BBM_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']
            ticker_data['BB_lower'] = bb[f'BBL_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']

            # SMAs
            ticker_data['SMA_short'] = ta.sma(ticker_data['Close'], length=self.ta_config['sma_short'])
            ticker_data['SMA_long'] = ta.sma(ticker_data['Close'], length=self.ta_config['sma_long'])

            # OBV
            ticker_data['OBV'] = ta.obv(ticker_data['Close'], ticker_data['Volume'])

            features.append(ticker_data)

        return pd.concat(features, axis=0).sort_index()

    def create_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode sectors"""
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        return pd.concat([df, sector_dummies], axis=1)

    @classmethod
    def get_feature_columns(cls, model_type: str) -> List[str]:
        """Prepare feature columns based on model type"""

        if model_type == 'arima':
            return ['Close']  # Will be handled specially in prepare_data

        # Define feature groups
        technical_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                              'BB_upper', 'BB_middle', 'BB_lower',
                              'SMA_short', 'SMA_long', 'OBV']
        price_features = ['returns', 'log_returns', 'volume_ratio']
        economic_features = ['VIXCLS', 'GDP', 'T10YIE']

        # Base features used by all non-ARIMA models
        base_features = technical_features + price_features + economic_features

        if model_type == 'tabular':
            # Only tabular models use sector encoding
            sector_features = ['sector_Automotive', 'sector_E-commerce', 'sector_Index', 'sector_Technology']
            return base_features + sector_features
        elif model_type in ['sequence', 'cnn']:
            # RNN and CNN use raw features without sector encoding
            return base_features
        else:
            raise ValueError(f"Unknown model type: {model_type}")
