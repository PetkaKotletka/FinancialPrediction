import pandas as pd
import pandas_ta as ta
from typing import List, Dict


class FeatureEngineer:
    def __init__(self, data_config: dict):
        self.config = data_config
        self.ta_config = config['features']['technical_indicators']

    def create_technical_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create technical indicators for each ticker"""
        for ticker, df in data_dict.items():
            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=self.ta_config['rsi_period'])

            # MACD
            macd = ta.macd(
                df['Close'],
                fast=self.ta_config['macd_fast'],
                slow=self.ta_config['macd_slow'],
                signal=self.ta_config['macd_signal']
            )
            df['MACD'] = macd[f'MACD_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']
            df['MACD_signal'] = macd[f'MACDs_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']
            df['MACD_hist'] = macd[f'MACDh_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']

            # Bollinger Bands
            bb = ta.bbands(
                df['Close'],
                length=self.ta_config['bb_period'],
                std=self.ta_config['bb_std']
            )
            df['BB_upper'] = bb[f'BBU_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']
            df['BB_middle'] = bb[f'BBM_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']
            df['BB_lower'] = bb[f'BBL_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']

            # SMAs
            df['SMA_short'] = ta.sma(df['Close'], length=self.ta_config['sma_short'])
            df['SMA_long'] = ta.sma(df['Close'], length=self.ta_config['sma_long'])

            # OBV
            df['OBV'] = ta.obv(df['Close'], df['Volume'])

            data_dict[ticker] = df

        return data_dict

    def create_sector_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add sector features to each ticker"""
        sector_map = self.config['tickers']
        for ticker, df in data_dict.items():
            df['sector'] = sector_map.get(ticker, 'Unknown')
            data_dict[ticker] = df

        return data_dict

    @classmethod
    def get_feature_columns(cls, model_type: str, sectors: List[str]) -> List[str]:
        """Prepare feature columns based on model type"""
        if model_type == 'arima':
            return ['Close']

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
            sector_features = [f'sector_{sector}' for sector in sectors]
            return base_features + sector_features
        elif model_type in ['sequence', 'cnn']:
            # RNN and CNN use raw features without sector encoding
            return base_features
        else:
            raise ValueError(f"Unknown model type: {model_type}")