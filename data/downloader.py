import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
from typing import List, Dict
import requests


class DataDownloader:
    def __init__(self, config: dict):
        self.config = config['data']
        self.ta_config = config['features']
        self.data_dir = Path(config['paths']['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """Download and preprocess data from yfinance and FRED"""
        stock_data = self._download_stock_data()
        fred_data = self._download_fred_data()

        processed_data = {}

        for ticker, df in stock_data.items():
            # Merge with FRED data
            df = df.merge(fred_data, how='left', left_index=True, right_index=True)

            # Forward fill missing economic data
            economic_cols = self.config['fred_indicators']
            df[economic_cols] = df[economic_cols].ffill()

            # Add basic features
            df['returns'] = df['Adj Close'].pct_change()
            df['log_returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            # Calendar features -- First and last day of a business week
            df['is_monday'] = (df.index.dayofweek == 0).astype(int)
            df['is_friday'] = (df.index.dayofweek == 4).astype(int)

            # Create prediction targets
            df['return_1d'] = df['Adj Close'].pct_change(1).shift(-1)
            df['direction_1d'] = (df['return_1d'] > 0).astype(int)
            df['return_5d'] = df['Adj Close'].pct_change(5).shift(-5)

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

            # RSI
            df['RSI'] = ta.rsi(df['Adj Close'], length=self.ta_config['rsi_period'])

            # MACD
            macd = ta.macd(
                df['Adj Close'],
                fast=self.ta_config['macd_fast'],
                slow=self.ta_config['macd_slow'],
                signal=self.ta_config['macd_signal']
            )
            df['MACD'] = macd[f'MACD_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']
            df['MACD_signal'] = macd[f'MACDs_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']
            df['MACD_hist'] = macd[f'MACDh_{self.ta_config["macd_fast"]}_{self.ta_config["macd_slow"]}_{self.ta_config["macd_signal"]}']

            # Bollinger Bands
            bb = ta.bbands(
                df['Adj Close'],
                length=self.ta_config['bb_period'],
                std=self.ta_config['bb_std']
            )
            df['BB_upper'] = bb[f'BBU_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']
            df['BB_middle'] = bb[f'BBM_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']
            df['BB_lower'] = bb[f'BBL_{self.ta_config["bb_period"]}_{self.ta_config["bb_std"]}.0']

            # SMAs
            df['SMA_short'] = ta.sma(df['Adj Close'], length=self.ta_config['sma_short'])
            df['SMA_long'] = ta.sma(df['Adj Close'], length=self.ta_config['sma_long'])

            # OBV
            df['OBV'] = ta.obv(df['Adj Close'], df['Volume'])

            processed_data[ticker] = df.dropna()

        return processed_data

    def _download_stock_data(self) -> Dict[str, pd.DataFrame]:
        """Download stock data from yfinance"""
        stock_data = {}
        tickers = list(self.config['tickers'].keys())

        print(f"Downloading {', '.join(tickers)}...")
        data_raw = yf.download(
            tickers,
            start=self.config['start_date'],
            end=self.config['end_date'],
            group_by='ticker',
            auto_adjust=False
        )

        for ticker in tickers:
            df = data_raw[ticker].copy()
            df.columns.name = None  # remove 'ticker' index
            df['ticker'] = ticker
            df['sector'] = self._get_sector(ticker)
            stock_data[ticker] = df
            print(f"Downloaded {ticker} data: {len(df)} rows")

        return stock_data

    def _download_fred_data(self) -> pd.DataFrame:
        """Download economic indicators from FRED"""
        # For now, it's a local csv file
        fred_path = self.data_dir / 'FRED.csv'
        fred_data = pd.read_csv(fred_path)
        fred_data['observation_date'] = pd.to_datetime(fred_data['observation_date'])
        fred_data.set_index('observation_date', inplace=True)
        return fred_data

    def _get_sector(self, ticker: str) -> str:
        """Map ticker to sector"""
        sector_map = self.config['tickers']
        if ticker not in sector_map.keys():
            print(f'Warning: Ticker {ticker} not found in config')
        return sector_map.get(ticker, 'Unknown')
