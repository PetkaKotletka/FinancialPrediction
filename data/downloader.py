import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict
import requests


class DataDownloader:
    def __init__(self, data_config: dict):
        self.config = data_config
        self.data_dir = Path(config['paths']['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_stock_data() -> Dict[str, pd.DataFrame]:
        """Download stock data from yfinance"""
        stock_data = {}
        tickers = self.config['tickers'].keys()

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

    def download_fred_data() -> pd.DataFrame:
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
