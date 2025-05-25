import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict
import requests


class DataDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['paths']['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_stock_data(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Download stock data from yfinance"""
        stock_data = {}

        print(f"Downloading {', '.join(tickers)}...")
        data_raw = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)

        for ticker in tickers:
            df = data_raw[ticker].copy()
            df.columns.name = None  # remove 'ticker' index
            df['ticker'] = ticker
            df['sector'] = self._get_sector(ticker)
            stock_data[ticker] = df
            print(f"Downloaded {ticker} data: {len(df)} rows")

        return stock_data

    def download_fred_data(self, indicators: List[str], start: str, end: str) -> pd.DataFrame:
        """Download economic indicators from FRED"""
        # TODO: Implement FRED API download
        # For now, assume CSV file exists
        fred_path = self.data_dir / 'FRED.csv'
        if fred_path.exists():
            fred_data = pd.read_csv(fred_path)
            fred_data['observation_date'] = pd.to_datetime(fred_data['observation_date'])
            fred_data.set_index('observation_date', inplace=True)
            return fred_data
        else:
            # Create placeholder
            date_range = pd.date_range(start=start, end=end, freq='D')
            return pd.DataFrame(index=date_range)

    def _get_sector(self, ticker: str) -> str:
        """Map ticker to sector"""
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'AMZN': 'E-commerce',
            'TSLA': 'Automotive',
            '^GSPC': 'Index',
            '^DJI': 'Index',
            '^IXIC': 'Index'
        }
        return sector_map.get(ticker, 'Unknown')
