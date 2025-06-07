import pandas as pd
from typing import Dict, Tuple
from datetime import datetime


class DataSplitter:
    def __init__(self, data_config: dict):
        self.config = data_config

    def get_split_dates(self) -> Dict[str, str]:
        """Return fixed date boundaries for all models"""
        return {
            'train_start': self.config['train_start'],
            'train_end': self.config['train_end'],
            'val_start': self.config['val_start'],
            'val_end': self.config['val_end'],
            'test_start': self.config['test_start'],
            'test_end': self.config['test_end']
        }

    def filter_data_by_dates(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter dataframe by date range"""
        return df[(df.index >= start_date) & (df.index <= end_date)]

    def get_date_masks(self, dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Return boolean masks for train/val/test periods"""
        split_dates = self.get_split_dates()

        return {
            'train': (dates >= split_dates['train_start']) & (dates <= split_dates['train_end']),
            'val': (dates >= split_dates['val_start']) & (dates <= split_dates['val_end']),
            'test': (dates >= split_dates['test_start']) & (dates <= split_dates['test_end'])
        }