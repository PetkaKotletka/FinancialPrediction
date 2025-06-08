import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .strategies import DirectionalStrategy, ThresholdStrategy

@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    position_size: float = 0.1  # 10% of capital per trade
    transaction_cost: float = 0.001  # 0.1% per trade
    threshold: float = 0.0  # Minimum predicted return to trade
    max_positions: int = 5  # Maximum concurrent positions

class Backtester:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run_backtest(self, predictions: pd.DataFrame, actual_returns: pd.DataFrame,
                 model_type='regression', strategy=None) -> Dict:
        """
        Run backtest on predictions
        predictions: DataFrame with columns [date, ticker, predicted_return]
        actual_returns: DataFrame with columns [date, ticker, actual_return]
        model_type: 'regression' or 'classification' to determine strategy
        """
        # Auto-select strategy if not provided
        if strategy is None:
            if model_type == 'classification':
                strategy = DirectionalStrategy()
            else:
                strategy = ThresholdStrategy(
                    buy_threshold=self.config.threshold,
                    sell_threshold=-self.config.threshold
                )

        data = predictions.merge(actual_returns, on=['date', 'ticker'])
        data = data.sort_values('date')

        # Initialize portfolio tracking
        capital = self.config.initial_capital
        positions = {}  # ticker -> (entry_price, shares)
        equity_curve = []
        trades = []

        # Group by date for daily processing
        for date, daily_data in data.groupby('date'):
            # Generate signals
            signals = strategy.generate_signals(daily_data, positions)

            # Execute trades
            for _, signal in signals.iterrows():
                if signal['action'] == 'buy' and len(positions) < self.config.max_positions:
                    # Calculate position size
                    position_value = capital * self.config.position_size
                    cost = position_value * self.config.transaction_cost

                    if capital >= position_value + cost:
                        positions[signal['ticker']] = {
                            'entry_date': date,
                            'position_value': position_value,
                            'entry_return': 0
                        }
                        capital -= (position_value + cost)

                        trades.append({
                            'date': date,
                            'ticker': signal['ticker'],
                            'action': 'buy',
                            'value': position_value,
                            'predicted_return': signal['predicted_return']
                        })

                elif signal['action'] == 'sell' and signal['ticker'] in positions:
                    # Close position
                    position = positions[signal['ticker']]
                    # Calculate return
                    cum_return = (1 + signal['actual_return']) - 1
                    exit_value = position['position_value'] * (1 + cum_return)
                    cost = exit_value * self.config.transaction_cost

                    capital += (exit_value - cost)

                    trades.append({
                        'date': date,
                        'ticker': signal['ticker'],
                        'action': 'sell',
                        'value': exit_value,
                        'return': cum_return,
                        'profit': exit_value - position['position_value'] - 2*cost
                    })

                    del positions[signal['ticker']]

            # Update position values
            portfolio_value = capital
            for ticker, position in positions.items():
                ticker_return = daily_data[daily_data['ticker'] == ticker]['actual_return'].values
                if len(ticker_return) > 0:
                    position['entry_return'] = ((1 + position['entry_return']) * (1 + ticker_return[0])) - 1
                    current_value = position['position_value'] * (1 + position['entry_return'])
                    portfolio_value += current_value

            equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': capital,
                'n_positions': len(positions)
            })

        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades)

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': self._calculate_metrics(equity_df, trades_df)
        }

    def _calculate_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(equity_df) < 2:
            return {}

        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()

        # Basic metrics
        total_return = (equity_df['portfolio_value'].iloc[-1] / equity_df['portfolio_value'].iloc[0]) - 1

        # Sharpe ratio (annualized)
        if equity_df['returns'].std() > 0:
            sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std()
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = (1 + equity_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['profit'] > 0] if 'profit' in trades_df else pd.DataFrame()
            win_rate = len(winning_trades) / len(trades_df[trades_df['action'] == 'sell']) if len(trades_df[trades_df['action'] == 'sell']) > 0 else 0
            avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if len(trades_df[trades_df['profit'] < 0]) > 0 else 0
        else:
            win_rate = avg_win = avg_loss = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'n_trades': len(trades_df[trades_df['action'] == 'sell']) if len(trades_df) > 0 else 0
        }