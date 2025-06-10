import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


class ModelPlotter:
    def __init__(self, config):
        self.config = config
        self.data = None

    def scatter(self, model):
        """Generate scatter plot of predicted vs actual returns"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'scatter'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions from model's stored test data
        y_pred = model.predict()
        y_test = model.y_test
        test_dates = y_pred.index.get_level_values('date')

        print(f"Plotting {len(y_pred)} predictions")

        # Create single scatter plot for all data
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Simple scatter plot
        ax.scatter(y_test, y_pred, c='#3498db', alpha=0.6, s=30,
                   label=f'{len(y_test)} predictions')

        # Determine axis bounds centered at 0
        all_values = list(y_test) + list(y_pred)
        max_abs = max(abs(min(all_values)), abs(max(all_values)))
        axis_limit = max_abs * 1.1

        # Add diagonal line (perfect predictions)
        ax.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit],
               'k--', alpha=0.5, lw=1)

        # Set axis bounds and formatting
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

        # Labels and title
        ax.set_xlabel('Actual Returns')
        ax.set_ylabel('Predicted Returns')
        ax.set_title('Predicted vs Actual Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Overall title with test period info
        plt.suptitle(f'{model.model_class_name.upper()} Model\n'
                    f'Test Period: {test_dates[0].date()} to {test_dates[-1].date()}',
                    fontsize=14)
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'{model.model_class_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return [str(plot_path)]

    def line_chart(self, model):
        """Generate line chart of actual vs predicted prices over time for each ticker"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'line_charts'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions and test data
        y_pred_returns = model.predict()
        y_test_returns = model.y_test

        # Get unique tickers
        tickers = y_pred_returns.index.get_level_values('ticker').unique()
        n_tickers = len(tickers)

        # Create subplots
        fig, axes = plt.subplots(n_tickers, 1, figsize=(16, 4 * n_tickers))
        if n_tickers == 1:
            axes = [axes]

        plot_paths = []

        for i, ticker in enumerate(tickers):
            ax = axes[i]

            # Filter data for this ticker
            ticker_pred = y_pred_returns.xs(ticker, level='ticker')
            ticker_actual = y_test_returns.xs(ticker, level='ticker')
            dates = ticker_pred.index

            # Get actual prices from the raw data to convert returns to prices
            ticker_data = self.data[ticker].loc[dates[0]:dates[-1]]
            actual_prices = ticker_data['Adj Close'].reindex(dates, method='ffill')

            # Convert returns to price predictions
            predicted_prices = []
            actual_price_series = []

            for j, date in enumerate(dates):
                if j == 0:
                    # First prediction: use previous day's price as base
                    base_price = actual_prices.iloc[j]
                    predicted_prices.append(base_price * (1 + ticker_pred.iloc[j]))
                    actual_price_series.append(base_price * (1 + ticker_actual.iloc[j]))
                else:
                    # Use previous actual price as base for next prediction
                    base_price = actual_price_series[j-1]
                    predicted_prices.append(base_price * (1 + ticker_pred.iloc[j]))
                    actual_price_series.append(base_price * (1 + ticker_actual.iloc[j]))

            # Convert to pandas series for easier manipulation
            predicted_prices = pd.Series(predicted_prices, index=dates)
            actual_price_series = pd.Series(actual_price_series, index=dates)

            # Apply smoothing (7-day rolling average)
            window = min(7, len(dates) // 4)  # Adaptive window size
            if window > 1:
                predicted_smooth = predicted_prices.rolling(window=window, center=True).mean()
                actual_smooth = actual_price_series.rolling(window=window, center=True).mean()
            else:
                predicted_smooth = predicted_prices
                actual_smooth = actual_price_series

            # Plot prices
            ax.plot(dates, actual_smooth, color='gray', linewidth=2, label='Actual Prices', alpha=0.8)
            ax.plot(dates, predicted_smooth, color='red', linewidth=2, label='Predicted Prices', alpha=0.8)

            # Calculate error metrics
            rmse = np.sqrt(np.mean((predicted_prices - actual_price_series) ** 2))
            mae = np.mean(np.abs(predicted_prices - actual_price_series))
            mape = np.mean(np.abs((predicted_prices - actual_price_series) / actual_price_series)) * 100

            # Add metrics to plot
            ax.text(0.02, 0.98, f'RMSE: ${rmse:.2f}\nMAE: ${mae:.2f}\nMAPE: {mape:.1f}%',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'{ticker} - Actual vs Predicted Prices')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

        # Format dates and overall title
        plt.xlabel('Date')
        plt.suptitle(f'{model.model_class_name.upper()} Model - Price Predictions\n'
                    f'Test Period: {dates[0].date()} to {dates[-1].date()}',
                    fontsize=14)
        fig.autofmt_xdate()
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'{model.model_class_name}_price_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated price prediction chart with {n_tickers} tickers")
        return [str(plot_path)]

    def backtest_history(self, model, backtest_results, trade_data):
        """Generate backtest performance visualization"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'backtest'
        plots_dir.mkdir(parents=True, exist_ok=True)

        dates = trade_data['dates']
        returns = trade_data['returns']
        predictions = trade_data['predictions']
        actual = trade_data['actual']

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns) - 1

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Cumulative Returns
        ax1.plot(dates, cumulative_returns * 100, 'b-', linewidth=2, label='Strategy')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.set_title(f'{model.model_class_name.upper()} Confidence Backtest - {trade_data["ticker"]}\n'
                      f'Threshold: {backtest_results["confidence_threshold"]:.2f} | '
                      f'Trades: {backtest_results["total_trades"]} | '
                      f'Win Rate: {backtest_results["win_rate"]:.1%}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Individual Trade Returns
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.scatter(dates, returns * 100, c=colors, alpha=0.6, s=30)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Trade Return (%)')
        ax2.set_title('Individual Trade Performance')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Accuracy Over Time (Rolling)
        window_size = min(20, len(predictions) // 4)
        if window_size > 1:
            rolling_accuracy = pd.Series(predictions == actual).rolling(window=window_size, center=True).mean()
            ax3.plot(dates, rolling_accuracy, 'purple', linewidth=2, label=f'{window_size}-trade rolling accuracy')
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
            ax3.axhline(y=backtest_results['accuracy'], color='orange', linestyle='-', alpha=0.7, label='Overall accuracy')
            ax3.set_ylabel('Accuracy')
            ax3.set_xlabel('Date')
            ax3.set_title('Directional Accuracy Over Time')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Insufficient trades for rolling accuracy',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Directional Accuracy Over Time')

        # Format dates
        fig.autofmt_xdate()
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'{model.model_class_name}_confidence_backtest.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Backtest plot saved to: {plot_path}")
        return str(plot_path)