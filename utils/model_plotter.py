import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data import DataSplitter


class ModelPlotter:
    def __init__(self, config):
        self.config = config
        self.data = None
        # self.volatility_colors = {
        #     'Low': '#2ecc71',
        #     'Medium': '#f39c12',
        #     'High': '#e74c3c'
        # }

    def _check_model_test_data(self, model):
        """Check how much test data each model can actually use"""
        splitter = DataSplitter(self.config['data'])
        split_dates = splitter.get_split_dates()

        print(f"Test data availability for {model.model_class_name}:")
        print(f"Test period: {split_dates['test_start']} to {split_dates['test_end']}")

        for ticker in self.config['data']['tickers']:
            available_dates = model.get_available_dates(
                self.data[self.data['ticker'] == ticker],
                split_dates['test_start'],
                split_dates['test_end']
            )
            print(f"  {ticker}: {len(available_dates)} available dates")

    def scatter(self, model):
        """Generate scatter plots of predicted vs actual returns for each ticker"""
        # Create plots directory with scatter subfolder
        plots_dir = Path(self.config['paths']['plots_dir']) / 'scatter'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Use fixed test dates
        splitter = DataSplitter(self.config['data'])
        split_dates = splitter.get_split_dates()

        # Check what data is available
        self._check_model_test_data(model)

        # Get tickers
        tickers = self.config['data']['tickers']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # First pass: collect all data to determine common axis bounds
        all_values = []
        plot_data = {}

        for ticker in tickers:
            # Get ticker data for test period only
            ticker_data = self.data[self.data['ticker'] == ticker].copy()
            test_ticker_data = splitter.filter_data_by_dates(
                ticker_data,
                split_dates['test_start'],
                split_dates['test_end']
            )

            if len(test_ticker_data) == 0:
                print(f"Warning: No test data for {ticker}")
                continue

            # Prepare data for this ticker's test period
            try:
                X, y = model.prepare_data(test_ticker_data)

                if len(X) == 0 or len(y) == 0:
                    print(f"Warning: No valid predictions for {ticker}")
                    continue

                # Get predictions
                y_pred = model.predict(X)

                # Handle case where predictions don't match expectations
                min_len = min(len(y), len(y_pred))
                y_test = y[:min_len]
                y_pred = y_pred[:min_len]

                # Store data for plotting
                plot_data[ticker] = {
                    'y_test': y_test,
                    'y_pred': y_pred
                }

                # Collect all values for axis bounds
                all_values.extend(y_test)
                all_values.extend(y_pred)

            except Exception as e:
                print(f"Warning: Could not prepare data for {ticker}: {e}")
                continue

        if not plot_data:
            print("Error: No valid data for any ticker")
            return []

        # Determine common axis bounds centered at 0
        max_abs = max(abs(min(all_values)), abs(max(all_values)))
        axis_limit = max_abs * 1.1  # Add 10% padding

        # Plot each ticker
        plotted_tickers = list(plot_data.keys())
        for idx, ticker in enumerate(plotted_tickers):
            if idx >= len(axes):
                break

            ax = axes[idx]
            data = plot_data[ticker]

            # Simple scatter plot - all points same color
            ax.scatter(
                data['y_test'],
                data['y_pred'],
                c='#3498db',  # Blue color
                alpha=0.6,
                s=30,
                label=f'{len(data["y_test"])} predictions'
            )

            # Add diagonal line (perfect predictions)
            ax.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit],
                   'k--', alpha=0.5, lw=1)

            # Set common axis bounds centered at 0
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)

            # Add grid lines through origin
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

            # Labels and title
            ax.set_xlabel('Actual Returns')
            ax.set_ylabel('Predicted Returns')
            ax.set_title(f'{ticker}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(plotted_tickers), len(axes)):
            fig.delaxes(axes[idx])

        # Overall title with test period info
        plt.suptitle(f'{model.model_class_name.upper()} Model - Predicted vs Actual Returns\n'
                    f'Test Period: {split_dates["test_start"]} to {split_dates["test_end"]}',
                    fontsize=14)
        plt.tight_layout()

        # Save plot with model class name
        plot_path = plots_dir / f'{model.model_class_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plotted {len(plotted_tickers)} tickers: {', '.join(plotted_tickers)}")
        return [str(plot_path)]