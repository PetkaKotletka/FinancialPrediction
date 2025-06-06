import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ModelPlotter:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.volatility_colors = {
            'Low': '#2ecc71',
            'Medium': '#f39c12',
            'High': '#e74c3c'
        }

    def scatter(self, model):
        """Generate scatter plots of predicted vs actual returns for each ticker"""
        # Create plots directory with scatter subfolder
        plots_dir = Path(self.config['paths']['plots_dir']) / 'scatter'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get tickers
        tickers = self.config['data']['tickers']
        n_tickers = len(tickers)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # First pass: collect all data to determine common axis bounds
        all_values = []
        plot_data = {}

        for ticker in tickers:
            # Get ticker data
            ticker_data = self.data[self.data['ticker'] == ticker].copy()

            # Prepare data for this ticker
            X, y = model.prepare_data(ticker_data)
            splits = model.split_data(X, y, ticker_data.index)

            # Get test data
            X_test = splits['X_test']
            y_test = splits['y_test']
            test_indices = splits['idx_test']

            # Get predictions
            y_pred = model.predict(X_test)

            # Get volatility regimes for test data
            test_data = ticker_data.iloc[test_indices]
            volatility_regimes = test_data['volatility_regime'].values

            # Store data for plotting
            plot_data[ticker] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'volatility_regimes': volatility_regimes
            }

            # Collect all values for axis bounds
            all_values.extend(y_test)
            all_values.extend(y_pred)

        # Determine common axis bounds centered at 0
        max_abs = max(abs(min(all_values)), abs(max(all_values)))
        axis_limit = max_abs * 1.1  # Add 10% padding

        # Plot each ticker
        for idx, ticker in enumerate(tickers):
            ax = axes[idx]
            data = plot_data[ticker]

            # Plot points colored by volatility regime
            for regime in ['Low', 'Medium', 'High']:
                mask = data['volatility_regimes'] == regime
                if mask.sum() > 0:
                    ax.scatter(
                        data['y_test'][mask],
                        data['y_pred'][mask],
                        c=self.volatility_colors[regime],
                        label=f'{regime} ({mask.sum()})',
                        alpha=0.6,
                        s=30
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

        # Remove empty subplots if any
        for idx in range(n_tickers, len(axes)):
            fig.delaxes(axes[idx])

        # Overall title
        plt.suptitle(f'{model.model_class_name.upper()} Model - Predicted vs Actual Relative Returns (1-day)',
                    fontsize=14)
        plt.tight_layout()

        # Save plot with model class name
        plot_path = plots_dir / f'{model.model_class_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return [str(plot_path)]