import matplotlib.pyplot as plt
import numpy as np
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
        """Generate line chart of actual vs predicted returns over time"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'line_charts'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions from model's stored test data
        y_pred = model.predict()
        y_test = model.y_test
        test_dates = y_pred.index.get_level_values('date')

        print(f"Plotting {len(y_pred)} predictions over time")

        # Create time series plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Main plot: actual vs predicted returns
        ax1.plot(test_dates, y_test, label='Actual Returns', color='gray', alpha=0.7)
        ax1.plot(test_dates, y_pred, label='Predicted Returns', color='blue', alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Returns')
        ax1.set_title('Actual vs Predicted Returns Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: prediction errors
        errors = y_pred - y_test
        ax2.plot(test_dates, errors, color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Prediction Errors')
        ax2.grid(True, alpha=0.3)

        # Format dates
        fig.autofmt_xdate()

        # Overall title
        plt.suptitle(f'{model.model_class_name.upper()} Model - Time Series\n'
                    f'Test Period: {test_dates[0].date()} to {test_dates[-1].date()}',
                    fontsize=14)
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'{model.model_class_name}_line_chart.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return [str(plot_path)]