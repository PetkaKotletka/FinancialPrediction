import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation import ModelEvaluator


class ModelPlotter:
    def __init__(self, config):
        self.config = config
        self.data = None

    def scatter(self, models):
        """Generate grid of scatter plots for predicted vs actual returns"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'scatter'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Detect target type from first model
        target_type = models[0].target_config['name']  # 'return_1d' or 'return_5d'
        horizon = '1-Day' if '1d' in target_type else '5-Day'

        n_models = len(models)
        cols = min(3, n_models)  # Max 3 columns
        rows = (n_models + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes for easy iteration
        axes_flat = axes.flatten() if n_models > 1 else axes

        for i, model in enumerate(models):
            ax = axes_flat[i]

            # Get predictions from model's stored test data
            y_pred = model.predict()
            y_test = model.y_test

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)

            # Create scatter plot
            ax.scatter(y_test, y_pred, c='#3498db', alpha=0.6, s=20)

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
            ax.set_title(f'{model.model_class_name.upper()}\nR² = {r2:.3f}')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_models, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Overall title
        test_dates = models[0].predict().index.get_level_values('date')
        plt.suptitle(f'{horizon} Return Models - Predicted vs Actual\n'
                    f'Test Period: {test_dates[0].date()} to {test_dates[-1].date()}',
                    fontsize=14)
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'scatter_{target_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot grid for {len(models)} models")
        return str(plot_path)

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

    def compare_regression_metrics(self, models):
        """Generate directional accuracy comparison and separate RMSE plots for 1d and 5d returns"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'return_model_comparison'
        plots_dir.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator()

        # Group models by class and target type
        model_data = {}
        for model in models:
            model_class = model.model_class_name.upper()
            target_type = model.target_config['name']  # 'return_1d' or 'return_5d'

            if model_class not in model_data:
                model_data[model_class] = {}

            metrics = evaluator.evaluate_model(model)
            model_data[model_class][target_type] = {
                'rmse': metrics['rmse'],
                'directional_accuracy': metrics['directional_accuracy'],
                'y_test': model.y_test
            }

        model_classes = list(model_data.keys())
        x = np.arange(len(model_classes))
        bar_width = 0.35

        # Plot 1: Directional Accuracy Comparison (grouped bars)
        fig, ax = plt.subplots(figsize=(14, 8))

        acc_1d = []
        acc_5d = []

        for model_class in model_classes:
            acc_1d.append(model_data[model_class].get('return_1d', {}).get('directional_accuracy', 0) * 100)
            acc_5d.append(model_data[model_class].get('return_5d', {}).get('directional_accuracy', 0) * 100)

        colors_1d = ['green' if acc >= 50 else 'red' for acc in acc_1d]
        colors_5d = ['green' if acc >= 50 else 'red' for acc in acc_5d]

        bars1 = ax.bar(x - bar_width/2, acc_1d, bar_width,
                       color=colors_1d, alpha=0.7, hatch='//', label='1-Day Returns')
        bars2 = ax.bar(x + bar_width/2, acc_5d, bar_width,
                       color=colors_5d, alpha=0.7, hatch='O', label='5-Day Returns')

        ax.axhline(y=50, color='black', linestyle='--', linewidth=2,
                   label='Random Chance (50%)')

        ax.set_ylim(0, max(acc_1d + acc_5d) * 1.1)
        ax.set_xlabel('Model Types')
        ax.set_ylabel('Directional Accuracy (%)')
        ax.set_title('Model Directional Accuracy: 1-Day vs 5-Day Returns')
        ax.set_xticks(x)
        ax.set_xticklabels(model_classes)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars, values in [(bars1, acc_1d), (bars2, acc_5d)]:
            for bar, val in zip(bars, values):
                if val > 0:  # Only show if model exists
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.xticks(rotation=45)
        plt.tight_layout()
        accuracy_path = plots_dir / 'directional_accuracy_comparison.png'
        plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: RMSE Comparison for 1-Day Returns
        fig, ax = plt.subplots(figsize=(12, 8))

        rmse_1d = []
        model_classes_1d = []

        for model_class in model_classes:
            if 'return_1d' in model_data[model_class]:
                rmse_1d.append(model_data[model_class]['return_1d']['rmse'])
                model_classes_1d.append(model_class)

        # Calculate 1d random walk baseline
        first_1d_model = next(m for m in models if m.target_config['name'] == 'return_1d')
        random_walk_rmse_1d = np.sqrt(np.mean(first_1d_model.y_test ** 2))

        colors_1d = ['green' if rmse < random_walk_rmse_1d else 'red' for rmse in rmse_1d]
        bars = ax.bar(model_classes_1d, rmse_1d, color=colors_1d, alpha=0.7, hatch='//')

        ax.axhline(y=random_walk_rmse_1d, color='black', linestyle='--', linewidth=2,
                   label=f'Random Walk Baseline (RMSE: {random_walk_rmse_1d:.4f})')

        ax.set_ylim(0, max(rmse_1d + [random_walk_rmse_1d]) * 1.1)
        ax.set_xlabel('Model Types')
        ax.set_ylabel('RMSE Values')
        ax.set_title('Model RMSE Performance - 1-Day Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rmse in zip(bars, rmse_1d):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45)
        plt.tight_layout()
        rmse_1d_path = plots_dir / 'rmse_comparison_1d.png'
        plt.savefig(rmse_1d_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: RMSE Comparison for 5-Day Returns
        fig, ax = plt.subplots(figsize=(12, 8))

        rmse_5d = []
        model_classes_5d = []

        for model_class in model_classes:
            if 'return_5d' in model_data[model_class]:
                rmse_5d.append(model_data[model_class]['return_5d']['rmse'])
                model_classes_5d.append(model_class)

        # Calculate 5d random walk baseline
        first_5d_model = next(m for m in models if m.target_config['name'] == 'return_5d')
        random_walk_rmse_5d = np.sqrt(np.mean(first_5d_model.y_test ** 2))

        colors_5d = ['green' if rmse < random_walk_rmse_5d else 'red' for rmse in rmse_5d]
        bars = ax.bar(model_classes_5d, rmse_5d, color=colors_5d, alpha=0.7, hatch='O')

        ax.axhline(y=random_walk_rmse_5d, color='black', linestyle='--', linewidth=2,
                   label=f'Random Walk Baseline (RMSE: {random_walk_rmse_5d:.4f})')

        ax.set_ylim(0, max(rmse_5d + [random_walk_rmse_5d]) * 1.1)
        ax.set_xlabel('Model Types')
        ax.set_ylabel('RMSE Values')
        ax.set_title('Model RMSE Performance - 5-Day Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rmse in zip(bars, rmse_5d):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45)
        plt.tight_layout()
        rmse_5d_path = plots_dir / 'rmse_comparison_5d.png'
        plt.savefig(rmse_5d_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print results
        print(f"Directional Accuracy Results:")
        for model_class in model_classes:
            print(f"  {model_class}:")
            if 'return_1d' in model_data[model_class]:
                acc_1d = model_data[model_class]['return_1d']['directional_accuracy']
                status = "BETTER" if acc_1d > 0.5 else "WORSE"
                print(f"    1-Day: {acc_1d*100:.1f}% ({status})")
            if 'return_5d' in model_data[model_class]:
                acc_5d = model_data[model_class]['return_5d']['directional_accuracy']
                status = "BETTER" if acc_5d > 0.5 else "WORSE"
                print(f"    5-Day: {acc_5d*100:.1f}% ({status})")

        print(f"\nRMSE Results:")
        print(f"1-Day Random Walk Baseline: {random_walk_rmse_1d:.4f}")
        print(f"5-Day Random Walk Baseline: {random_walk_rmse_5d:.4f}")
        for model_class in model_classes:
            print(f"  {model_class}:")
            if 'return_1d' in model_data[model_class]:
                rmse_1d = model_data[model_class]['return_1d']['rmse']
                status = "BETTER" if rmse_1d < random_walk_rmse_1d else "WORSE"
                print(f"    1-Day: {rmse_1d:.4f} ({status})")
            if 'return_5d' in model_data[model_class]:
                rmse_5d = model_data[model_class]['return_5d']['rmse']
                status = "BETTER" if rmse_5d < random_walk_rmse_5d else "WORSE"
                print(f"    5-Day: {rmse_5d:.4f} ({status})")

        return [str(accuracy_path), str(rmse_1d_path), str(rmse_5d_path)]

    def classification_heatmap(self, models):
        """Generate classification metrics heatmap for direction_1d models"""

        plots_dir = Path(self.config['paths']['plots_dir']) / 'classification_metrics'
        plots_dir.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator()
        metrics_data = []

        # Calculate metrics for each model
        for model in models:
            metrics = evaluator.evaluate_model(model)
            metrics_data.append({
                'Model': model.model_class_name.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1_score']
            })

        # Create DataFrame and pivot for heatmap
        df = pd.DataFrame(metrics_data)
        heatmap_data = df.set_index('Model')

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5,
                    vmin=0, vmax=1, fmt='.3f', ax=ax)

        ax.set_title('Classification Metrics Heatmap (Direction Prediction)')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Models')

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / 'classification_metrics_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Classification metrics comparison for {len(models)} models")
        return str(plot_path)

    def return_heatmap(self, models):
        """Generate return metrics heatmap with binary color coding vs baselines"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'return_model_comparison'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Detect target type from first model
        target_type = models[0].target_config['name']  # 'return_1d' or 'return_5d'
        horizon = '1-Day' if '1d' in target_type else '5-Day'

        evaluator = ModelEvaluator()
        metrics_data = []

        # Calculate baselines from first model (same for all)
        first_model = models[0]
        y_true = first_model.y_test
        random_rmse = np.sqrt(np.mean(y_true ** 2))  # Random walk baseline
        random_mae = np.sqrt(np.mean(y_true ** 2))   # Same as RMSE for return data
        random_r2 = 0.0  # No predictive power
        random_dir_acc = 0.5  # Random chance

        baselines = {
            'RMSE': random_rmse,
            'MAE': random_mae,
            'MAE %': 100.0,  # 100% of mean absolute return
            'R²': random_r2,
            'Dir. Acc': random_dir_acc
        }

        # Calculate metrics for each model
        for model in models:
            metrics = evaluator.evaluate_model(model)
            model_name = f"{model.model_class_name.upper()}_{model.target_config['name'].replace('return_', '')}"

            metrics_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAE %': metrics['mae_pct'],
                'R²': metrics['r2'],
                'Dir. Acc': metrics['directional_accuracy']
            })

        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        heatmap_data = df.set_index('Model')

        # Create binary performance matrix (1 = outperforms baseline, 0 = underperforms)
        performance_matrix = np.zeros_like(heatmap_data.values)

        for i, metric in enumerate(heatmap_data.columns):
            baseline = baselines[metric]
            for j, value in enumerate(heatmap_data[metric]):
                if metric in ['RMSE', 'MAE', 'MAE %']:  # Lower is better
                    performance_matrix[j, i] = 1 if value < baseline else 0
                else:  # Higher is better (R², Dir. Acc)
                    performance_matrix[j, i] = 1 if value > baseline else 0

        # Create custom colormap (pastel red/green)
        colors = ['#ffcccc', '#ccffcc']  # Light red, light green
        n_bins = 2
        cmap = plt.matplotlib.colors.ListedColormap(colors)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(performance_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)

        # Add value annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                baseline = baselines[heatmap_data.columns[j]]

                # Format value display
                if heatmap_data.columns[j] in ['RMSE', 'MAE']:
                    text = f'{value:.4f}\n(vs {baseline:.4f})'
                elif heatmap_data.columns[j] == 'MAE %':
                    text = f'{value:.1f}%\n(vs {baseline:.1f}%)'
                elif heatmap_data.columns[j] == 'R²':
                    text = f'{value:.3f}\n(vs {baseline:.1f})'
                else:  # Dir. Acc
                    text = f'{value:.3f}\n(vs {baseline:.1f})'

                color = 'black' if performance_matrix[i, j] == 1 else 'darkred'
                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontweight='bold', fontsize=9)

        ax.set_title(f'{horizon} Return Model Performance vs Random Baselines\n(Green = Outperforms, Red = Underperforms)')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Models')

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'return_metrics_heatmap_{target_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Return metrics heatmap for {len(models)} models")
        return str(plot_path)

    def regime_heatmap(self, model_1d, model_5d):
        """Generate regime-based directional accuracy heatmaps for 1d and 5d models of same class"""
        if model_1d.model_class_name != model_5d.model_class_name:
            raise ValueError("Both models must be from the same class")

        plots_dir = Path(self.config['paths']['plots_dir']) / 'regime_analysis'
        plots_dir.mkdir(parents=True, exist_ok=True)

        model_class = model_1d.model_class_name.upper()

        # Get predictions and actual values for both models
        pred_1d = model_1d.predict()
        actual_1d = model_1d.y_test
        pred_5d = model_5d.predict()
        actual_5d = model_5d.y_test

        # Calculate directional accuracy for each model
        dir_acc_1d = (pred_1d > 0) == (actual_1d > 0)
        dir_acc_5d = (pred_5d > 0) == (actual_5d > 0)

        # Get regime data for the test periods
        def get_regime_data(predictions):
            regime_data = []
            for date, ticker in predictions.index:
                if ticker in self.data and date in self.data[ticker].index:
                    ticker_data = self.data[ticker].loc[date]
                    regime_data.append({
                        'volatility_regime': ticker_data['volatility_regime'],
                        'economic_regime': 'Expansion' if ticker_data['economic_regime'] == 1 else 'Contraction'
                    })
                else:
                    regime_data.append({'volatility_regime': 'Medium', 'economic_regime': 'Expansion'})
            return pd.DataFrame(regime_data, index=predictions.index)

        regime_1d = get_regime_data(pred_1d)
        regime_5d = get_regime_data(pred_5d)

        # Create accuracy matrices and count matrices for heatmaps
        def create_matrices(directional_acc, regime_df):
            # Combine accuracy with regime data
            combined = pd.concat([directional_acc, regime_df], axis=1)
            combined.columns = ['accuracy', 'volatility_regime', 'economic_regime']

            # Group by regimes and calculate mean accuracy and count
            grouped = combined.groupby(['volatility_regime', 'economic_regime']).agg({
                'accuracy': ['mean', 'count']
            }).round(3)

            # Create matrices for heatmap
            vol_regimes = ['Low', 'Medium', 'High']
            econ_regimes = ['Contraction', 'Expansion']

            accuracy_matrix = np.full((len(vol_regimes), len(econ_regimes)), np.nan)
            count_matrix = np.zeros((len(vol_regimes), len(econ_regimes)))

            for i, vol in enumerate(vol_regimes):
                for j, econ in enumerate(econ_regimes):
                    if (vol, econ) in grouped.index:
                        accuracy_matrix[i, j] = grouped.loc[(vol, econ), ('accuracy', 'mean')]
                        count_matrix[i, j] = grouped.loc[(vol, econ), ('accuracy', 'count')]

            return accuracy_matrix, count_matrix

        matrix_1d, count_1d = create_matrices(dir_acc_1d, regime_1d)
        matrix_5d, count_5d = create_matrices(dir_acc_5d, regime_5d)

        # Create figure with proper spacing for colorbar
        fig = plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(1, 3, 1)  # Left third
        ax2 = plt.subplot(1, 3, 2)  # Middle third
        cax = plt.subplot(1, 3, 3)  # Right third for colorbar

        vol_labels = ['Low', 'Medium', 'High']
        econ_labels = ['Contraction', 'Expansion']

        # Create masked arrays to handle NaN values
        matrix_1d_masked = np.ma.masked_where(np.isnan(matrix_1d), matrix_1d)
        matrix_5d_masked = np.ma.masked_where(np.isnan(matrix_5d), matrix_5d)

        # 1-day return heatmap
        im1 = ax1.imshow(matrix_1d_masked, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(econ_labels)))
        ax1.set_yticks(range(len(vol_labels)))
        ax1.set_xticklabels(econ_labels)
        ax1.set_yticklabels(vol_labels)
        ax1.set_xlabel('Economic Regime')
        ax1.set_ylabel('Volatility Regime')
        ax1.set_title(f'{model_class} - 1-Day Returns\nDirectional Accuracy')

        # Add text annotations with sample counts
        for i in range(len(vol_labels)):
            for j in range(len(econ_labels)):
                if not np.isnan(matrix_1d[i, j]):
                    text_color = 'white' if matrix_1d[i, j] < 0.5 else 'black'
                    ax1.text(j, i, f'{matrix_1d[i, j]:.3f}\n(n={int(count_1d[i, j])})',
                            ha='center', va='center', color=text_color, fontweight='bold')
                else:
                    ax1.text(j, i, 'No Data', ha='center', va='center',
                            color='gray', fontweight='bold')

        # 5-day return heatmap
        im2 = ax2.imshow(matrix_5d_masked, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax2.set_xticks(range(len(econ_labels)))
        ax2.set_yticks(range(len(vol_labels)))
        ax2.set_xticklabels(econ_labels)
        ax2.set_yticklabels(vol_labels)
        ax2.set_xlabel('Economic Regime')
        ax2.set_ylabel('Volatility Regime')
        ax2.set_title(f'{model_class} - 5-Day Returns\nDirectional Accuracy')

        # Add text annotations with sample counts
        for i in range(len(vol_labels)):
            for j in range(len(econ_labels)):
                if not np.isnan(matrix_5d[i, j]):
                    text_color = 'white' if matrix_5d[i, j] < 0.5 else 'black'
                    ax2.text(j, i, f'{matrix_5d[i, j]:.3f}\n(n={int(count_5d[i, j])})',
                            ha='center', va='center', color=text_color, fontweight='bold')
                else:
                    ax2.text(j, i, 'No Data', ha='center', va='center',
                            color='gray', fontweight='bold')

        # Create colorbar in the third subplot area
        cax.axis('off')  # Hide the third subplot
        cbar = fig.colorbar(im1, ax=cax, shrink=0.8, aspect=20)
        cbar.set_label('Directional Accuracy', rotation=270, labelpad=20)

        # Adjust layout manually instead of tight_layout
        plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.3)

        # Save plot
        plot_path = plots_dir / f'{model_class.lower()}_regime_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Regime heatmap comparison for {model_class} model")
        return str(plot_path)

    def backtest_heatmap(self, model):
        """Generate backtest performance heatmap across all tickers vs baselines"""

        if model.target_config['name'] != 'direction_1d':
            print(f"Backtest heatmap only supports direction_1d models, got {model.target_config['name']}")
            return None

        plots_dir = Path(self.config['paths']['plots_dir']) / 'backtest'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get all available tickers
        tickers = list(self.config['data']['tickers'].keys())
        metrics = ['Cumulative Return', 'Sharpe Ratio', 'Win Rate', 'Accuracy']

        # Initialize matrices
        results_data = []
        evaluator = ModelEvaluator()

        print(f"Running backtests for {len(tickers)} tickers...")

        for ticker in tickers:
            backtest_results = evaluator.confidence_backtest(model, ticker, self, plot_results=False, verbose=False)

            if backtest_results:
                strategy = backtest_results['strategy']
                baseline = backtest_results['buy_hold_baseline']

                results_data.append({
                    'Ticker': ticker,
                    'Cumulative Return': strategy['cumulative_return'],
                    'Sharpe Ratio': strategy['sharpe_ratio'],
                    'Win Rate': strategy['win_rate'],
                    'Accuracy': strategy['accuracy'],
                    'Baseline Return': baseline['cumulative_return'],
                    'Baseline Sharpe': baseline['sharpe_ratio']
                })
            else:
                # Handle failed backtests
                results_data.append({
                    'Ticker': ticker,
                    'Cumulative Return': 0.0,
                    'Sharpe Ratio': 0.0,
                    'Win Rate': 0.0,
                    'Accuracy': 0.0,
                    'Baseline Return': 0.0,
                    'Baseline Sharpe': 0.0
                })

        # Create performance matrix (1 = outperforms baseline, 0 = underperforms)
        performance_matrix = np.zeros((len(tickers), len(metrics)))

        for i, row in enumerate(results_data):
            # Compare against appropriate baselines
            performance_matrix[i, 0] = 1 if row['Cumulative Return'] > row['Baseline Return'] else 0
            performance_matrix[i, 1] = 1 if row['Sharpe Ratio'] > row['Baseline Sharpe'] else 0
            performance_matrix[i, 2] = 1 if row['Win Rate'] > 0.5 else 0  # Random chance
            performance_matrix[i, 3] = 1 if row['Accuracy'] > 0.5 else 0  # Random chance

        # Create heatmap using same color scheme as confusion matrix
        colors = ['#ffcccc', '#ccffcc']  # Light red, light green
        cmap = plt.matplotlib.colors.ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(performance_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(tickers)

        # Add value annotations
        for i in range(len(tickers)):
            for j in range(len(metrics)):
                row = results_data[i]

                # Format value display based on metric
                if j == 0:  # Cumulative Return
                    value = row['Cumulative Return']
                    baseline = row['Baseline Return']
                    text = f'{value:.2f}\n(vs {baseline:.2f})'
                elif j == 1:  # Sharpe Ratio
                    value = row['Sharpe Ratio']
                    baseline = row['Baseline Sharpe']
                    text = f'{value:.2f}\n(vs {baseline:.2f})'
                elif j == 2:  # Win Rate
                    value = row['Win Rate']
                    text = f'{value:.2f}\n(vs 0.50)'
                else:  # Accuracy
                    value = row['Accuracy']
                    text = f'{value:.2f}\n(vs 0.50)'

                color = 'black' if performance_matrix[i, j] == 1 else 'darkred'
                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontweight='bold', fontsize=8)

        ax.set_title(f'{model.model_class_name.upper()} Trading Performance by Ticker\n(Green = Outperforms Baseline, Red = Underperforms)')
        ax.set_xlabel('Trading Metrics')
        ax.set_ylabel('Tickers')

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'{model.model_class_name}_backtest_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Backtest heatmap for {len(tickers)} tickers completed")
        return str(plot_path)

    def plot_confusion_matrix(self, model):
        """Plot confusion matrix for a single direction_1d model"""
        plots_dir = Path(self.config['paths']['plots_dir']) / 'classification_metrics'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions and actual values
        y_pred = model.predict()
        y_test = model.y_test

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_percent = cm / cm.sum() * 100

        # Create custom color matrix with more contrast
        colors = np.zeros((2, 2, 3))  # RGB colors for each cell

        for i in range(2):
            for j in range(2):
                saturation = cm_percent[i,j] / 60.0
                if i == j:  # Correct predictions (main diagonal: 0/0, 1/1)
                    # Green-hued with more contrast
                    colors[i,j] = [0.7 - saturation*0.5, 0.85 - saturation*0.4, 0.7 - saturation*0.5]
                else:  # Confusion (off-diagonal: 0/1, 1/0)
                    # Red-hued with more contrast
                    colors[i,j] = [0.9 - saturation*0.4, 0.7 - saturation*0.6, 0.7 - saturation*0.6]

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(colors, aspect='equal', origin='lower')

        # Add borders between sectors
        ax.axhline(y=0.5, color='black', linewidth=2)
        ax.axvline(x=0.5, color='black', linewidth=2)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)',
                       ha='center', va='center', fontsize=14, fontweight='bold')

        # Set labels and ticks
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Down (0)', 'Up (1)'])
        ax.set_yticklabels(['Down (0)', 'Up (1)'])
        ax.set_xlabel('Predicted Direction')
        ax.set_ylabel('Actual Direction')
        ax.set_title(f'Confusion Matrix - {model.model_class_name.upper()} Model')

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / f'{model.model_class_name}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def roc_curves(self, models):
        """Plot ROC curves for multiple direction_1d models"""

        plots_dir = Path(self.config['paths']['plots_dir']) / 'classification_metrics'
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, model in enumerate(models):
            # Get predictions and actual values
            y_pred = model.predict()
            y_test = model.y_test

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label=f'{model.model_class_name.upper()} (AUC = {roc_auc:.3f})')

        # Add diagonal line for random chance
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
                label='Random Chance (AUC = 0.500)')

        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Direction Prediction Models')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / 'roc_curves_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)