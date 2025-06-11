import sys
import traceback
from pathlib import Path
from collections import defaultdict
import pandas as pd
from config import CONFIG
from data import DataDownloader
from models import BaseModel, LinearModel, ARIMAModel, DecisionTreeModel, XGBoostModel, LSTMModel, GRUModel, CNNModel
from evaluation import ModelEvaluator
from utils import ModelIO, ModelPlotter


class StockPredictionCLI:
    def __init__(self):
        self.config = CONFIG
        self.model_io = ModelIO(self.config)
        self.evaluator = ModelEvaluator()
        self.plotter = ModelPlotter(self.config)
        self.target_dict = {t['name']: t for t in self.config['targets']}

        # Model registry
        self.model_classes = {
            'linear': LinearModel,
            'arima': ARIMAModel,
            'tree': DecisionTreeModel,
            'xgboost': XGBoostModel,
            'lstm': LSTMModel,
            'gru': GRUModel,
            'cnn': CNNModel
        }

        # Plot types and methods of ModelPlotter
        self.plot_methods = {
            'scatter': 'scatter',
            'line': 'line_chart',
            'bar': 'compare_regression_metrics',
            'class_heatmap': 'classification_heatmap',
            'return_heatmap': 'return_heatmap',
            'backtest_heatmap': 'backtest_heatmap',
            'regime': 'regime_heatmap',
            'confusion': 'plot_confusion_matrix',
            'roc': 'roc_curves',
        }

        self.models = {}
        self.data = None

    def startup(self):
        """Initialize data and models on startup"""
        print("Stock Market Prediction System")
        print("=" * 50)

        # Load or preprocess data
        self._load_data()
        self.plotter.data = self.data

        # Load saved models
        self._load_models()

        print("\nSystem ready.")
        self.print_commands()

    def print_commands(self):
        """Print available commands"""
        print("-" * 50)
        print("\nAvailable commands:")
        print("  models     - Show model status")
        print("  train      - Train a model on a specific target (e.g., 'train xgboost return_1d' or 'train linear')")
        print("  evaluate   - Evaluate models (e.g., 'evaluate' or 'evaluate lstm xgboost')")
        print("  backtest   - Backtest a model (e.g., 'backtest lstm')")
        print("  plot       - Generate plots (e.g., 'plot scatter' for all models or 'plot scatter xgboost')")
        print("  help       - Show this help message")
        print("  exit (q)   - Exit program")
        print("-" * 50)
        print(f"Available models: {', '.join(self.model_classes.keys())}")
        print(f"Available targets: {', '.join(self.target_dict.keys())}")
        print(f"Available plots: {', '.join(self.plot_methods.keys())}")
        print("-" * 50)

    def _load_data(self):
        """Load preprocessed data or create it if doesn't exist"""
        processed_path = Path(self.config['paths']['processed_dir'])

        # Check if individual ticker files exist
        ticker_files_exist = all(
            (processed_path / f'{ticker}_processed.csv').exists()
            for ticker in self.config['data']['tickers']
        )

        if ticker_files_exist:
            print("Loading preprocessed data...")
            self.data = {}
            for ticker in self.config['data']['tickers']:
                ticker_path = processed_path / f'{ticker}_processed.csv'
                self.data[ticker] = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
            print(f"Data loaded: {len(self.data)} tickers")
        else:
            print("Some data is missing. Preprocessing...")

            # Download data and create features
            downloader = DataDownloader(self.config)
            data_dict = downloader.get_data()

            # Save each ticker separately
            processed_path.mkdir(parents=True, exist_ok=True)
            for ticker, df in data_dict.items():
                df.to_csv(processed_path / f'{ticker}_processed.csv')

            self.data = data_dict
            print("Data preprocessed and saved per ticker.")

    def _load_models(self):
        """Load saved models and prepare data if needed"""
        print("\nLoading models...")

        for model_class in self.model_classes:
            for target_name in self.target_dict:
                model_name = f'{model_class}_{target_name}'

                # Initialize model
                model = self._init_model(model_class, model_name, target_name)

                if self.model_io.model_exists(model_name):
                    # Load trained model
                    model_data = self.model_io.load_model(model_name, self.target_dict[target_name]['type'])

                    # Restore model weights
                    model.model = model_data['model']
                    if hasattr(model, 'scaler') and 'scaler' in model_data:
                        model.scaler = model_data['scaler']

                    # Restore test targets and dates
                    model.y_test = model_data['y_test']

                    # Restore processed data
                    if model.get_model_type() != 'arima':
                        model.X_train = model_data['X_train']
                        model.y_train = model_data['y_train']
                        model.X_val = model_data['X_val']
                        model.y_val = model_data['y_val']
                        model.X_test = model_data['X_test']
                        model.X_test_index = model_data['X_test_index']
                        if model.get_model_type() == 'sequence':
                            model.categorical_vocab_sizes = model_data['categorical_vocab_sizes']
                            model.numerical_features = model_data['numerical_features']
                            model.categorical_features = model_data['categorical_features']
                    else:
                        model.ticker_data = model_data['ticker_data']

                    print(f"  ✓ {model_name} loaded")

                # Store model
                self.models[model_name] = {
                    'model': model, 'class': model_class, 'target': target_name, 'trained': self.model_io.model_exists(model_name)
                }

    def _init_model(self, model_class_name: str, model_name: str, target_name: str) -> BaseModel:
        """Initialize model based on model name"""
        model = self.model_classes[model_class_name](model_name, self.config, self.target_dict[target_name])
        return model

    def cmd_models(self):
        """Show model status"""
        print("\nModel Status:")
        print("-" * 50)

        trained_models = {info['class']: [] for info in self.models.values()}
        untrained_models = {info['class']: [] for info in self.models.values()}

        for model_name, info in self.models.items():
            if info['trained']:
                trained_models[info['class']].append(info['target'])
            else:
                untrained_models[info['class']].append(info['target'])

        print("Trained models:")
        for model_class, targets in trained_models.items():
            if targets:
                print(f"  • {model_class}: {', '.join(targets)}")

        print("\nUntrained models:")
        for model_class, targets in untrained_models.items():
            if targets:
                print(f"  • {model_class}: {', '.join(targets)}")

    def cmd_train(self, model_class_name, target_name=None):
        """Train a specific model for a specific target (or all if not specified)"""
        if model_class_name not in self.model_classes:
            print(f"Error: Unknown model '{model_class_name}'")
            return

        if target_name and target_name not in self.target_dict:
            print(f"Error: Unknown target '{target_name}'")
            return

        target_names = self.target_dict.keys() if not target_name else [target_name]

        # Train for each target
        for target_name in target_names:
            model_name = f'{model_class_name}_{target_name}'

            if model_name not in self.models:
                print(f"Error: Model '{model_name}' not found.")
                continue

            model = self.models[model_name]['model']

            if self.models[model_name]['trained']:
                print(f"Model '{model_name}' already trained. Retraining...")
            else:
                print(f"Training '{model_name}'...")
            print(f"  Preparing data for {model_name}")
            model.prepare_data(self.data)

            model.build_model()
            model.train()

            self.model_io.save_model(model, model_name)
            self.models[model_name]['trained'] = True
            print(f"✓ {model_name} trained and saved.")

    def cmd_evaluate(self, model_names=None):
        """Evaluate trained models"""
        # Select models to evaluate
        if not model_names:
            models_to_evaluate = {name: info for name, info in self.models.items() if info['trained']}
        else:
            models_to_evaluate = {}
            for model_class in model_names:
                for model_name, info in self.models.items():
                    if info['class'] == model_class and info['trained']:
                        models_to_evaluate[model_name] = info

        if not models_to_evaluate:
            print("No trained models to evaluate.")
            return

        print(f"\nEvaluating {len(models_to_evaluate)} models...")
        print("-" * 50)

        # Evaluate each model
        results = {}

        for model_name, model_info in models_to_evaluate.items():
            model = model_info['model']
            results[model_name] = self.evaluator.evaluate_model(model)

        # Display results
        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")

        for model_name, result in results.items():
            print(f"\n{model_name}:")
            if 'directional_accuracy' in result:
                print(f"  RMSE: {result['rmse']:.4f}")
                print(f"  Directional Accuracy: {result['directional_accuracy']:.4f}")
            else:
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  F1 Score: {result['f1_score']:.4f}")

    def cmd_backtest(self, model_class_name):
        """Run confidence-based backtest for direction model"""
        model_name = f'{model_class_name}_direction_1d'

        if model_name not in self.models or not self.models[model_name]['trained']:
            print(f"Error: Model '{model_class_name}' not trained for direction_1d")
            return

        model = self.models[model_name]['model']
        results = self.evaluator.confidence_backtest(model, 'TSLA', self.plotter, True)

        if results:
            print(f"\nConfidence-based backtest completed for {model_class_name}")

    def cmd_plot(self, plot_type, model_class_name=None):
        """Generate plots for a model or all models"""
        if plot_type not in self.plot_methods:
            print(f"Error: Unknown plot type '{plot_type}'")
            print(f"Available plot types: {', '.join(self.plot_methods.keys())}")
            return

        if plot_type in ['class_heatmap', 'roc']:
            # Get all available models trained on direction_1d
            available_models = []
            for model_name, model_info in self.models.items():
                if model_name.endswith('_direction_1d') and model_info['trained']:
                    available_models.append(model_info['model'])

            if len(available_models) < 2:
                print("Error: Need at least 2 trained models for classification comparison")
                return

            print(f"\nGenerating {plot_type} plot for {len(available_models)} models...")
            plotter_method = getattr(self.plotter, self.plot_methods[plot_type])
            result = plotter_method(available_models)
            print(f"Saved plot to: {result}")
            return

        if plot_type == 'return_heatmap':
            # Get available return models grouped by target type
            return_1d_models = []
            return_5d_models = []

            for model_name, model_info in self.models.items():
                if model_name.endswith('_return_1d') and model_info['trained']:
                    return_1d_models.append(model_info['model'])
                elif model_name.endswith('_return_5d') and model_info['trained']:
                    return_5d_models.append(model_info['model'])

            if return_1d_models:
                print(f"\nGenerating return_1d metrics heatmap for {len(return_1d_models)} models...")
                result_1d = self.plotter.return_heatmap(return_1d_models)
                print(f"Saved 1d heatmap to: {result_1d}")

            if return_5d_models:
                print(f"\nGenerating return_5d metrics heatmap for {len(return_5d_models)} models...")
                result_5d = self.plotter.return_heatmap(return_5d_models)
                print(f"Saved 5d heatmap to: {result_5d}")

            return

        if plot_type == 'backtest_heatmap':
            if not model_class_name:
                print("Error: backtest_heatmap requires model class name (e.g., 'plot backtest_heatmap xgboost')")
                return

            model_name = f'{model_class_name}_direction_1d'
            if model_name not in self.models or not self.models[model_name]['trained']:
                print(f"Error: Model '{model_class_name}' not trained for direction_1d")
                return

            print(f"\nGenerating backtest heatmap for {model_class_name}...")
            model = self.models[model_name]['model']
            result = self.plotter.backtest_heatmap(model)
            print(f"Saved plot to: {result}")
            return

        if plot_type == 'scatter':
            # Get available return models grouped by target type
            return_1d_models = []
            return_5d_models = []

            for model_name, model_info in self.models.items():
                if model_name.endswith('_return_1d') and model_info['trained']:
                    return_1d_models.append(model_info['model'])
                elif model_name.endswith('_return_5d') and model_info['trained']:
                    return_5d_models.append(model_info['model'])

            results = []
            if return_1d_models:
                print(f"\nGenerating 1-day return scatter plots for {len(return_1d_models)} models...")
                result_1d = self.plotter.scatter(return_1d_models)
                results.append(result_1d)
                print(f"Saved 1d scatter to: {result_1d}")

            if return_5d_models:
                print(f"\nGenerating 5-day return scatter plots for {len(return_5d_models)} models...")
                result_5d = self.plotter.scatter(return_5d_models)
                results.append(result_5d)
                print(f"Saved 5d scatter to: {result_5d}")

            if not results:
                print("No return models available for scatter plots")

            return

        if plot_type == 'bar':
            # Get all available models trained on return_1d and return_5d
            available_models = []
            model_classes_found = set()

            for model_name, model_info in self.models.items():
                if (model_name.endswith('_return_1d') or model_name.endswith('_return_5d')) and model_info['trained']:
                    available_models.append(model_info['model'])
                    model_classes_found.add(model_info['class'])

            if len(available_models) < 2:
                print("Error: Need at least 2 trained models for RMSE comparison")
                return

            print(f"\nGenerating RMSE comparison for {len(available_models)} models ({len(model_classes_found)} model classes)...")
            results = self.plotter.compare_regression_metrics(available_models)
            print(f"\nSaved {len(results)} plots:")
            for result in results:
                print(f"  {result}")
            return

        if plot_type == 'confusion':
            model_name = f'{model_class_name}_direction_1d'
            if model_name not in self.models or not self.models[model_name]['trained']:
                print(f"Error: Model '{model_class_name}' not trained for direction_1d")
                return

            print(f"\nGenerating confusion matrix for {model_class_name}...")
            model = self.models[model_name]['model']
            result = self.plotter.plot_confusion_matrix(model)
            print(f"Saved plot to: {result}")
            return

        if plot_type == 'regime':
            if not model_class_name:
                print("Error: regime plot requires model class name (e.g., 'plot regime xgboost')")
                return

            model_1d_name = f'{model_class_name}_return_1d'
            model_5d_name = f'{model_class_name}_return_5d'

            if (model_1d_name not in self.models or not self.models[model_1d_name]['trained'] or
                model_5d_name not in self.models or not self.models[model_5d_name]['trained']):
                print(f"Error: Both {model_class_name} models (return_1d and return_5d) must be trained")
                return

            print(f"\nGenerating regime heatmap for {model_class_name}...")
            model_1d = self.models[model_1d_name]['model']
            model_5d = self.models[model_5d_name]['model']
            result = self.plotter.regime_heatmap(model_1d, model_5d)
            print(f"Saved plot to: {result}")
            return

        # If no model specified, plot for all available models
        if model_class_name is None:
            available_models = []
            for model_name, model_info in self.models.items():
                if model_name.endswith('_return_1d') and model_info['trained']:
                    model_class = model_info['class']
                    available_models.append(model_class)

            if not available_models:
                print("Error: No models trained for return_1d target")
                return

            print(f"\nGenerating {plot_type} plots for all available models: {', '.join(available_models)}")
            all_results = []
            for model_class in available_models:
                model_name = f'{model_class}_return_1d'
                model = self.models[model_name]['model']
                print(f"  Plotting {model_class}...")
                plotter_method = getattr(self.plotter, self.plot_methods[plot_type])
                results = plotter_method(model)
                all_results.extend(results)

            print(f"\nSaved {len(all_results)} plots:")
            for result in all_results:
                print(f"  {result}")
            return

        # Single model case
        model_name = f'{model_class_name}_return_1d'
        if model_name not in self.models or not self.models[model_name]['trained']:
            print(f"Error: Model '{model_class_name}' not trained for return_1d")
            return

        print(f"\nGenerating {plot_type} plots for {model_class_name}...")
        model = self.models[model_name]['model']
        plotter_method = getattr(self.plotter, self.plot_methods[plot_type])
        results = plotter_method(model)
        print(f"Saved plot to: {results[0]}")

    def run(self):
        """Main CLI loop"""
        self.startup()

        commands = {
            'models': lambda args: self.cmd_models(),
            'train': lambda args: self.cmd_train(args[0], args[1]) if len(args) > 1 else self.cmd_train(args[0]),
            'evaluate': lambda args: self.cmd_evaluate(args) if len(args) > 0 else self.cmd_evaluate(),
            'backtest': lambda args: self.cmd_backtest(args[0]),
            'help': lambda args: self.print_commands(),
            'plot': lambda args: (
                self.cmd_plot(args[0], args[1]) if len(args) > 1
                else self.cmd_plot(args[0]) if len(args) == 1
                else print("Usage: plot <type> [model]")
            ),
            'exit': lambda args: sys.exit(0),
            'q': lambda args: sys.exit(0)
        }

        while True:
            try:
                user_input = input("\n> ").strip().lower().split()

                if not user_input:
                    continue

                cmd = user_input[0]
                args = user_input[1:]

                if cmd in commands:
                    commands[cmd](args)
                else:
                    print(f"Unknown command: '{cmd}'")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()


def main():
    cli = StockPredictionCLI()
    cli.run()


if __name__ == "__main__":
    main()
