import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
from config import CONFIG
from data import DataDownloader, DataPreprocessor, FeatureEngineer, DataSplitter
from models import BaseModel, LinearModel, ARIMAModel, DecisionTreeModel, XGBoostModel, LSTMModel, GRUModel, CNNModel
from evaluation import ModelEvaluator
from utils import ModelIO, ModelPlotter


class StockPredictionCLI:
    def __init__(self):
        self.config = CONFIG
        self.model_io = ModelIO(CONFIG)
        self.evaluator = ModelEvaluator()
        self.plotter = ModelPlotter(CONFIG)
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
            'line': 'line_chart'
        }

        self.trained_models = {}
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
        print("\n" + "-" * 50)
        self.print_commands()

    def print_commands(self):
        """Print available commands"""
        print("-" * 50)
        print("\nAvailable commands:")
        print("  models     - Show model status")
        print("  train      - Train a model on a specific target (e.g., 'train xgboost return_1d' or 'train linear')")
        print("  evaluate   - Evaluate models (e.g., 'evaluate' or 'evaluate lstm xgboost')")
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
        processed_path = Path(self.config['paths']['processed_dir']) / 'processed_data.csv'

        if processed_path.exists():
            print("Loading preprocessed data...")
            self.data = pd.read_csv(processed_path, index_col=0, parse_dates=True)
            print(f"Data loaded: {len(self.data)} rows")
        else:
            print("No preprocessed data found. Preprocessing...")

            # Initialize components
            downloader = DataDownloader(self.config)
            preprocessor = DataPreprocessor(self.config)
            feature_engineer = FeatureEngineer(self.config)

            # Download data
            stock_data = downloader.download_stock_data(
                self.config['data']['tickers'],
                self.config['data']['start_date'],
                self.config['data']['end_date']
            )
            fred_data = downloader.download_fred_data(
                self.config['data']['fred_indicators'],
                self.config['data']['start_date'],
                self.config['data']['end_date']
            )

            # Preprocess
            df = preprocessor.clean_data(stock_data, fred_data)
            df = preprocessor.create_targets(df)
            df = preprocessor.create_regime_features(df)

            # Feature engineering
            df = feature_engineer.create_technical_features(df)
            df = feature_engineer.create_sector_features(df)

            # Save
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path)
            self.data = df
            print("Data preprocessed and saved.")

    def _load_models(self):
        """Load saved trained models"""
        print("\nLoading models...")

        for model_class in self.model_classes:
            for target_name in self.target_dict:
                model_name = f'{model_class}_{target_name}'
                if self.model_io.model_exists(model_name):
                    model_data = self.model_io.load_model(model_name, self.target_dict[target_name]['type'])

                    model = self._init_model(model_class, model_name, target_name)
                    model.model = model_data['model']

                    if hasattr(model, 'scaler'):
                        model.scaler = model_data['scaler']

                    self.trained_models[model_name] = {
                        'model': model,
                        'class': model_class,
                        'target': target_name
                    }
                    print(f"  ✓ {model_name} loaded")

        print(f"Loaded {len(self.trained_models)} models.")

    def _init_model(self, model_class_name: str, model_name: str, target_name: str) -> BaseModel:
        """Initialize model based on model name"""
        model = self.model_classes[model_class_name](model_name, self.config['data'], self.config['models'], self.target_dict[target_name])
        return model

    def cmd_models(self):
        """Show model status"""
        model_class_targets = defaultdict(list)

        for info in self.trained_models.values():
            model_class = info['class']
            target_name = info['target']
            model_class_targets[model_class].append(target_name)

        print("\nModel Status:")
        print("-" * 50)

        print("Trained models:")
        if model_class_targets:
            for model_class, target_names in model_class_targets.items():
                print(f"  • {model_class}: {', '.join(target_names)}")
        else:
            print("  None")

        print("\nUntrained models:")
        found = False
        for model_class in self.model_classes:
            if model_class not in model_class_targets and self.model_classes[model_class].is_implemented:
                print(f"  • {model_class}")
                found = True

        if not found:
            print("  None")

    def cmd_train(self, model_class_name, target_name=None):
        """Train a specific model for a specific target (or all if not specified)"""
        if model_class_name not in self.model_classes:
            print(f"Error: Unknown model '{model_class_name}'")
            return

        if target_name and target_name not in self.target_dict:
            print(f"Error: Unknown target '{target_name}'")
            return

        target_names = self.target_dict.keys() if not target_name else [target_name]

        if model_class_name == 'arima': # ARIMA can only be used for 1-day return prediction
            if target_name != 'return_1d':
                print("Warning: ARIMA can only be used for 1-day return prediction, changing target to 'return_1d'")
            target_names = ['return_1d']

        # Train for each target
        for target_name in target_names:
            model_name = f'{model_class_name}_{target_name}'

            if model_name in self.trained_models.keys():
                print(f"Model '{model_class_name}' is already trained for target '{target_name}'. Retraining...")
            else:
                print(f"\nTraining '{model_class_name}' for target: '{target_name}'.")

            model = self._init_model(model_class_name, model_name, target_name)

            # Use date-based preparation
            from data.data_splitter import DataSplitter
            splitter = DataSplitter(self.config['data'])
            split_dates = splitter.get_split_dates()

            # Prepare training data (train + val periods)
            train_data = splitter.filter_data_by_dates(self.data, split_dates['train_start'], split_dates['train_end'])
            val_data = splitter.filter_data_by_dates(self.data, split_dates['val_start'], split_dates['val_end'])

            # Prepare data for training
            X_train, y_train = model.prepare_data(train_data)
            X_val, y_val = model.prepare_data(val_data)

            # Build model
            model.build_model()

            # Train
            history = model.train(X_train, y_train, X_val, y_val)

            # Save model
            self.model_io.save_model(model, model_name)
            self.trained_models[model_name] = {'model': model, 'history': history, 'class': model_class_name, 'target': target_name}
            print(f"Model '{model_name}' trained and saved.")

    def cmd_evaluate(self, model_names=None):
        """Evaluate trained models with two-tier analysis"""
        # If no specific models requested, evaluate all trained models
        if not model_names:
            models_to_evaluate = self.trained_models
        else:
            # Build dict from provided model class names
            models_to_evaluate = {}
            for model_class in model_names:
                for model_name, info in self.trained_models.items():
                    if info['class'] == model_class:
                        models_to_evaluate[model_name] = info

        if not models_to_evaluate:
            print("No models to evaluate.")
            return

        print(f"\nEvaluating {len(models_to_evaluate)} models with two-tier analysis...")
        print("-" * 50)

        # Use new two-tier evaluation
        results = self.evaluator.evaluate_models_two_tier(self.config['data'], models_to_evaluate, self.data)

        # Display summary
        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")

        for model_name, result in results.items():
            print(f"\n{model_name}:")

            if result['tier1']:
                tier1 = result['tier1']
                print(f"  Tier 1 ({result['available_dates']} dates): ", end="")
                if 'directional_accuracy' in tier1:
                    print(f"RMSE={tier1['rmse']:.4f}, Dir.Acc={tier1['directional_accuracy']:.4f}")
                else:
                    print(f"Accuracy={tier1['accuracy']:.4f}")

            if result['tier2']:
                tier2 = result['tier2']
                print(f"  Tier 2 (common dates): ", end="")
                if 'directional_accuracy' in tier2:
                    print(f"RMSE={tier2['rmse']:.4f}, Dir.Acc={tier2['directional_accuracy']:.4f}")
                else:
                    print(f"Accuracy={tier2['accuracy']:.4f}")

    def cmd_plot(self, plot_type, model_class_name=None):
        """Generate plots for a model or all models"""

        if plot_type not in self.plot_methods:
            print(f"Error: Unknown plot type '{plot_type}'")
            print(f"Available plot types: {', '.join(self.plot_methods.keys())}")
            return

        # If no model specified, plot for all available models
        if model_class_name is None:
            available_models = []
            for model_name, model_info in self.trained_models.items():
                if model_name.endswith('_return_1d'):
                    model_class = model_info['class']
                    available_models.append(model_class)

            if not available_models:
                print("Error: No models trained for return_1d target")
                return

            print(f"\nGenerating {plot_type} plots for all available models: {', '.join(available_models)}")

            all_results = []
            for model_class in available_models:
                model_name = f'{model_class}_return_1d'
                model = self.trained_models[model_name]['model']

                print(f"  Plotting {model_class}...")
                plotter_method = getattr(self.plotter, self.plot_methods[plot_type])
                results = plotter_method(model)
                all_results.extend(results)

            print(f"\nSaved {len(all_results)} plots:")
            for result in all_results:
                print(f"  {result}")
            return

        # Single model case (existing logic)
        model_name = f'{model_class_name}_return_1d'
        if model_name not in self.trained_models:
            print(f"Error: Model '{model_class_name}' not trained for return_1d")
            return

        print(f"\nGenerating {plot_type} plots for {model_class_name}...")
        model = self.trained_models[model_name]['model']

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


def main():
    cli = StockPredictionCLI()
    cli.run()


if __name__ == "__main__":
    main()
