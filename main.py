import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
from config import CONFIG
from data import DataDownloader, DataPreprocessor, FeatureEngineer
from models import BaseModel, LinearModel, ARIMAModel, DecisionTreeModel, XGBoostModel, LSTMModel, CNNModel
from evaluation import ModelEvaluator
from utils import ModelIO


class StockPredictionCLI:
    def __init__(self):
        self.config = CONFIG
        self.model_io = ModelIO(CONFIG)
        self.evaluator = ModelEvaluator()
        self.target_dict = {t['name']: t for t in self.config['targets']}

        # Model registry
        self.model_classes = {
            'linear': LinearModel,
            'arima': ARIMAModel,
            'tree': DecisionTreeModel,
            'xgboost': XGBoostModel,
            'lstm': LSTMModel,
            'cnn': CNNModel
        }

        self.trained_models = {}
        self.data = None

    def startup(self):
        """Initialize data and models on startup"""
        print("Stock Market Prediction System")
        print("=" * 50)

        # Load or preprocess data
        self._load_data()

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
        print("  help       - Show this help message")
        print("  exit       - Exit program")
        print("-" * 50)
        print(f"Available models: {', '.join(self.model_classes.keys())}")
        print(f"Available targets: {', '.join(self.target_dict.keys())}")
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
                    model_data = self.model_io.load_model(model_name)

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
        model = self.model_classes[model_class_name](model_name, self.config['models'], self.target_dict[target_name])
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
            print(f"Error: Unknown model '{model_name}'")
            return

        if target_name and target_name not in self.target_dict:
            print(f"Error: Unknown target '{target_name}'")
            return

        target_names = self.target_dict.keys() if not target_name else [target_name]

        # Train for each target
        for target_name in target_names:
            model_name = f'{model_class_name}_{target_name}'

            if model_name in self.trained_models.keys():
                print(f"Model '{model_class_name}' is already trained for target '{target_name}'. Retraining...")
            else:
                print(f"\nTraining '{model_class_name}' for target: '{target_name}'.")

            model = self._init_model(model_class_name, model_name, target_name)

            # Prepare data
            X, y = model.prepare_data(self.data)
            splits = model.split_data(X, y, self.data.index)

            # Build model
            model.build_model()

            # Train
            history = model.train(
                splits['X_train'], splits['y_train'],
                splits['X_val'], splits['y_val']
            )

            # Save model
            self.model_io.save_model(model, model_name)
            self.trained_models[model_name] = {'model': model, 'history': history, 'class': model_class_name, 'target': target_name}
            print(f"Model '{model_name}' trained and saved.")

    def cmd_evaluate(self, model_names=None):
        """Evaluate trained models with enhanced metrics"""
        # If no specific models requested, evaluate all trained models
        if not model_names:
            models_to_evaluate = list(self.trained_models.keys())
        else:
            # Build full model names from provided model class names
            models_to_evaluate = []
            for model_class in model_names:
                for model_name, info in self.trained_models.items():
                    if info['class'] == model_class:
                        models_to_evaluate.append(model_name)

        if not models_to_evaluate:
            print("No models to evaluate.")
            return

        print(f"\nEvaluating {len(models_to_evaluate)} models...")
        print("-" * 50)

        for model_name in models_to_evaluate:
            model_info = self.trained_models[model_name]
            model = model_info['model']

            # Prepare test data
            X, y = model.prepare_data(self.data)
            splits = model.split_data(X, y, self.data.index)

            # Get test data subset for regime analysis
            test_data = self.data.iloc[splits['idx_test']]

            # Evaluate with enhanced metrics
            metrics = self.evaluator.evaluate_model(
                model,
                splits['X_test'],
                splits['y_test'],
                test_data
            )

            # Display results
            print(f"\n{model_name}:")
            print(f"  Test samples: {len(splits['X_test'])}")

            if model.target_config['type'] == 'regression':
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  Directional Accuracy: {metrics['directional_accuracy']:.4f}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"  Hit Rate: {metrics['hit_rate']:.4f}")
            else:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")

            # Show regime analysis if available
            if 'regime_analysis' in metrics:
                print("\n  Performance by regime:")

                if 'volatility_regimes' in metrics['regime_analysis']:
                    print("    Volatility:")
                    for regime, data in metrics['regime_analysis']['volatility_regimes'].items():
                        print(f"      {regime}: {data['key_metric']:.4f} (n={data['n_samples']})")

                if 'sectors' in metrics['regime_analysis']:
                    print("    Sectors:")
                    for sector, data in metrics['regime_analysis']['sectors'].items():
                        print(f"      {sector}: {data['key_metric']:.4f} (n={data['n_samples']})")

    def run(self):
        """Main CLI loop"""
        self.startup()

        commands = {
            'models': lambda args: self.cmd_models(),
            'train': lambda args: self.cmd_train(args[0], args[1]) if len(args) > 1 else self.cmd_train(args[0]),
            'evaluate': lambda args: self.cmd_evaluate(args) if len(args) > 0 else self.cmd_evaluate(),
            'help': lambda args: self.print_commands(),
            'exit': lambda args: sys.exit(0)
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
