import sys
from pathlib import Path
import pandas as pd
from config import CONFIG
from data import DataDownloader, DataPreprocessor, FeatureEngineer
from models import LinearModel, ARIMAModel, XGBoostModel, LSTMModel, CNNModel
from evaluation import ModelEvaluator
from utils import ModelIO


class StockPredictionCLI:
    def __init__(self):
        self.config = CONFIG
        self.model_io = ModelIO(CONFIG)
        self.evaluator = ModelEvaluator(CONFIG)

        # Model registry
        self.model_classes = {
            'linear': LinearModel,
            'arima': ARIMAModel,
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
        print("\nAvailable commands:")
        print("  models     - Show model status")
        print("  train      - Train a model (e.g., 'train xgboost')")
        print("  evaluate   - Evaluate models (e.g., 'evaluate' or 'evaluate lstm xgboost')")
        print("  help       - Show this help message")
        print("  exit       - Exit program")

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
        """Load saved models and initialize pre-trained ones"""
        print("\nLoading models...")

        for model_name in self.model_classes:
            try:
                # Try to load saved model
                model = self.model_io.load_model(model_name)
                self.trained_models[model_name] = model
                print(f"  ✓ {model_name} loaded")
            except FileNotFoundError:
                # Check if it's a pre-trained model that should be initialized
                if self._is_pretrained_available(model_name):
                    model = self.model_classes[model_name](self.config)
                    model.load_pretrained_from_web()  # Assuming this method exists
                    self.trained_models[model_name] = model
                    print(f"  ✓ {model_name} initialized (pre-trained)")
                else:
                    print(f"  ✗ {model_name} not found")

    def _is_pretrained_available(self, model_name):
        """Check if pre-trained weights are available online"""
        # Placeholder - implement based on actual pre-trained model availability
        return False

    def cmd_models(self):
        """Show model status"""
        print("\nModel Status:")
        print("-" * 30)

        print("Trained models:")
        if self.trained_models:
            for name in self.trained_models:
                print(f"  • {name}")
        else:
            print("  None")

        print("\nUntrained models:")
        untrained = [name for name in self.model_classes if name not in self.trained_models]
        if untrained:
            for name in untrained:
                print(f"  • {name}")
        else:
            print("  None")

    def cmd_train(self, model_name):
        """Train a specific model"""
        if model_name not in self.model_classes:
            print(f"Error: Unknown model '{model_name}'")
            return

        if model_name in self.trained_models:
            print(f"Model '{model_name}' is already trained. Retraining...")

        print(f"\nTraining {model_name}...")

        # Initialize model
        model = self.model_classes[model_name](self.config)

        # Train for each target
        for target_config in self.config['targets']:
            target_name = target_config['name']
            print(f"  Training for target: {target_name}")

            # Prepare data
            X, y = model.prepare_data(self.data, target_name)
            splits = model.split_data(X, y, self.data.index)

            # Build model
            model.build_model(X.shape[1:])

            # Train
            history = model.train(
                splits['X_train'], splits['y_train'],
                splits['X_val'], splits['y_val']
            )

        # Save model
        self.model_io.save_model(model, model_name, metadata={'targets': [t['name'] for t in self.config['targets']]})
        self.trained_models[model_name] = model
        print(f"Model '{model_name}' trained and saved.")

    def cmd_evaluate(self, model_names=None):
        """Evaluate trained models"""
        if model_names:
            # Evaluate specific models
            models_to_eval = []
            for name in model_names:
                if name in self.trained_models:
                    models_to_eval.append(name)
                else:
                    print(f"Warning: Model '{name}' is not trained, skipping.")
        else:
            # Evaluate all trained models
            models_to_eval = list(self.trained_models.keys())

        if not models_to_eval:
            print("No trained models to evaluate.")
            return

        print(f"\nEvaluating models: {', '.join(models_to_eval)}")
        print("-" * 50)

        for model_name in models_to_eval:
            model = self.trained_models[model_name]
            print(f"\n{model_name.upper()}:")

            # Evaluate for each target
            for target_config in self.config['targets']:
                target_name = target_config['name']
                print(f"  Target: {target_name}")

                # Get test data
                X, y = model.prepare_data(self.data, target_name)
                splits = model.split_data(X, y, self.data.index)

                # Evaluate
                results = self.evaluator.evaluate_model(
                    model,
                    splits['X_test'],
                    splits['y_test'],
                    target_config['type']
                )

                # Display results
                for metric, value in results.items():
                    print(f"    {metric}: {value:.4f}")

    def run(self):
        """Main CLI loop"""
        self.startup()

        commands = {
            'models': lambda args: self.cmd_models(),
            'train': lambda args: self.cmd_train(args[0]) if args else print("Error: Specify model name"),
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
