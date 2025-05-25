from pathlib import Path
import pandas as pd
from config import CONFIG
from data import DataDownloader, DataPreprocessor, FeatureEngineer
from models import LinearModel, ARIMAModel, XGBoostModel, LSTMModel, CNNModel
from evaluation import calculate_metrics, ModelEvaluator
from utils import ModelIO


def main():
    # Initialize components
    downloader = DataDownloader(CONFIG)
    preprocessor = DataPreprocessor(CONFIG)
    feature_engineer = FeatureEngineer(CONFIG)
    model_io = ModelIO(CONFIG)

    # Download data
    print("Downloading data...")
    stock_data = downloader.download_stock_data(
        CONFIG['data']['tickers'],
        CONFIG['data']['start_date'],
        CONFIG['data']['end_date']
    )
    fred_data = downloader.download_fred_data(
        CONFIG['data']['fred_indicators'],
        CONFIG['data']['start_date'],
        CONFIG['data']['end_date']
    )

    # Preprocess data
    print("Preprocessing data...")
    df = preprocessor.clean_data(stock_data, fred_data)
    df = preprocessor.create_targets(df)
    df = preprocessor.create_regime_features(df)

    # Feature engineering
    print("Creating features...")
    df = feature_engineer.create_technical_features(df)
    df = feature_engineer.create_sector_features(df)

    # Save processed data
    processed_path = Path(CONFIG['paths']['processed_dir'])
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path / 'processed_data.csv')

    # Initialize models
    models = {
        'linear': LinearModel(CONFIG),
        'xgboost': XGBoostModel(CONFIG),
        'lstm': LSTMModel(CONFIG),
        # Add more models as implemented
    }

    # Train and evaluate each model
    results = {}
    for target_config in CONFIG['targets']:
        target_name = target_config['name']
        print(f"\nTraining models for target: {target_name}")

        for model_name, model in models.items():
            print(f"  Training {model_name}...")

            # TODO: Implement training pipeline
            # - Prepare data for model
            # - Split data
            # - Train model
            # - Evaluate
            # - Save model and results

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
