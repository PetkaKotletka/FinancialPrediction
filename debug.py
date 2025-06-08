import sys
import pandas as pd
import numpy as np
from pathlib import Path

from config import CONFIG
from data import DataDownloader, DataPreprocessor, FeatureEngineer
from models import ARIMAModel, LSTMModel, GRUModel, CNNModel
from main import StockPredictionCLI
from utils import ModelIO

def load_data():
    """Load preprocessed data"""
    processed_path = Path(CONFIG['paths']['processed_dir']) / 'processed_data.csv'
    if processed_path.exists():
        return pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        print("No preprocessed data found!")
        return None

def debug_model_predictions(model_name, target_name='return_1d'):
    """Debug predictions for a specific model"""
    print(f"\n{'='*50}")
    print(f"DEBUGGING: {model_name}_{target_name}")
    print(f"{'='*50}")

    # Load data
    data = load_data()
    if data is None:
        return

    # Initialize model IO
    model_io = ModelIO(CONFIG)
    full_model_name = f'{model_name}_{target_name}'

    # Check if model exists
    if not model_io.model_exists(full_model_name):
        print(f"Model {full_model_name} not found!")
        return

    # Load target config
    target_dict = {t['name']: t for t in CONFIG['targets']}
    target_config = target_dict[target_name]

    # Initialize model
    model_classes = {
        'arima': ARIMAModel,
        'lstm': LSTMModel,
        'gru': GRUModel,
        'cnn': CNNModel
    }

    model = model_classes[model_name](full_model_name, CONFIG['models'], target_config)

    # Load trained model
    model_data = model_io.load_model(full_model_name, target_config['type'])
    model.model = model_data['model']
    if hasattr(model, 'scaler') and 'scaler' in model_data:
        model.scaler = model_data['scaler']

    # Prepare data
    X, y = model.prepare_data(data)
    splits = model.split_data(X, y, data.index)

    # Make predictions on test set
    X_test = splits['X_test']
    y_test = splits['y_test']

    print(f"Test set size: {len(X_test)}")
    print(f"Input shape: {X_test.shape}")

    # Get predictions
    y_pred = model.predict(X_test)

    print(f"Predictions shape: {y_pred.shape}")
    print(f"\nFirst 10 predictions:")
    print(y_pred[:10])

    print(f"\nFirst 10 actual values:")
    print(y_test[:10])

    print(f"\nPrediction statistics:")
    print(f"  Min: {np.min(y_pred):.6f}")
    print(f"  Max: {np.max(y_pred):.6f}")
    print(f"  Mean: {np.mean(y_pred):.6f}")
    print(f"  Std: {np.std(y_pred):.6f}")
    print(f"  Zeros: {np.sum(y_pred == 0)} out of {len(y_pred)}")
    print(f"  Near-zeros (abs < 1e-6): {np.sum(np.abs(y_pred) < 1e-6)} out of {len(y_pred)}")

    print(f"\nActual values statistics:")
    print(f"  Min: {np.min(y_test):.6f}")
    print(f"  Max: {np.max(y_test):.6f}")
    print(f"  Mean: {np.mean(y_test):.6f}")
    print(f"  Std: {np.std(y_test):.6f}")

def debug_lstm_model():
    """Detailed LSTM debugging"""

    cli = StockPredictionCLI()
    cli._load_data()
    cli._load_models()

    # Get LSTM model
    model_name = 'lstm_return_1d'
    if model_name not in cli.trained_models:
        print(f"Model {model_name} not found!")
        return

    model = cli.trained_models[model_name]['model']

    # Check model weights
    for layer in model.model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                print(f"{layer.name} - Weight stats:")
                for i, w in enumerate(weights):
                    print(f"  Weight {i}: mean={np.mean(w):.6f}, std={np.std(w):.6f}")

    # Test on single ticker
    test_data = cli.data[cli.data['ticker'] == 'AAPL'].iloc[-100:]
    X, y = model.prepare_data(test_data)

    if len(X) > 0:
        # Check raw vs scaled data
        X_sample = X[:5]
        n_samples, n_timesteps, n_features = X_sample.shape
        X_reshaped = X_sample.reshape(-1, n_features)
        X_scaled = model.scaler.transform(X_reshaped)

        print(f"\nRaw data sample: {X_reshaped[0, :5]}")
        print(f"Scaled data sample: {X_scaled[0, :5]}")

        # Get predictions
        predictions = model.predict(X_sample)
        print(f"\nSample predictions: {predictions}")
        print(f"Sample actuals: {y[:5]}")

def main():
    """Debug problematic models"""
    problematic_models = ['arima', 'lstm', 'gru', 'cnn']

    for model_name in problematic_models:
        debug_model_predictions(model_name)

    print(f"\n{'='*50}")
    print("DEBUG COMPLETE")
    print(f"{'='*50}")

if __name__ == "__main__":
    debug_lstm_model()