import pickle
import joblib
from pathlib import Path
import tensorflow as tf
import json

from models import BaseModel


class ModelIO:
    def __init__(self, config: dict):
        self.models_dir = Path(config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str):
        """Save model with minimal necessary data"""
        model_path = self.models_dir / model_name

        model_data = { 'y_test': model.y_test }

        # Handle model weights
        if hasattr(model.model, 'save') and hasattr(model.model, 'fit'):
            model.model.save(f"{model_path}_keras.keras")
            model_data['model'] = 'keras'
        else:
            model_data['model'] = model.model

        # Save scaler for models that have it
        if hasattr(model, 'scaler'):
            model_data['scaler'] = model.scaler

        # Save processed data (except for ARIMA)
        if model.get_model_type() != 'arima':
            model_data.update({
                'X_train': model.X_train,
                'y_train': model.y_train,
                'X_val': model.X_val,
                'y_val': model.y_val,
                'X_test': model.X_test,
                'X_test_index': model.X_test_index,
            })
            if model.get_model_type() == 'sequence':
                model_data.update({
                    'categorical_vocab_sizes': model.categorical_vocab_sizes,
                    'numerical_features': model.numerical_features,
                    'categorical_features': model.categorical_features
                })
        else:
            # ARIMA only needs its ticker_data
            model_data['ticker_data'] = model.ticker_data

        joblib.dump(model_data, f"{model_path}.pkl")

    def load_model(self, model_name: str, target_type: str):
        """Load model and restore its data"""
        model_path = self.models_dir / model_name

        if not self.model_exists(model_name):
            raise FileNotFoundError(f"Model {model_name} not found")

        model_data = joblib.load(f"{model_path}.pkl")

        # Load Keras model if needed
        if model_data.get('model') == 'keras':
            loaded_model = tf.keras.models.load_model(f"{model_path}_keras.keras", compile=False)
            BaseModel.compile_keras_model(loaded_model, target_type)

            model_data['model'] = loaded_model

        return model_data

    def model_exists(self, model_name: str) -> bool:
        """Check if a saved model exists"""
        model_path = self.models_dir / model_name
        return Path(f"{model_path}.pkl").exists()
