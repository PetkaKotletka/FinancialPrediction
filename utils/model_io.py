import pickle
import joblib
from pathlib import Path
import tensorflow as tf
import json


class ModelIO:
    def __init__(self, config: dict):
        self.models_dir = Path(config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str):
        """Save model"""
        model_path = self.models_dir / model_name

        # Save model
        model_data = {}

        # Special handling for Keras models
        if hasattr(model.model, 'save') and hasattr(model.model, 'fit'):
            # It's a Keras model, save it separately
            model.model.save(f"{model_path}_keras.keras")
            model_data['model'] = 'keras'  # Placeholder to indicate Keras model
        else:
            model_data['model'] = model.model

        # Save scaler for models that have it (linear and cnn)
        if hasattr(model, 'scaler'):
            model_data['scaler'] = model.scaler

        joblib.dump(model_data, f"{model_path}.pkl")

    def load_model(self, model_name: str):
        """Load model - returns model data"""
        model_path = self.models_dir / model_name

        if self.model_exists(model_name):
            model_data = joblib.load(f"{model_path}.pkl")

            # Load Keras model if needed
            if model_data.get('model') == 'keras':
                model_data['model'] = tf.keras.models.load_model(f"{model_path}_keras.keras", compile=False)

            return model_data
        else:
            raise FileNotFoundError(f"Model {model_name} not found")

    def model_exists(self, model_name: str) -> bool:
        """Check if a saved model exists"""
        model_path = self.models_dir / model_name
        return Path(f"{model_path}.pkl").exists()
