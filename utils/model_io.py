import pickle
import joblib
from pathlib import Path
import json


class ModelIO:
    def __init__(self, config: dict):
        self.models_dir = Path(config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str):
        """Save model"""
        model_path = self.models_dir / model_name

        # Save model
        model_data = {
            'model': model.model
        }

        # Save scaler for LinearModel
        if hasattr(model, 'scaler'):
            model_data['scaler'] = model.scaler

        joblib.dump(model_data, f"{model_path}.pkl")

    def load_model(self, model_name: str):
        """Load model - returns model data"""
        model_path = self.models_dir / model_name

        if self.model_exists(model_name):
            model_data = joblib.load(f"{model_path}.pkl")
            return model_data
        else:
            raise FileNotFoundError(f"Model {model_name} not found")

    def model_exists(self, model_name: str) -> bool:
        """Check if a saved model exists"""
        model_path = self.models_dir / model_name
        return Path(f"{model_path}.pkl").exists()
