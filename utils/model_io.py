import pickle
import joblib
from pathlib import Path
import json


class ModelIO:
    def __init__(self, config: dict):
        self.models_dir = Path(config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str):
        """Save model and its metadata"""
        model_path = self.models_dir / model_name

        # Save metadata
        metadata = {
            'model_type': model.__class__.__name__,
            'target_config': model.target_config,
            'feature_columns': model.feature_columns,
            'target_column': model.target_column
        }

        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f)

        # Save sklearn model and scaler for LinearModel
        if hasattr(model, 'model') and hasattr(model, 'scaler'):
            model_data = {
                'model': model.model,
                'scaler': model.scaler
            }
            joblib.dump(model_data, f"{model_path}.pkl")
        else:
            # TODO: Handle other model types
            pass

    def load_model(self, model_name: str):
        """Load model - returns model data and metadata"""
        model_path = self.models_dir / model_name

        # Load metadata
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)

        # Load model data
        if Path(f"{model_path}.pkl").exists():
            model_data = joblib.load(f"{model_path}.pkl")
            return model_data, metadata
        else:
            raise FileNotFoundError(f"Model {model_name} not found")

    def model_exists(self, model_name: str) -> bool:
        """Check if a saved model exists"""
        model_path = self.models_dir / model_name
        return Path(f"{model_path}.pkl").exists() and Path(f"{model_path}_metadata.json").exists()
