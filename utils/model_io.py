import pickle
import joblib
from pathlib import Path
import json


class ModelIO:
    def __init__(self, config: dict):
        self.models_dir = Path(config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str, metadata: dict = None):
        """Save model and its metadata"""
        model_path = self.models_dir / f"{model_name}.pkl"

        # Save model
        if hasattr(model, 'save'):  # Keras models
            model.save(model_path.with_suffix('.h5'))
        else:  # Sklearn models
            joblib.dump(model, model_path)

        # Save metadata
        if metadata:
            meta_path = self.models_dir / f"{model_name}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def load_model(self, model_name: str):
        """Load model and its metadata"""
        model_path = self.models_dir / f"{model_name}.pkl"

        if model_path.with_suffix('.h5').exists():
            # TODO: Load Keras model
            pass
        elif model_path.exists():
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model {model_name} not found")
