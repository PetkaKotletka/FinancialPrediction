import numpy as np
from .metrics import calculate_metrics

class ModelEvaluator:
    def evaluate_model(self, model):
        """Evaluate a single model using its stored test data"""
        # Get predictions and actual values
        y_pred = model.predict()
        y_test = model.y_test

        # Print sample count for verification
        print(f"  {model.model_name}: {len(y_pred)} predicted samples")

        # Calculate metrics based on task type
        metrics = calculate_metrics(y_test, y_pred, model.target_config['type'])

        return metrics