import numpy as np
from .metrics import calculate_metrics

class ModelEvaluator:
    def evaluate_model(self, model):
        """Evaluate a single model using its stored test data"""
        # Get predictions and actual values
        y_pred = model.predict()
        y_test = model.y_test

        # Validate MultiIndex alignment
        if not y_pred.index.equals(y_test.index):
            raise ValueError(f"MultiIndex misalignment in {model.model_name}")

        print(f"  {model.model_name}: {len(y_pred)} predicted samples "
              f"({len(y_pred.index.get_level_values('ticker').unique())} tickers)")

        # Calculate metrics based on task type
        metrics = calculate_metrics(y_test, y_pred, model.target_config['type'])

        return metrics