from .metrics import calculate_metrics


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, X_test, y_test, task_type='regression'):
        """Basic evaluation implementation"""
        # TODO: Comprehensive evaluation including:
        # - Performance by market regime
        # - Performance by sector
        # - Rolling window analysis
        y_pred = model.predict(X_test)
        return calculate_metrics(y_test, y_pred, task_type)
