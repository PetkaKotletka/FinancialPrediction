from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler


class LinearModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, config: dict, target_config: dict):
        super().__init__(model_name, config, target_config)
        self.scaler = StandardScaler()

    def build_model(self):
        """Will build while training to adjust hyperparameters"""
        pass

    def train(self):
        """Train the linear model using stored data with hyperparameter tuning"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)

        # Hyperparameter tuning using validation set
        if self.target_config['type'] == 'regression':
            # Try different alpha values for Ridge regression
            alphas = np.logspace(-3, 3, 10) # [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
            best_score = float('inf')
            best_alpha = alphas[0]

            for alpha in alphas:
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled, self.y_train)
                val_pred = model.predict(X_val_scaled)
                val_rmse = np.sqrt(np.mean((self.y_val - val_pred) ** 2))

                if val_rmse < best_score:
                    best_score = val_rmse
                    best_alpha = alpha

            # Train final model with best hyperparameter
            self.model = Ridge(alpha=best_alpha)

        else:  # classification
            # Try different C values for Logistic regression
            C_values = np.logspace(-3, 3, 10) # [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
            best_score = 0.0
            best_C = C_values[0]

            for C in C_values:
                model = LogisticRegression(C=C, max_iter=1000, random_state=1543)
                model.fit(X_train_scaled, self.y_train)
                val_pred = model.predict(X_val_scaled)
                val_accuracy = np.mean(self.y_val == val_pred)

                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_C = C

            # Train final model with best hyperparameter
            self.model = LogisticRegression(C=best_C, max_iter=1000, random_state=1543)

        # Fit final model
        self.model.fit(X_train_scaled, self.y_train)

    def predict(self, return_probabilities=False) -> pd.Series:
        """Make predictions on stored test data with scaling"""
        X_test_scaled = self.scaler.transform(self.X_test)

        if self.target_config['type'] == 'classification' and return_probabilities:
            # Return probabilities for positive class
            predictions = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            predictions = self.model.predict(X_test_scaled)
            if self.target_config['type'] == 'classification' and not return_probabilities:
                predictions = (predictions > 0.5).astype(int)

        return pd.Series(predictions, index=self.X_test_index, name='predictions').sort_index()
