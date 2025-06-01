from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb


class DecisionTreeModel(BaseModel):
    is_implemented = True

    def build_model(self):
        """Initialize sklearn decision tree based on task type"""
        config = self.models_config['decision_tree']
        if self.target_config['type'] == 'regression':
            self.model = DecisionTreeRegressor(
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                random_state=1543
            )
        else:  # classification
            self.model = DecisionTreeClassifier(
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                random_state=1543
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Train the decision tree model"""
        # Fit model (no scaling needed for trees)
        self.model.fit(X_train, y_train)

        # Calculate training and validation metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        history = {}
        if self.target_config['type'] == 'regression':
            history['train_rmse'] = np.sqrt(np.mean((y_train - train_pred) ** 2))
            history['val_rmse'] = np.sqrt(np.mean((y_val - val_pred) ** 2))
        else:
            history['train_accuracy'] = np.mean(y_train == train_pred)
            history['val_accuracy'] = np.mean(y_val == val_pred)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (no scaling needed)"""
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    is_implemented = True

    def build_model(self):
        """Initialize XGBoost model based on task type"""
        config = self.models_config['xgboost']
        params = {
            'n_estimators': config['n_estimators'],
            'max_depth': config['max_depth'],
            'learning_rate': config['learning_rate'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'early_stopping_rounds': config['early_stopping_rounds'],
            'random_state': 1543,
            'n_jobs': -1
        }

        model_class = xgb.XGBRegressor if self.target_config['type'] == 'regression' else xgb.XGBClassifier
        self.model = model_class(**params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Train XGBoost with early stopping"""

        # Train with early stopping using validation set
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Calculate training and validation metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        history = {}
        if self.target_config['type'] == 'regression':
            history['train_rmse'] = np.sqrt(np.mean((y_train - train_pred) ** 2))
            history['val_rmse'] = np.sqrt(np.mean((y_val - val_pred) ** 2))
        else:
            history['train_accuracy'] = np.mean(y_train == train_pred)
            history['val_accuracy'] = np.mean(y_val == val_pred)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
