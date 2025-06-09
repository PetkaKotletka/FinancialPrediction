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

    def train(self) -> dict:
        """Train the decision tree model using stored data"""
        # Fit model (no scaling needed for trees)
        self.model.fit(self.X_train, self.y_train)

        # Calculate training and validation metrics
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)

        history = {}
        if self.target_config['type'] == 'regression':
            history['train_rmse'] = np.sqrt(np.mean((self.y_train - train_pred) ** 2))
            history['val_rmse'] = np.sqrt(np.mean((self.y_val - val_pred) ** 2))
        else:
            history['train_accuracy'] = np.mean(self.y_train == train_pred)
            history['val_accuracy'] = np.mean(self.y_val == val_pred)

        return history

    def predict(self) -> np.ndarray:
        """Make predictions on stored test data"""
        return self.model.predict(self.X_test)


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
            'random_state': 1543,
            'n_jobs': -1
        }

        if self.target_config['type'] == 'regression':
            self.model = xgb.XGBRegressor(**params)
        else:
            self.model = xgb.XGBClassifier(**params)

    def train(self) -> dict:
        """Train XGBoost with early stopping using stored data"""
        # Train with early stopping using validation set
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )

        # Calculate training and validation metrics
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)

        history = {}
        if self.target_config['type'] == 'regression':
            history['train_rmse'] = np.sqrt(np.mean((self.y_train - train_pred) ** 2))
            history['val_rmse'] = np.sqrt(np.mean((self.y_val - val_pred) ** 2))
        else:
            history['train_accuracy'] = np.mean(self.y_train == train_pred)
            history['val_accuracy'] = np.mean(self.y_val == val_pred)

        return history

    def predict(self) -> np.ndarray:
        """Make predictions on stored test data"""
        return self.model.predict(self.X_test)