from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timedelta


class LSTMModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, data_config: dict, models_config: dict, target_config: dict):
        super().__init__(model_name, data_config, models_config, target_config)
        self.scaler = StandardScaler()
        config = self.models_config['lstm']
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.hidden_units = config['hidden_units']
        self.n_layers = config['n_layers']
        self.dropout_rate = config['dropout_rate']

    def get_model_type(self):
        """RNN uses sequence data"""
        return 'sequence'

    def build_model(self):
        """Build LSTM architecture"""
        n_features = len(self.feature_columns)

        model = keras.Sequential()

        # Input layer
        model.add(layers.InputLayer(shape=(self.window_size, n_features)))

        # LSTM layers
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)  # Only last layer returns single output
            model.add(layers.LSTM(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            ))

        # Dense layer
        model.add(layers.Dense(16, activation='relu'))

        # Output layer based on task type
        if self.target_config['type'] == 'regression':
            model.add(layers.Dense(1, kernel_initializer='normal'))
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
            loss=loss,
            metrics=metrics
        )

        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Train LSTM with early stopping"""
        # Scale features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], n_timesteps, n_features)

        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Learning rate reduction
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )

        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, lr_reducer],
            verbose=0
        )

        # Get final metrics
        train_results = self.model.evaluate(X_train_scaled, y_train, verbose=0)
        val_results = self.model.evaluate(X_val_scaled, y_val, verbose=0)

        if self.target_config['type'] == 'regression':
            return {
                'train_rmse': np.sqrt(train_results[0]),
                'val_rmse': np.sqrt(val_results[0])
            }
        else:
            return {
                'train_accuracy': train_results[1],
                'val_accuracy': val_results[1]
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with scaling"""
        # print(f"LSTM predict - Input shape: {X.shape}")
        # print(f"Scaler mean: {self.scaler.mean_[:5]}...")  # First 5 features
        # print(f"Scaler scale: {self.scaler.scale_[:5]}...")

        # Scale features
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

        predictions = self.model.predict(X_scaled, verbose=0)

        # Convert probabilities to binary predictions for classification
        if self.target_config['type'] == 'classification':
            predictions = (predictions > 0.5).astype(int)

        # print(f"Predictions - min: {predictions.min():.6f}, max: {predictions.max():.6f}")

        return predictions.flatten()


class GRUModel(LSTMModel):
    """GRU Model - inherits from LSTM with only architecture difference"""
    is_implemented = True

    def __init__(self, model_name: str, data_config: dict, models_config: dict, target_config: dict):
        super().__init__(model_name, data_config, models_config, target_config)
        config = self.models_config['gru']
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.hidden_units = config['hidden_units']
        self.n_layers = config['n_layers']
        self.dropout_rate = config['dropout_rate']

    def build_model(self):
        """Build GRU architecture - same as LSTM but with GRU layers"""
        n_features = len(self.feature_columns)

        model = keras.Sequential()

        # Input layer
        model.add(layers.InputLayer(shape=(self.window_size, n_features)))

        # GRU layers
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)
            model.add(layers.GRU(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            ))

        # Dense layer
        model.add(layers.Dense(16, activation='relu'))

        # Output layer based on task type
        if self.target_config['type'] == 'regression':
            model.add(layers.Dense(1, kernel_initializer='normal'))
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
            loss=loss,
            metrics=metrics
        )

        self.model = model