from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNNModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, data_config: dict, models_config: dict, target_config: dict):
        super().__init__(model_name, data_config, models_config, target_config)
        self.scaler = StandardScaler()
        config = self.models_config['cnn']
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

    def get_model_type(self):
        """CNN needs different data format"""
        return 'cnn'

    def build_model(self):
        """Build CNN architecture for time series"""
        n_features = len(self.feature_columns)

        model = keras.Sequential([
            # Use Conv1D for time series
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),

            # Single convolutional block
            layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),  # Better than Flatten for time series

            # Single dense layer
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
        ])

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
        """Train CNN with early stopping"""
        # Scale features (3D data)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], n_timesteps, n_features)

        # Early stopping callback
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
        # Scale features (3D data)
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

        predictions = self.model.predict(X_scaled, verbose=0)

        # Convert probabilities to binary predictions for classification
        if self.target_config['type'] == 'classification':
            predictions = (predictions > 0.5).astype(int)

        return predictions.flatten()
