from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNNModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, models_config: dict, target_config: dict):
        super().__init__(model_name, models_config, target_config)
        self.scaler = StandardScaler()
        config = self.models_config['cnn']
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

    def get_model_type(self):
        """CNN needs different data format"""
        return 'cnn'

    def prepare_data(self, df: pd.DataFrame):
        """Transform data into sequences for CNN"""
        # Get feature columns and add sector for sequence creation
        feature_cols = self.feature_columns + ['ticker']
        data_with_target = df[feature_cols + [self.target_column]].copy()

        # Create sequences for each ticker
        sequences = []
        targets = []
        tickers = []

        for ticker in df['ticker'].unique():
            ticker_data = data_with_target[data_with_target['ticker'] == ticker].copy()
            ticker_data = ticker_data.drop('ticker', axis=1)
            ticker_data = ticker_data.dropna()

            if len(ticker_data) < self.window_size:
                continue

            # Create sliding windows
            for i in range(len(ticker_data) - self.window_size):
                seq = ticker_data.iloc[i:i + self.window_size][self.feature_columns].values
                target = ticker_data.iloc[i + self.window_size][self.target_column]

                sequences.append(seq)
                targets.append(target)
                tickers.append(ticker)

        # Convert to numpy arrays
        X = np.array(sequences)  # Shape: (samples, window_size, n_features)
        y = np.array(targets)

        return X, y

    def build_model(self):
        """Build CNN architecture for time series"""
        n_features = len(self.feature_columns)

        model = keras.Sequential([
            # Reshape for CNN: add channel dimension
            layers.Reshape((self.window_size, n_features, 1)),

            # First conv block
            layers.Conv2D(32, (5, 1), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 1)),

            # Second conv block
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 1)),

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
        ])

        # Output layer based on task type
        if self.target_config['type'] == 'regression':
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
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

        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
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
        return predictions.flatten()
