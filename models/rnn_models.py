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

    def __init__(self, model_name: str, config: dict, target_config: dict):
        super().__init__(model_name, config, target_config)
        self.scaler = StandardScaler()
        model_config = self._get_model_config()
        self.window_size = model_config['window_size']
        self.batch_size = model_config['batch_size']
        self.epochs = model_config['epochs']
        self.hidden_units = model_config['hidden_units']
        self.n_layers = model_config['n_layers']
        self.dropout_rate = model_config['dropout_rate']

    def _get_model_config(self):
        return self.models_config['lstm']

    def get_model_type(self):
        """RNN uses sequence data"""
        return 'sequence'

    def build_model(self):
        """Build LSTM architecture with embedding layers for categorical features"""
        # Separate numerical and categorical features
        n_numerical = len(self.numerical_features)
        n_categorical = len(self.categorical_features)

        # Numerical input
        numerical_input = layers.Input(shape=(self.window_size, n_numerical), name='numerical_input')

        # Categorical inputs and embeddings
        categorical_inputs = []
        embeddings = []

        for i, cat_feature in enumerate(self.categorical_features):
            # Get vocabulary size for this categorical feature
            cat_name = cat_feature.replace('_encoded', '')  # Remove _encoded suffix
            vocab_size = self.categorical_vocab_sizes[cat_name]
            embedding_dim = min(50, (vocab_size + 1) // 2)  # Rule of thumb for embedding size

            # Categorical input for this feature
            cat_input = layers.Input(shape=(self.window_size,), name=f'{cat_feature}_input')
            categorical_inputs.append(cat_input)

            # Embedding layer
            embedding = layers.Embedding(vocab_size, embedding_dim, name=f'{cat_feature}_embedding')(cat_input)
            embeddings.append(embedding)

        # Concatenate all features
        if embeddings:
            # Combine numerical and embedded categorical features
            all_features = layers.Concatenate(axis=-1)([numerical_input] + embeddings)
        else:
            # Only numerical features
            all_features = numerical_input

        # LSTM layers
        x = all_features
        x = self._build_rnn_layers(x)

        # Dense layer
        x = layers.Dense(16, activation='relu')(x)

        # Output layer
        if self.target_config['type'] == 'regression':
            output = layers.Dense(1, kernel_initializer='normal')(x)
        else:
            output = layers.Dense(1, activation='sigmoid')(x)

        # Create model with multiple inputs
        inputs = [numerical_input] + categorical_inputs
        self.model = keras.Model(inputs=inputs, outputs=output)

        # Compile using base class method
        self.compile_keras_model(self.model, self.target_config['type'])

    def _build_rnn_layers(self, x):
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)
            x = layers.LSTM(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            )(x)
        return x

    def train(self) -> dict:
        """Train LSTM with early stopping using stored data"""
        # Prepare data for multiple inputs
        train_inputs = self._prepare_model_inputs(self.X_train)
        val_inputs = self._prepare_model_inputs(self.X_val)

        # Train with multiple inputs
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
        )

        self.model.fit(
            train_inputs, self.y_train,
            validation_data=(val_inputs, self.y_val),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=[early_stop, lr_reducer], verbose=0
        )

    def _prepare_model_inputs(self, X):
        """Split data into numerical and categorical inputs with scaling"""
        n_numerical = len(self.numerical_features)
        n_categorical = len(self.categorical_features)

        # Split based on known order: numerical first, then categorical
        X_numerical = X[:, :, :n_numerical]

        # Scale numerical features
        n_samples, n_timesteps, n_features = X_numerical.shape
        X_num_reshaped = X_numerical.reshape(-1, n_features)
        X_num_scaled = self.scaler.fit_transform(X_num_reshaped) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(X_num_reshaped)
        X_num_scaled = X_num_scaled.reshape(n_samples, n_timesteps, n_features)

        # Extract categorical features (each gets its own input)
        inputs = [X_num_scaled]
        for i in range(n_categorical):
            cat_idx = n_numerical + i
            inputs.append(X[:, :, cat_idx])

        return inputs

    def predict(self) -> pd.Series:
        """Make predictions on stored test data"""
        test_inputs = self._prepare_model_inputs(self.X_test)
        predictions = self.model.predict(test_inputs, verbose=0)

        if self.target_config['type'] == 'classification':
            predictions = (predictions > 0.5).astype(int)

        return pd.Series(predictions.flatten(), index=self.X_test_index, name='predictions').sort_index()


class GRUModel(LSTMModel):
    """GRU Model - inherits from LSTM with only architecture difference"""
    is_implemented = True

    def _get_model_config(self):
        return self.models_config['gru']

    def _build_rnn_layers(self, x):
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)
            x = layers.GRU(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            )(x)
        return x