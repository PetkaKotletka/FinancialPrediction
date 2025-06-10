from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNNModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, config: dict, target_config: dict):
        super().__init__(model_name, config, target_config)
        self.scaler = StandardScaler()
        model_config = self.models_config['cnn']
        self.window_size = model_config['window_size']
        self.batch_size = model_config['batch_size']
        self.epochs = model_config['epochs']

    def get_model_type(self):
        """CNN uses sequence data like RNN"""
        return 'sequence'

    def build_model(self):
        """Build 1D CNN architecture with embedding layers for categorical features"""
        # Separate numerical and categorical features (same as RNN)
        n_numerical = len(self.numerical_features)
        n_categorical = len(self.categorical_features)

        # Numerical input
        numerical_input = layers.Input(shape=(self.window_size, n_numerical), name='numerical_input')

        # Categorical inputs and embeddings (same as RNN)
        categorical_inputs = []
        embeddings = []

        for i, cat_feature in enumerate(self.categorical_features):
            cat_name = cat_feature.replace('_encoded', '')
            vocab_size = self.categorical_vocab_sizes[cat_name]
            embedding_dim = min(50, (vocab_size + 1) // 2)

            cat_input = layers.Input(shape=(self.window_size,), name=f'{cat_feature}_input')
            categorical_inputs.append(cat_input)

            embedding = layers.Embedding(vocab_size, embedding_dim, name=f'{cat_feature}_embedding')(cat_input)
            embeddings.append(embedding)

        # Concatenate all features
        if embeddings:
            all_features = layers.Concatenate(axis=-1)([numerical_input] + embeddings)
        else:
            all_features = numerical_input

        # CNN layers (1D convolution for time series)
        x = all_features
        x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)  # Better than Flatten for time series

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)

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

    def train(self) -> dict:
        """Train CNN with early stopping using stored data"""
        # Prepare data for multiple inputs (same as RNN)
        train_inputs = self._prepare_model_inputs(self.X_train)
        val_inputs = self._prepare_model_inputs(self.X_val)

        # Train with callbacks
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
        """Split data into numerical and categorical inputs with scaling (same as RNN)"""
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