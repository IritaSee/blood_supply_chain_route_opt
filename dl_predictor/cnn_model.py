"""
1D CNN model for delivery time prediction.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

logger = logging.getLogger(__name__)


class DeliveryTimeCNN:
    """1D CNN for predicting delivery time/lateness."""
    
    def __init__(self, config: Dict):
        """
        Initialize 1D CNN model.
        
        Args:
            config: Model configuration dict
        """
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self,
                   input_shape: Tuple[int, int],
                   conv_filters: List[int] = [64, 128, 64],
                   kernel_sizes: List[int] = [3, 3, 3],
                   pool_sizes: List[int] = [2, 2, 2],
                   dense_units: List[int] = [128, 64],
                   dropout_rate: float = 0.3,
                   activation: str = 'relu') -> keras.Model:
        """
        Build 1D CNN architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
            conv_filters: Number of filters per conv layer
            kernel_sizes: Kernel sizes
            pool_sizes: Max pooling sizes
            dense_units: Dense layer units
            dropout_rate: Dropout probability
            activation: Activation function
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building 1D CNN with input shape: {input_shape}")
        
        model = models.Sequential(name='DeliveryTimeCNN')
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Conv blocks
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(conv_filters, kernel_sizes, pool_sizes)
        ):
            model.add(layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            model.add(layers.MaxPooling1D(
                pool_size=pool_size,
                name=f'maxpool_{i+1}'
            ))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Flatten
        model.add(layers.Flatten(name='flatten'))
        
        # Dense layers
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(
                units,
                activation=activation,
                name=f'dense_{i+1}'
            ))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))
        
        # Output layer (regression)
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        self.model = model
        return model
    
    def compile_model(self,
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam',
                     loss: str = 'mse',
                     metrics: List[str] = ['mae', 'mse']):
        """Compile the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer}, loss={loss}")
    
    def get_callbacks(self,
                     model_dir: str = 'results/dl_models',
                     early_stopping_patience: int = 15,
                     reduce_lr_patience: int = 7,
                     reduce_lr_factor: float = 0.5,
                     min_lr: float = 1e-6) -> List[callbacks.Callback]:
        """Create training callbacks."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(model_path / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=str(model_path / 'logs'),
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32,
             validation_split: float = 0.15,
             verbose: int = 1) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features (n_samples, seq_len, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split if X_val not provided
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info(f"Training model for {epochs} epochs, batch size {batch_size}")
        
        # Validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            val_split = 0
        else:
            validation_data = None
            val_split = validation_split
        
        # Get callbacks
        cb_list = self.get_callbacks(
            model_dir=self.config.get('model_dir', 'results/dl_models'),
            early_stopping_patience=self.config.get('early_stopping_patience', 15),
            reduce_lr_patience=self.config.get('reduce_lr_patience', 7),
            reduce_lr_factor=self.config.get('reduce_lr_factor', 0.5),
            min_lr=self.config.get('min_lr', 1e-6)
        )
        
        # Train
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=verbose
        )
        
        self.history = history
        logger.info("Training complete")
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not built.")
        
        logger.info(f"Evaluating on {len(X_test)} test samples")
        
        # Get predictions
        y_pred = self.model.predict(X_test).flatten()
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'n_samples': len(X_test),
        }
        
        logger.info(f"Test Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built.")
        
        predictions = self.model.predict(X).flatten()
        return predictions
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not built.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            logger.warning("Model not built yet.")
