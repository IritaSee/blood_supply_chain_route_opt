"""
Deep Learning predictor configuration.
1D CNN for delivery time estimation.
"""

# Data preprocessing
DATA_CONFIG = {
    'test_size': 0.2,
    'validation_split': 0.15,
    'random_seed': 42,
    'normalize': True,
    'sequence_length': 10,  # Number of past trips to consider
}

# Feature engineering
FEATURE_CONFIG = {
    'use_distance': True,
    'use_destination': True,
    'use_month': True,
    'use_day_of_week': True,
    'use_lateness_history': True,
    'destination_embedding_dim': 8,
}

# 1D CNN Architecture
MODEL_CONFIG = {
    'conv_filters': [64, 128, 64],  # Number of filters per conv layer
    'kernel_sizes': [3, 3, 3],  # Kernel size per conv layer
    'pool_sizes': [2, 2, 2],  # Max pooling sizes
    'dropout_rate': 0.3,
    'dense_units': [128, 64],  # Dense layer units
    'activation': 'relu',
    'output_activation': 'linear',  # For regression
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mse',  # Mean squared error for time prediction
    'metrics': ['mae', 'mse'],  # Mean absolute error, MSE
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6,
}

# Model saving
SAVE_CONFIG = {
    'model_dir': 'results/dl_models',
    'save_best_only': True,
    'save_weights_only': False,
    'checkpoint_monitor': 'val_loss',
    'checkpoint_mode': 'min',
}

# Prediction
PREDICTION_CONFIG = {
    'target': 'duration_minutes',  # or 'lateness_minutes'
    'confidence_interval': 0.95,
}
