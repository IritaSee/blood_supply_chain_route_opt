# Deep Learning Time Predictor

1D CNN-based delivery time estimation for blood supply distribution.

## Overview

Predicts delivery duration using historical trip data via a 1D convolutional neural network trained on sequential patterns of past deliveries.

## Structure

- [data_preprocessor.py](data_preprocessor.py) — Load, clean, feature engineer trip data; create time-series sequences
- [cnn_model.py](cnn_model.py) — 1D CNN architecture (conv → batch norm → pooling → dense → output)
- [train.py](train.py) — Full training pipeline using config settings
- [../config/dl_config.py](../config/dl_config.py) — Hyperparameter configuration

## Features Used

- `distance_km`, `log_distance` — Trip distance metrics
- `month`, `day_of_week`, `day_of_month` — Temporal patterns
- `lateness_minutes` — Historical delay (as feature, not target)
- `destination_encoded` — Destination ID (label encoded)

## Model Architecture

```
Conv1D(filters) → BatchNorm → MaxPool → Dropout
    ↓
Conv1D(filters) → BatchNorm → MaxPool → Dropout
    ↓
Flatten → Dense → Dropout → Dense → Dropout → Output(1)
```

Default: `[64, 128, 64]` conv filters, `[128, 64]` dense units, 30% dropout.

## Configuration

Edit [../config/dl_config.py](../config/dl_config.py):

```python
MODEL_CONFIG = {
    'conv_filters': [64, 128, 64],
    'kernel_sizes': [3, 3, 3],
    'pool_sizes': [2, 2, 2],
    'dense_units': [128, 64],
    'dropout_rate': 0.3,
}

TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'early_stopping_patience': 15,
}

DATA_CONFIG = {
    'sequence_length': 10,  # Past trips to consider
    'test_size': 0.2,
    'validation_split': 0.15,
}
```

## Quick Start

### Demo (20 epochs, small model)
```bash
python demo_dl_predictor.py
```

### Full Training
```bash
python dl_predictor/train.py
```

Outputs:
- `results/dl_models/best_model.keras` — Best model checkpoint
- `results/dl_models/final_model.keras` — Final trained model
- `results/dl_models/training_results.json` — Metrics & config

## Evaluation Metrics

- **MAE** (Mean Absolute Error) — Average prediction error in minutes
- **RMSE** (Root Mean Squared Error) — Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error) — Relative error
- **R²** (Coefficient of Determination) — Variance explained (1.0 = perfect)

## Notes

- Small dataset (287 trips → ~225 training sequences) limits deep learning gains
- Current demo shows MAE ~432 min on test set; hyperparameter tuning and more data would improve this
- For production, consider ensemble with GA route optimization predictions
- Temporal validation split preserves time ordering (no shuffle)

## Dependencies

- TensorFlow/Keras 2.20+
- scikit-learn 1.8+
- pandas, numpy, openpyxl
