"""
Demo: DL predictor integration (standalone).
Trains DL time predictor and shows metrics.
"""

import logging
from pathlib import Path

from dl_predictor.data_preprocessor import TripDataPreprocessor
from dl_predictor.cnn_model import DeliveryTimeCNN
from config.dl_config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train DL predictor and show results."""
    logger.info("\n" + "="*80)
    logger.info("DL TIME PREDICTOR - STANDALONE DEMO")
    logger.info("="*80 + "\n")
    
    # Prepare data
    logger.info("[1] Loading and preprocessing trip data...")
    preprocessor = TripDataPreprocessor(file_path="All Droping.xlsx")
    data = preprocessor.prepare_data(
        target_col='duration_minutes',
        sequence_length=DATA_CONFIG.get('sequence_length', 5),
        test_size=DATA_CONFIG.get('test_size', 0.2),
        random_seed=DATA_CONFIG.get('random_seed', 42)
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Input shape: {X_train.shape}")
    
    # Build model
    logger.info("\n[2] Building 1D CNN model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    output_dir = Path("results/dl_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        **MODEL_CONFIG,
        **TRAINING_CONFIG,
        'model_dir': str(output_dir),
    }
    
    cnn = DeliveryTimeCNN(config=config)
    cnn.build_model(
        input_shape=input_shape,
        conv_filters=MODEL_CONFIG.get('conv_filters', [32, 64]),
        kernel_sizes=MODEL_CONFIG.get('kernel_sizes', [3, 3]),
        pool_sizes=MODEL_CONFIG.get('pool_sizes', [1, 1]),
        dense_units=MODEL_CONFIG.get('dense_units', [64, 32]),
        dropout_rate=MODEL_CONFIG.get('dropout_rate', 0.3),
    )
    
    # Compile
    logger.info("[3] Compiling model...")
    cnn.compile_model(
        learning_rate=TRAINING_CONFIG.get('learning_rate', 0.001),
        optimizer=TRAINING_CONFIG.get('optimizer', 'adam'),
        loss=TRAINING_CONFIG.get('loss', 'mse'),
        metrics=TRAINING_CONFIG.get('metrics', ['mae', 'mse'])
    )
    
    # Train
    logger.info("\n[4] Training (20 epochs for quick demo)...")
    cnn.train(
        X_train,
        y_train,
        epochs=20,
        batch_size=TRAINING_CONFIG.get('batch_size', 32),
        validation_split=DATA_CONFIG.get('validation_split', 0.15),
        verbose=1
    )
    
    # Evaluate
    logger.info("\n[5] Evaluating on test set...")
    metrics = cnn.evaluate(X_test, y_test)

    # Export history and metrics to CSV for plotting
    history_csv = output_dir / "training_history.csv"
    metrics_csv = output_dir / "test_metrics.csv"
    cnn.export_history_to_csv(str(history_csv))
    cnn.export_metrics_to_csv(metrics, str(metrics_csv))
    
    logger.info("\n" + "="*80)
    logger.info("DL MODEL EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Test MAE:  {metrics['mae']:.2f} minutes")
    logger.info(f"Test RMSE: {metrics['rmse']:.2f} minutes")
    logger.info(f"Test R²:   {metrics['r2_score']:.4f}")
    logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
    logger.info("="*80)
    
    # Save model
    model_file = output_dir / 'integrated_model.keras'
    cnn.save(str(model_file))
    logger.info(f"\nModel saved to: {model_file}")
    logger.info(f"Training history CSV: {history_csv}")
    logger.info(f"Test metrics CSV: {metrics_csv}")
    
    logger.info("\n✓ DL integration demo complete!")


if __name__ == '__main__':
    main()
