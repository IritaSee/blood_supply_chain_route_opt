"""
Demo: 1D CNN Delivery Time Predictor.
Quick training run with reduced epochs for demonstration.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dl_predictor.data_preprocessor import TripDataPreprocessor
from dl_predictor.cnn_model import DeliveryTimeCNN
from config import dl_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run quick demo of 1D CNN predictor."""
    logger.info("\n" + "="*80)
    logger.info("1D CNN DELIVERY TIME PREDICTOR - DEMO")
    logger.info("="*80 + "\n")
    
    # Step 1: Load and preprocess data
    logger.info("=== Step 1: Data Preprocessing ===\n")
    preprocessor = TripDataPreprocessor(file_path="All Droping.xlsx")
    
    data = preprocessor.prepare_data(
        target_col='duration_minutes',
        sequence_length=5,  # Shorter sequences for demo
        test_size=0.2,
        random_seed=42
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Input shape: {X_train.shape}")
    logger.info(f"Features: {data['feature_names']}\n")
    
    # Step 2: Build model
    logger.info("=== Step 2: Building 1D CNN ===\n")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Simplified config for demo
    config = {
        'model_dir': 'results/dl_models_demo',
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-6,
    }
    
    cnn = DeliveryTimeCNN(config=config)
    cnn.build_model(
        input_shape=input_shape,
        conv_filters=[32, 64],  # Reduced layers for small sequences
        kernel_sizes=[3, 3],
        pool_sizes=[1, 1],  # No pooling to preserve dimensions
        dense_units=[64, 32],
        dropout_rate=0.3,
        activation='relu'
    )
    
    logger.info("Model architecture:")
    cnn.summary()
    print()
    
    # Step 3: Compile
    logger.info("=== Step 3: Compiling Model ===\n")
    cnn.compile_model(
        learning_rate=0.001,
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Step 4: Train (reduced epochs for demo)
    logger.info("=== Step 4: Training (Quick Demo - 20 epochs) ===\n")
    history = cnn.train(
        X_train,
        y_train,
        epochs=20,  # Reduced for demo
        batch_size=16,
        validation_split=0.15,
        verbose=2  # Less verbose output
    )
    
    # Step 5: Evaluate
    logger.info("\n=== Step 5: Evaluation ===\n")
    metrics = cnn.evaluate(X_test, y_test)
    
    logger.info("Test Performance:")
    logger.info(f"  Mean Absolute Error (MAE):  {metrics['mae']:.2f} minutes")
    logger.info(f"  Root Mean Squared Error:    {metrics['rmse']:.2f} minutes")
    logger.info(f"  Mean Absolute % Error:      {metrics['mape']:.2f}%")
    logger.info(f"  R² Score:                   {metrics['r2_score']:.4f}")
    
    # Step 6: Sample predictions
    logger.info("\n=== Step 6: Sample Predictions ===\n")
    sample_idx = [0, 10, 20, 30, 40]
    predictions = cnn.predict(X_test[sample_idx])
    
    logger.info("Sample predictions vs actual:")
    for i, idx in enumerate(sample_idx):
        if idx < len(y_test):
            logger.info(f"  Sample {idx}: Predicted={predictions[i]:.1f} min, "
                       f"Actual={y_test[idx]:.1f} min, "
                       f"Error={abs(predictions[i] - y_test[idx]):.1f} min")
    
    # Save model
    logger.info("\n=== Step 7: Saving Model ===\n")
    cnn.save('results/dl_models_demo/demo_model.keras')
    logger.info("Model saved to results/dl_models_demo/demo_model.keras")
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)
    
    return cnn, metrics


if __name__ == '__main__':
    main()
