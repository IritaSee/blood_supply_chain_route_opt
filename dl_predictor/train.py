"""
Training script for 1D CNN delivery time predictor.
"""

import logging
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dl_predictor.data_preprocessor import TripDataPreprocessor
from dl_predictor.cnn_model import DeliveryTimeCNN
from config.dl_config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    SAVE_CONFIG,
    PREDICTION_CONFIG
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(data_file: str = "All Droping.xlsx",
               target_col: str = None,
               save_dir: str = None):
    """
    Train 1D CNN model for delivery time prediction.
    
    Args:
        data_file: Path to Excel file with trip data
        target_col: Target column ('duration_minutes' or 'lateness_minutes')
        save_dir: Directory to save models
    """
    logger.info("=" * 80)
    logger.info("1D CNN DELIVERY TIME PREDICTOR - TRAINING")
    logger.info("=" * 80)
    
    # Use config defaults if not specified
    target_col = target_col or PREDICTION_CONFIG['target']
    save_dir = save_dir or SAVE_CONFIG['model_dir']
    
    # Step 1: Preprocess data
    logger.info("\n=== Step 1: Data Preprocessing ===")
    preprocessor = TripDataPreprocessor(file_path=data_file)
    
    data = preprocessor.prepare_data(
        target_col=target_col,
        sequence_length=DATA_CONFIG['sequence_length'],
        test_size=DATA_CONFIG['test_size'],
        random_seed=DATA_CONFIG['random_seed']
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Input shape: {X_train.shape}")
    logger.info(f"Features: {data['feature_names']}")
    
    # Step 2: Build model
    logger.info("\n=== Step 2: Building 1D CNN Model ===")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_config = {
        **MODEL_CONFIG,
        **TRAINING_CONFIG,
        **SAVE_CONFIG,
    }
    
    cnn = DeliveryTimeCNN(config=model_config)
    cnn.build_model(
        input_shape=input_shape,
        conv_filters=MODEL_CONFIG['conv_filters'],
        kernel_sizes=MODEL_CONFIG['kernel_sizes'],
        pool_sizes=MODEL_CONFIG['pool_sizes'],
        dense_units=MODEL_CONFIG['dense_units'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        activation=MODEL_CONFIG['activation']
    )
    
    cnn.summary()
    
    # Step 3: Compile model
    logger.info("\n=== Step 3: Compiling Model ===")
    cnn.compile_model(
        learning_rate=TRAINING_CONFIG['learning_rate'],
        optimizer=TRAINING_CONFIG['optimizer'],
        loss=TRAINING_CONFIG['loss'],
        metrics=TRAINING_CONFIG['metrics']
    )
    
    # Step 4: Train model
    logger.info("\n=== Step 4: Training Model ===")
    history = cnn.train(
        X_train,
        y_train,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        validation_split=DATA_CONFIG['validation_split'],
        verbose=1
    )
    
    # Step 5: Evaluate model
    logger.info("\n=== Step 5: Evaluating Model ===")
    test_metrics = cnn.evaluate(X_test, y_test)
    
    logger.info("\nTest Metrics:")
    logger.info(f"  MAE:  {test_metrics['mae']:.2f} minutes")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f} minutes")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    logger.info(f"  R²:   {test_metrics['r2_score']:.4f}")
    
    # Step 6: Save model and results
    logger.info("\n=== Step 6: Saving Model & Results ===")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = save_path / 'final_model.keras'
    cnn.save(str(model_file))
    
    # Save metrics
    results = {
        'target': target_col,
        'test_metrics': test_metrics,
        'config': {
            'data': DATA_CONFIG,
            'model': MODEL_CONFIG,
            'training': TRAINING_CONFIG,
        },
        'data_info': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'input_shape': list(input_shape),
            'features': data['feature_names'],
        }
    }
    
    results_file = save_path / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Model saved: {model_file}")
    logger.info(f"Results saved: {results_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    return cnn, test_metrics, history


if __name__ == '__main__':
    train_model()
