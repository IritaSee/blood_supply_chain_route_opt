"""
Data preprocessing for deep learning time predictor.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class TripDataPreprocessor:
    """Preprocess historical trip data for 1D CNN training."""
    
    def __init__(self, file_path: str = "All Droping.xlsx"):
        """Initialize preprocessor."""
        self.file_path = Path(file_path)
        self.scaler = StandardScaler()
        self.destination_encoder = LabelEncoder()
        self.feature_names = []
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and clean trip data."""
        logger.info(f"Loading trip data from {self.file_path}")
        
        df = pd.read_excel(
            self.file_path,
            sheet_name="Keterlambatan & Waktu Trip",
            header=1
        )
        
        # Filter valid trips
        df = df[df['Tanggal Pengiriman'].notna()].copy()
        df = df[df['Jarak (km)'].notna()].copy()
        
        # Clean numeric columns
        df['Jarak (km)'] = pd.to_numeric(df['Jarak (km)'], errors='coerce')
        df['convert Durasi (Menit)'] = pd.to_numeric(
            df['convert Durasi (Menit)'],
            errors='coerce'
        )
        df['Convert Waktu Terlambat (Menit)'] = pd.to_numeric(
            df['Convert Waktu Terlambat (Menit)'],
            errors='coerce'
        ).fillna(0)
        
        # Parse dates
        df['Tanggal Pengiriman'] = pd.to_datetime(
            df['Tanggal Pengiriman'],
            errors='coerce'
        )
        
        # Remove invalid rows
        df = df.dropna(subset=['Jarak (km)', 'convert Durasi (Menit)', 'Tanggal Pengiriman'])
        df = df[df['Jarak (km)'] > 0].copy()
        
        logger.info(f"Loaded {len(df)} valid trips")
        self.data = df
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         use_distance: bool = True,
                         use_destination: bool = True,
                         use_temporal: bool = True) -> pd.DataFrame:
        """Create features for model training."""
        features = df.copy()
        
        # Temporal features
        if use_temporal:
            features['month'] = features['Tanggal Pengiriman'].dt.month
            features['day_of_week'] = features['Tanggal Pengiriman'].dt.dayofweek
            features['day_of_month'] = features['Tanggal Pengiriman'].dt.day
        
        # Distance-based features
        if use_distance:
            features['distance_km'] = features['Jarak (km)']
            features['log_distance'] = np.log1p(features['Jarak (km)'])
        
        # Destination encoding
        if use_destination and 'Tujuan 1' in features.columns:
            features['destination'] = features['Tujuan 1'].fillna('Unknown')
        
        # Target variables
        features['duration_minutes'] = features['convert Durasi (Menit)']
        features['lateness_minutes'] = features['Convert Waktu Terlambat (Menit)']
        features['is_late'] = (features['lateness_minutes'] > 0).astype(int)
        
        # Sort by date for sequence creation
        features = features.sort_values('Tanggal Pengiriman').reset_index(drop=True)
        
        return features
    
    def prepare_sequences(self, df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'duration_minutes',
                         sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for 1D CNN.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name
            sequence_length: Number of past trips to include
        
        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,)
        """
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Extract features and target
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        
        X_sequences = []
        y_sequences = []
        
        # Create sliding windows
        for i in range(sequence_length, len(df)):
            X_sequences.append(X_data[i-sequence_length:i])
            y_sequences.append(y_data[i])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def prepare_data(self,
                    target_col: str = 'duration_minutes',
                    sequence_length: int = 10,
                    test_size: float = 0.2,
                    random_seed: int = 42) -> Dict:
        """
        Full preprocessing pipeline.
        
        Returns:
            Dict with train/test splits and metadata
        """
        logger.info("Starting data preparation pipeline")
        
        # Load data
        if self.data is None:
            self.load_data()
        
        # Engineer features
        df = self.engineer_features(
            self.data,
            use_distance=True,
            use_destination=True,
            use_temporal=True
        )
        
        # Encode categorical features
        if 'destination' in df.columns:
            df['destination_encoded'] = self.destination_encoder.fit_transform(
                df['destination']
            )
        
        # Select numeric features
        feature_cols = [
            'distance_km',
            'log_distance',
            'month',
            'day_of_week',
            'day_of_month',
            'lateness_minutes',  # Historical lateness as feature
        ]
        
        if 'destination_encoded' in df.columns:
            feature_cols.append('destination_encoded')
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        self.feature_names = feature_cols
        
        logger.info(f"Using features: {feature_cols}")
        
        # Normalize features
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = self.prepare_sequences(
            df,
            feature_cols,
            target_col,
            sequence_length
        )
        
        # Train/test split (temporal aware - don't shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'destination_encoder': self.destination_encoder,
            'feature_names': self.feature_names,
            'n_features': len(feature_cols),
            'sequence_length': sequence_length,
        }
