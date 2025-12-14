"""
Data preprocessing and normalization for Multi-Track Transformer training
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from typing import Tuple, Dict, Optional
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, validation, normalization, and train/val split"""
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            normalization_method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        """
        self.normalization_method = normalization_method
        self.scalers = {}
        self.feature_stats = {}
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file with validation"""
        logger.info(f"Loading CSV from {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['video_id', 'source_video_name', 'time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for track columns
        track_cols = [col for col in df.columns if col.startswith('target_v')]
        logger.info(f"Found {len(track_cols)} track parameter columns")
        
        return df

    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        logger.info("Validating data...")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values, filling with 0")
            df = df.fillna(0)
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
        if inf_mask.sum() > 0:
            logger.warning(f"Found {inf_mask.sum()} rows with infinite values, replacing with 0")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Validate data types
        for col in df.columns:
            if col.startswith('target_v') and 'active' in col:
                # Active columns should be 0 or 1
                df[col] = df[col].astype(int)
                if not df[col].isin([0, 1]).all():
                    logger.warning(f"Column {col} contains values other than 0/1")
            elif col.startswith('target_v') and 'asset' in col:
                # Asset columns should be integers
                df[col] = df[col].astype(int)
        
        logger.info("Data validation complete")
        return df

    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: Input dataframe
            fit: If True, fit scalers on data. If False, use existing scalers
        
        Returns:
            Normalized dataframe
        """
        logger.info(f"Normalizing features using {self.normalization_method} method")
        
        df_normalized = df.copy()
        
        # Define parameter groups for normalization
        param_groups = {
            'scale': {'cols': [], 'range': (0, 500)},
            'position': {'cols': [], 'range': (-1000, 1000)},
            'crop': {'cols': [], 'range': (0, 100)}
        }
        
        # Collect columns for each parameter group
        for col in df.columns:
            if 'scale' in col and col.startswith('target_v'):
                param_groups['scale']['cols'].append(col)
            elif ('_x' in col or '_y' in col) and col.startswith('target_v'):
                param_groups['position']['cols'].append(col)
            elif 'crop' in col and col.startswith('target_v'):
                param_groups['crop']['cols'].append(col)
        
        # Normalize each parameter group
        for group_name, group_info in param_groups.items():
            cols = group_info['cols']
            if not cols:
                continue
            
            if fit:
                if self.normalization_method == 'standard':
                    scaler = StandardScaler()
                else:  # minmax
                    scaler = MinMaxScaler(feature_range=(0, 1))
                
                df_normalized[cols] = scaler.fit_transform(df[cols])
                self.scalers[group_name] = scaler
                
                # Store statistics
                self.feature_stats[group_name] = {
                    'mean': df[cols].mean().to_dict(),
                    'std': df[cols].std().to_dict(),
                    'min': df[cols].min().to_dict(),
                    'max': df[cols].max().to_dict()
                }
            else:
                if group_name not in self.scalers:
                    raise ValueError(f"Scaler for {group_name} not fitted yet")
                df_normalized[cols] = self.scalers[group_name].transform(df[cols])
        
        logger.info(f"Normalized {sum(len(g['cols']) for g in param_groups.values())} columns")
        return df_normalized

    
    def train_val_split(
        self, 
        df: pd.DataFrame, 
        val_ratio: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets by video_id
        
        Args:
            df: Input dataframe
            val_ratio: Ratio of validation data
            random_state: Random seed for reproducibility
        
        Returns:
            (train_df, val_df)
        """
        logger.info(f"Splitting data: {1-val_ratio:.0%} train, {val_ratio:.0%} validation")
        
        # Get unique video IDs
        unique_videos = df['video_id'].unique()
        logger.info(f"Total unique videos: {len(unique_videos)}")
        
        # Split video IDs
        train_videos, val_videos = train_test_split(
            unique_videos, 
            test_size=val_ratio, 
            random_state=random_state
        )
        
        # Split dataframe
        train_df = df[df['video_id'].isin(train_videos)].copy()
        val_df = df[df['video_id'].isin(val_videos)].copy()
        
        logger.info(f"Train: {len(train_df)} rows ({len(train_videos)} videos)")
        logger.info(f"Val: {len(val_df)} rows ({len(val_videos)} videos)")
        
        return train_df, val_df
    
    def save_scalers(self, output_path: str):
        """Save fitted scalers to file"""
        with open(output_path, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'feature_stats': self.feature_stats,
                'normalization_method': self.normalization_method
            }, f)
        logger.info(f"Scalers saved to {output_path}")
    
    def load_scalers(self, input_path: str):
        """Load fitted scalers from file"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.feature_stats = data['feature_stats']
            self.normalization_method = data['normalization_method']
        logger.info(f"Scalers loaded from {input_path}")



def preprocess_pipeline(
    csv_path: str,
    output_dir: str = 'preprocessed_data',
    val_ratio: float = 0.2,
    normalization_method: str = 'standard',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline
    
    Args:
        csv_path: Path to input CSV
        output_dir: Directory to save preprocessed data
        val_ratio: Validation split ratio
        normalization_method: 'standard' or 'minmax'
        random_state: Random seed
    
    Returns:
        (train_df, val_df)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(normalization_method=normalization_method)
    
    # Load and validate data
    df = preprocessor.load_csv(csv_path)
    df = preprocessor.validate_data(df)
    
    # Split data
    train_df, val_df = preprocessor.train_val_split(df, val_ratio, random_state)
    
    # Normalize features (fit on train, transform both)
    train_df = preprocessor.normalize_features(train_df, fit=True)
    val_df = preprocessor.normalize_features(val_df, fit=False)
    
    # Save preprocessed data
    train_path = os.path.join(output_dir, 'train_data.csv')
    val_path = os.path.join(output_dir, 'val_data.csv')
    scaler_path = os.path.join(output_dir, 'scalers.pkl')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    preprocessor.save_scalers(scaler_path)
    
    logger.info(f"Preprocessed data saved to {output_dir}")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Val: {val_path}")
    logger.info(f"  Scalers: {scaler_path}")
    
    return train_df, val_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="preprocessed_data", help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--normalization", choices=['standard', 'minmax'], default='standard')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    preprocess_pipeline(
        args.csv_path,
        args.output_dir,
        args.val_ratio,
        args.normalization,
        args.seed
    )
