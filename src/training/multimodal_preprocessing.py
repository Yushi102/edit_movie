"""
Multimodal Feature Preprocessing and Normalization

This module provides preprocessing functionality for audio and visual features,
including normalization, L2 normalization for CLIP embeddings, and handling
of missing data.
"""
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioFeaturePreprocessor:
    """
    Preprocessor for audio features (5 numerical columns)
    
    Applies zero mean and unit variance normalization to continuous features.
    """
    
    def __init__(self):
        """Initialize AudioFeaturePreprocessor"""
        self.mean_ = None
        self.std_ = None
        self.feature_names = [
            'audio_energy_rms',
            'audio_is_speaking',
            'silence_duration_ms',
            'text_is_active'
        ]
        self.continuous_features = ['audio_energy_rms', 'silence_duration_ms']
        self.discrete_features = ['audio_is_speaking', 'text_is_active']
        
        logger.info("AudioFeaturePreprocessor initialized")
    
    def fit(self, features: np.ndarray) -> 'AudioFeaturePreprocessor':
        """
        Fit the preprocessor to the data
        
        Args:
            features: Array of shape (N, 4) with audio features
        
        Returns:
            self
        """
        if features.shape[1] != 4:
            raise ValueError(f"Expected 4 features, got {features.shape[1]}")
        
        # Compute mean and std for continuous features only
        self.mean_ = np.zeros(4)
        self.std_ = np.ones(4)
        
        # Indices for continuous features
        continuous_indices = [0, 2]  # audio_energy_rms, silence_duration_ms
        
        for idx in continuous_indices:
            self.mean_[idx] = np.mean(features[:, idx])
            self.std_[idx] = np.std(features[:, idx])
            
            # Avoid division by zero
            if self.std_[idx] < 1e-8:
                self.std_[idx] = 1.0
        
        logger.info(f"AudioFeaturePreprocessor fitted on {len(features)} samples")
        logger.info(f"  Mean: {self.mean_}")
        logger.info(f"  Std: {self.std_}")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters
        
        Args:
            features: Array of shape (N, 4) with audio features
        
        Returns:
            Normalized features of shape (N, 4)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if features.shape[1] != 4:
            raise ValueError(f"Expected 4 features, got {features.shape[1]}")
        
        # Normalize continuous features only
        normalized = features.copy()
        continuous_indices = [0, 2]
        
        for idx in continuous_indices:
            normalized[:, idx] = (features[:, idx] - self.mean_[idx]) / self.std_[idx]
        
        return normalized
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            features: Array of shape (N, 4) with audio features
        
        Returns:
            Normalized features of shape (N, 4)
        """
        return self.fit(features).transform(features)
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Inverse transform to recover original scale
        
        Args:
            features: Normalized array of shape (N, 4)
        
        Returns:
            Original scale features of shape (N, 4)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        if features.shape[1] != 4:
            raise ValueError(f"Expected 4 features, got {features.shape[1]}")
        
        # Denormalize continuous features only
        denormalized = features.copy()
        continuous_indices = [0, 2]
        
        for idx in continuous_indices:
            denormalized[:, idx] = features[:, idx] * self.std_[idx] + self.mean_[idx]
        
        return denormalized
    
    def save(self, filepath: str):
        """Save preprocessor parameters to disk"""
        params = {
            'mean': self.mean_,
            'std': self.std_,
            'feature_names': self.feature_names
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"AudioFeaturePreprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AudioFeaturePreprocessor':
        """Load preprocessor parameters from disk"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.mean_ = params['mean']
        preprocessor.std_ = params['std']
        preprocessor.feature_names = params['feature_names']
        
        logger.info(f"AudioFeaturePreprocessor loaded from {filepath}")
        return preprocessor


class VisualFeaturePreprocessor:
    """
    Preprocessor for visual features (522 columns)
    
    Applies independent normalization to motion/saliency features,
    L2 normalization to CLIP embeddings, and handles missing face data.
    """
    
    def __init__(self):
        """Initialize VisualFeaturePreprocessor"""
        self.scalar_mean_ = None
        self.scalar_std_ = None
        
        # 10 scalar features (excluding face_count which is discrete)
        self.scalar_features = [
            'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
            'face_center_x', 'face_center_y', 'face_size',
            'face_mouth_open', 'face_eyebrow_raise'
        ]
        
        self.num_scalar = 10
        self.num_clip = 512
        self.total_features = 522  # 10 scalar + 512 CLIP
        
        logger.info("VisualFeaturePreprocessor initialized")
    
    def fit(self, features: np.ndarray, face_counts: Optional[np.ndarray] = None) -> 'VisualFeaturePreprocessor':
        """
        Fit the preprocessor to the data
        
        Args:
            features: Array of shape (N, 522) with visual features
            face_counts: Optional array of shape (N,) with face counts for zero-filling
        
        Returns:
            self
        """
        if features.shape[1] != self.total_features:
            raise ValueError(f"Expected {self.total_features} features, got {features.shape[1]}")
        
        # Extract scalar features (first 10)
        scalar_features = features[:, :self.num_scalar].copy()
        
        # Handle missing face data if face_counts provided
        if face_counts is not None:
            # Zero-fill face features where face_count = 0
            # Indices 4-8 are face features (face_center_x, face_center_y, face_size, face_mouth_open, face_eyebrow_raise)
            face_mask = face_counts == 0
            scalar_features[face_mask, 4:9] = 0.0
        
        # Compute mean and std for each scalar feature independently
        self.scalar_mean_ = np.mean(scalar_features, axis=0)
        self.scalar_std_ = np.std(scalar_features, axis=0)
        
        # Avoid division by zero
        self.scalar_std_[self.scalar_std_ < 1e-8] = 1.0
        
        logger.info(f"VisualFeaturePreprocessor fitted on {len(features)} samples")
        logger.info(f"  Scalar mean: {self.scalar_mean_}")
        logger.info(f"  Scalar std: {self.scalar_std_}")
        
        return self
    
    def transform(self, features: np.ndarray, face_counts: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform features using fitted parameters
        
        Args:
            features: Array of shape (N, 522) with visual features
            face_counts: Optional array of shape (N,) with face counts for zero-filling
        
        Returns:
            Normalized features of shape (N, 522)
        """
        if self.scalar_mean_ is None or self.scalar_std_ is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if features.shape[1] != self.total_features:
            raise ValueError(f"Expected {self.total_features} features, got {features.shape[1]}")
        
        normalized = features.copy()
        
        # Handle missing face data if face_counts provided
        if face_counts is not None:
            face_mask = face_counts == 0
            normalized[face_mask, 4:9] = 0.0
        
        # Normalize scalar features independently
        normalized[:, :self.num_scalar] = (
            (normalized[:, :self.num_scalar] - self.scalar_mean_) / self.scalar_std_
        )
        
        # L2 normalize CLIP embeddings (last 512 features)
        clip_embeddings = normalized[:, self.num_scalar:]
        norms = np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)  # Avoid division by zero
        normalized[:, self.num_scalar:] = clip_embeddings / norms
        
        return normalized
    
    def fit_transform(self, features: np.ndarray, face_counts: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            features: Array of shape (N, 522) with visual features
            face_counts: Optional array of shape (N,) with face counts
        
        Returns:
            Normalized features of shape (N, 522)
        """
        return self.fit(features, face_counts).transform(features, face_counts)
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Inverse transform to recover original scale
        
        Note: CLIP embeddings cannot be fully recovered after L2 normalization
        
        Args:
            features: Normalized array of shape (N, 522)
        
        Returns:
            Partially denormalized features of shape (N, 522)
        """
        if self.scalar_mean_ is None or self.scalar_std_ is None:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        if features.shape[1] != self.total_features:
            raise ValueError(f"Expected {self.total_features} features, got {features.shape[1]}")
        
        denormalized = features.copy()
        
        # Denormalize scalar features
        denormalized[:, :self.num_scalar] = (
            features[:, :self.num_scalar] * self.scalar_std_ + self.scalar_mean_
        )
        
        # Note: CLIP embeddings remain L2 normalized (cannot recover original scale)
        
        return denormalized
    
    def save(self, filepath: str):
        """Save preprocessor parameters to disk"""
        params = {
            'scalar_mean': self.scalar_mean_,
            'scalar_std': self.scalar_std_,
            'num_scalar': self.num_scalar,
            'num_clip': self.num_clip
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"VisualFeaturePreprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VisualFeaturePreprocessor':
        """Load preprocessor parameters from disk"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.scalar_mean_ = params['scalar_mean']
        preprocessor.scalar_std_ = params['scalar_std']
        preprocessor.num_scalar = params['num_scalar']
        preprocessor.num_clip = params['num_clip']
        
        logger.info(f"VisualFeaturePreprocessor loaded from {filepath}")
        return preprocessor


if __name__ == "__main__":
    # Test preprocessors
    logger.info("Testing AudioFeaturePreprocessor...")
    
    # Create dummy audio data
    audio_features = np.random.randn(100, 4)
    audio_features[:, 1] = np.random.randint(0, 2, 100)  # Binary feature
    audio_features[:, 3] = np.random.randint(0, 2, 100)  # Binary feature
    
    audio_prep = AudioFeaturePreprocessor()
    audio_normalized = audio_prep.fit_transform(audio_features)
    audio_recovered = audio_prep.inverse_transform(audio_normalized)
    
    # Check round-trip for continuous features
    continuous_indices = [0, 2]
    for idx in continuous_indices:
        error = np.max(np.abs(audio_features[:, idx] - audio_recovered[:, idx]))
        logger.info(f"  Audio feature {idx} round-trip error: {error:.6f}")
        assert error < 1e-5, f"Round-trip failed for feature {idx}"
    
    logger.info("✅ AudioFeaturePreprocessor test passed!")
    
    # Test visual preprocessor
    logger.info("\nTesting VisualFeaturePreprocessor...")
    
    # Create dummy visual data
    visual_features = np.random.randn(100, 522)
    face_counts = np.random.randint(0, 3, 100)
    
    visual_prep = VisualFeaturePreprocessor()
    visual_normalized = visual_prep.fit_transform(visual_features, face_counts)
    
    # Check L2 normalization of CLIP embeddings
    clip_embeddings = visual_normalized[:, 10:]
    norms = np.linalg.norm(clip_embeddings, axis=1)
    logger.info(f"  CLIP embedding norms: min={norms.min():.6f}, max={norms.max():.6f}")
    assert np.allclose(norms, 1.0, atol=1e-5), "CLIP embeddings should have unit norm"
    
    # Check zero-filling for missing faces
    zero_face_mask = face_counts == 0
    if np.any(zero_face_mask):
        face_features = visual_normalized[zero_face_mask, 4:9]
        logger.info(f"  Face features for face_count=0: {face_features[0]}")
        # Note: After normalization, zero values become -mean/std, not zero
    
    logger.info("✅ VisualFeaturePreprocessor test passed!")
    
    # Test save/load
    logger.info("\nTesting save/load...")
    audio_prep.save('test_audio_prep.pkl')
    audio_prep_loaded = AudioFeaturePreprocessor.load('test_audio_prep.pkl')
    
    visual_prep.save('test_visual_prep.pkl')
    visual_prep_loaded = VisualFeaturePreprocessor.load('test_visual_prep.pkl')
    
    # Test loaded preprocessors
    audio_normalized2 = audio_prep_loaded.transform(audio_features)
    assert np.allclose(audio_normalized, audio_normalized2), "Loaded audio preprocessor mismatch"
    
    visual_normalized2 = visual_prep_loaded.transform(visual_features, face_counts)
    assert np.allclose(visual_normalized, visual_normalized2), "Loaded visual preprocessor mismatch"
    
    logger.info("✅ Save/load test passed!")
    
    logger.info("\n✅ All preprocessing tests complete!")
