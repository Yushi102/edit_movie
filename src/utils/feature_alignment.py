"""
Feature Alignment and Interpolation for Multimodal Video Features

This module provides functionality to align and interpolate video features
(audio and visual) with editing track data based on timestamps.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict
from scipy import interpolate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureAligner:
    """
    Aligns and interpolates multimodal features to match track timestamps.
    
    Implements type-aware interpolation:
    - Linear interpolation for continuous features (RMS, motion, saliency)
    - Forward-fill for binary/discrete features (is_speaking, text_active, face_count)
    - CLIP embedding interpolation with L2 renormalization
    """
    
    def __init__(self, tolerance: float = 0.05):
        """
        Initialize FeatureAligner
        
        Args:
            tolerance: Timestamp matching tolerance in seconds (default: 0.05)
        """
        self.tolerance = tolerance
        
        # Define feature types for different interpolation strategies
        self.continuous_features = [
            'audio_energy_rms', 'silence_duration_ms',
            'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
            'face_center_x', 'face_center_y', 'face_size',
            'face_mouth_open', 'face_eyebrow_raise'
        ]
        
        # Text embeddings are also continuous (will be matched by prefix)
        self.speech_embedding_prefix = 'speech_emb_'
        self.telop_embedding_prefix = 'telop_emb_'
        
        self.discrete_features = [
            'audio_is_speaking', 'text_is_active', 'telop_active', 'face_count'
        ]
        
        self.clip_features = [f'clip_{i}' for i in range(512)]
        
        logger.info(f"FeatureAligner initialized with tolerance={tolerance}s")
    
    def align_features(
        self,
        track_times: np.ndarray,
        audio_df: Optional[pd.DataFrame] = None,
        visual_df: Optional[pd.DataFrame] = None,
        video_id: str = "unknown"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Dict[str, float]]:
        """
        Align audio and visual features to track timestamps
        
        Args:
            track_times: Array of track timestamps (seq_len,)
            audio_df: DataFrame with audio features (columns: time, audio_energy_rms, etc.)
            visual_df: DataFrame with visual features (columns: time, scene_change, etc.)
            video_id: Video identifier for logging
        
        Returns:
            Tuple of:
            - aligned_audio: Array of shape (seq_len, audio_features) or None
            - aligned_visual: Array of shape (seq_len, visual_features) or None
            - modality_mask: Boolean array of shape (seq_len, 3) for [audio, visual, track]
            - stats: Dict with alignment statistics
        """
        seq_len = len(track_times)
        stats = {
            'audio_interpolated_pct': 0.0,
            'visual_interpolated_pct': 0.0,
            'audio_coverage_pct': 0.0,
            'visual_coverage_pct': 0.0,
            'max_gap_audio': 0.0,
            'max_gap_visual': 0.0
        }
        
        # Initialize modality mask (all True for track, depends on availability for audio/visual)
        modality_mask = np.ones((seq_len, 3), dtype=bool)
        modality_mask[:, 2] = True  # Track is always available
        
        # Align audio features
        aligned_audio = None
        if audio_df is not None and len(audio_df) > 0:
            try:
                aligned_audio, audio_stats = self._align_audio_features(
                    track_times, audio_df, video_id
                )
                modality_mask[:, 0] = True
                stats.update({
                    'audio_interpolated_pct': audio_stats['interpolated_pct'],
                    'audio_coverage_pct': audio_stats['coverage_pct'],
                    'max_gap_audio': audio_stats['max_gap']
                })
            except Exception as e:
                logger.error(f"Failed to align audio features for {video_id}: {e}")
                modality_mask[:, 0] = False
        else:
            modality_mask[:, 0] = False
            logger.warning(f"No audio features available for {video_id}")
        
        # Align visual features
        aligned_visual = None
        if visual_df is not None and len(visual_df) > 0:
            try:
                aligned_visual, visual_stats = self._align_visual_features(
                    track_times, visual_df, video_id
                )
                modality_mask[:, 1] = True
                stats.update({
                    'visual_interpolated_pct': visual_stats['interpolated_pct'],
                    'visual_coverage_pct': visual_stats['coverage_pct'],
                    'max_gap_visual': visual_stats['max_gap']
                })
            except Exception as e:
                logger.error(f"Failed to align visual features for {video_id}: {e}")
                modality_mask[:, 1] = False
        else:
            modality_mask[:, 1] = False
            logger.warning(f"No visual features available for {video_id}")
        
        # Log warnings for quality issues
        if stats['audio_interpolated_pct'] > 50.0:
            logger.warning(f"{video_id}: High audio interpolation rate: {stats['audio_interpolated_pct']:.1f}%")
        
        if stats['visual_interpolated_pct'] > 50.0:
            logger.warning(f"{video_id}: High visual interpolation rate: {stats['visual_interpolated_pct']:.1f}%")
        
        if stats['max_gap_audio'] > 5.0:
            logger.warning(f"{video_id}: Large audio gap detected: {stats['max_gap_audio']:.2f}s")
        
        if stats['max_gap_visual'] > 5.0:
            logger.warning(f"{video_id}: Large visual gap detected: {stats['max_gap_visual']:.2f}s")
        
        return aligned_audio, aligned_visual, modality_mask, stats

    
    def _align_audio_features(
        self,
        track_times: np.ndarray,
        audio_df: pd.DataFrame,
        video_id: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Align audio features to track timestamps
        
        Args:
            track_times: Target timestamps
            audio_df: Audio features DataFrame
            video_id: Video identifier
        
        Returns:
            Tuple of (aligned_features, stats)
        """
        # Extract numerical audio features (including text embeddings if present)
        audio_features = ['audio_energy_rms', 'audio_is_speaking', 
                         'silence_duration_ms', 'text_is_active']
        
        # Add telop_active if it exists
        if 'telop_active' in audio_df.columns:
            audio_features.append('telop_active')
        
        # Add speech text embedding columns if they exist
        speech_emb_cols = [col for col in audio_df.columns if col.startswith('speech_emb_')]
        if speech_emb_cols:
            audio_features.extend(sorted(speech_emb_cols))  # Sort to ensure consistent order
        
        # Add telop text embedding columns if they exist
        telop_emb_cols = [col for col in audio_df.columns if col.startswith('telop_emb_')]
        if telop_emb_cols:
            audio_features.extend(sorted(telop_emb_cols))  # Sort to ensure consistent order
        
        # Verify all required features exist
        missing_features = [f for f in audio_features if f not in audio_df.columns]
        if missing_features:
            raise ValueError(f"Missing audio features: {missing_features}")
        
        audio_times = audio_df['time'].values
        seq_len = len(track_times)
        num_features = len(audio_features)
        
        # Initialize output array
        aligned = np.zeros((seq_len, num_features), dtype=np.float32)
        
        # Track interpolation statistics
        interpolated_count = 0
        matched_count = 0
        
        # Check for monotonic ordering
        if not np.all(np.diff(audio_times) >= 0):
            logger.warning(f"{video_id}: Audio timestamps are not monotonically increasing")
            # Sort by time
            audio_df = audio_df.sort_values('time').reset_index(drop=True)
            audio_times = audio_df['time'].values
        
        # Compute gaps
        gaps = np.diff(audio_times)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0.0
        
        # Align each feature
        interp_mask = None
        for feat_idx, feat_name in enumerate(audio_features):
            feat_values = audio_df[feat_name].values
            
            # Determine interpolation method based on feature type
            if feat_name in self.discrete_features:
                # Forward-fill for discrete features
                aligned[:, feat_idx] = self._forward_fill_interpolate(
                    track_times, audio_times, feat_values
                )
            elif feat_name.startswith(self.speech_embedding_prefix) or feat_name.startswith(self.telop_embedding_prefix):
                # Linear interpolation for text embeddings
                aligned[:, feat_idx], interp_mask = self._linear_interpolate(
                    track_times, audio_times, feat_values
                )
            else:
                # Linear interpolation for continuous features
                aligned[:, feat_idx], interp_mask = self._linear_interpolate(
                    track_times, audio_times, feat_values
                )
        
        # Count interpolated values (use mask from last continuous feature)
        if interp_mask is not None:
            interpolated_count = np.sum(interp_mask)
        
        # Compute statistics
        interpolated_pct = (interpolated_count / seq_len * 100) if seq_len > 0 else 0.0
        
        stats = {
            'interpolated_pct': interpolated_pct,
            'coverage_pct': 100.0 - interpolated_pct,  # Approximate coverage
            'max_gap': max_gap
        }
        
        return aligned, stats
    
    def _align_visual_features(
        self,
        track_times: np.ndarray,
        visual_df: pd.DataFrame,
        video_id: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Align visual features to track timestamps
        
        Args:
            track_times: Target timestamps
            visual_df: Visual features DataFrame
            video_id: Video identifier
        
        Returns:
            Tuple of (aligned_features, stats)
        """
        # Define visual features (11 scalar + 512 CLIP)
        scalar_features = [
            'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
            'face_count', 'face_center_x', 'face_center_y', 'face_size',
            'face_mouth_open', 'face_eyebrow_raise'
        ]
        
        # Verify features exist
        missing_features = [f for f in scalar_features if f not in visual_df.columns]
        if missing_features:
            raise ValueError(f"Missing visual features: {missing_features}")
        
        # Check for CLIP features
        clip_cols = [f'clip_{i}' for i in range(512)]
        missing_clip = [c for c in clip_cols if c not in visual_df.columns]
        if missing_clip:
            raise ValueError(f"Missing CLIP features: {len(missing_clip)} features")
        
        visual_times = visual_df['time'].values
        seq_len = len(track_times)
        num_features = len(scalar_features) + 512  # 11 + 512 = 523... wait, design says 522
        
        # Actually, let's check the design: 522 = 11 scalar + 512 CLIP - 1
        # Looking at the CSV, saliency_x and saliency_y might be combined, or one feature is missing
        # Let's use 11 + 512 = 523 for now, but we'll verify
        
        # Initialize output array (522 features as per design)
        # Design says: 11 scalar features + 512 CLIP - 1 = 522
        # Let's check: scene_change, visual_motion, saliency_x, saliency_y (4)
        # face_count, face_center_x, face_center_y, face_size (4)
        # face_mouth_open, face_eyebrow_raise (2)
        # Total scalar: 10 features + 512 CLIP = 522
        
        # Correction: remove one feature to match 522
        scalar_features = [
            'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
            'face_count', 'face_center_x', 'face_center_y', 'face_size',
            'face_mouth_open', 'face_eyebrow_raise'
        ]  # This is 10 features
        
        num_features = 522  # 10 scalar + 512 CLIP
        aligned = np.zeros((seq_len, num_features), dtype=np.float32)
        
        # Track interpolation statistics
        interpolated_count = 0
        
        # Check for monotonic ordering
        if not np.all(np.diff(visual_times) >= 0):
            logger.warning(f"{video_id}: Visual timestamps are not monotonically increasing")
            visual_df = visual_df.sort_values('time').reset_index(drop=True)
            visual_times = visual_df['time'].values
        
        # Compute gaps
        gaps = np.diff(visual_times)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0.0
        
        # Align scalar features
        interp_mask = None
        for feat_idx, feat_name in enumerate(scalar_features):
            feat_values = visual_df[feat_name].values
            
            # Handle missing face data (zero-fill when face_count = 0)
            if feat_name.startswith('face_') and feat_name != 'face_count':
                # Zero-fill where face_count is 0
                face_count = visual_df['face_count'].values
                feat_values = np.where(face_count == 0, 0.0, feat_values)
            
            if feat_name in self.discrete_features:
                # Forward-fill for discrete features
                aligned[:, feat_idx] = self._forward_fill_interpolate(
                    track_times, visual_times, feat_values
                )
            else:
                # Linear interpolation for continuous features
                aligned[:, feat_idx], interp_mask = self._linear_interpolate(
                    track_times, visual_times, feat_values
                )
        
        # Count interpolated values (use mask from last continuous feature)
        if interp_mask is not None:
            interpolated_count = np.sum(interp_mask)
        
        # Align CLIP embeddings with L2 renormalization
        clip_start_idx = len(scalar_features)
        for i in range(512):
            clip_col = f'clip_{i}'
            clip_values = visual_df[clip_col].values
            
            # Linear interpolation
            aligned[:, clip_start_idx + i], _ = self._linear_interpolate(
                track_times, visual_times, clip_values
            )
        
        # L2 renormalize CLIP embeddings
        clip_embeddings = aligned[:, clip_start_idx:]
        norms = np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        aligned[:, clip_start_idx:] = clip_embeddings / norms
        
        # Compute statistics
        interpolated_pct = (interpolated_count / seq_len * 100) if seq_len > 0 else 0.0
        
        stats = {
            'interpolated_pct': interpolated_pct,
            'coverage_pct': 100.0 - interpolated_pct,
            'max_gap': max_gap
        }
        
        return aligned, stats
    
    def _linear_interpolate(
        self,
        target_times: np.ndarray,
        source_times: np.ndarray,
        source_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform linear interpolation
        
        Args:
            target_times: Target timestamps
            source_times: Source timestamps
            source_values: Source values
        
        Returns:
            Tuple of (interpolated_values, interpolation_mask)
            interpolation_mask is True where interpolation was used
        """
        # Use numpy interp for linear interpolation
        interpolated = np.interp(target_times, source_times, source_values)
        
        # Create mask for interpolated values (not exact matches)
        interpolation_mask = np.zeros(len(target_times), dtype=bool)
        
        # Mark interpolated values (not within tolerance of any source time)
        for i, t in enumerate(target_times):
            # Check if this target time is close to any source time
            min_distance = np.min(np.abs(source_times - t))
            if min_distance > self.tolerance:
                # This is an interpolated value
                interpolation_mask[i] = True
        
        return interpolated, interpolation_mask
    
    def _forward_fill_interpolate(
        self,
        target_times: np.ndarray,
        source_times: np.ndarray,
        source_values: np.ndarray
    ) -> np.ndarray:
        """
        Perform forward-fill interpolation for discrete features
        
        Args:
            target_times: Target timestamps
            source_times: Source timestamps
            source_values: Source values
        
        Returns:
            Forward-filled values
        """
        result = np.zeros(len(target_times), dtype=source_values.dtype)
        
        for i, t in enumerate(target_times):
            # Find the last source time <= target time
            valid_indices = np.where(source_times <= t)[0]
            
            if len(valid_indices) > 0:
                # Use the last known value
                last_idx = valid_indices[-1]
                result[i] = source_values[last_idx]
            else:
                # No previous value, use first value
                result[i] = source_values[0] if len(source_values) > 0 else 0
        
        return result


if __name__ == "__main__":
    # Test FeatureAligner
    logger.info("Testing FeatureAligner...")
    
    # Create dummy data
    track_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Audio features
    audio_data = {
        'time': [0.0, 0.15, 0.35, 0.5],
        'audio_energy_rms': [0.1, 0.2, 0.3, 0.4],
        'audio_is_speaking': [0, 1, 1, 0],
        'silence_duration_ms': [100, 50, 30, 80],
        'text_is_active': [0, 1, 1, 0]
    }
    audio_df = pd.DataFrame(audio_data)
    
    # Visual features (simplified)
    visual_data = {
        'time': [0.0, 0.2, 0.4],
        'scene_change': [0.0, 0.5, 0.1],
        'visual_motion': [0.1, 0.3, 0.2],
        'saliency_x': [0.5, 0.6, 0.5],
        'saliency_y': [0.5, 0.4, 0.5],
        'face_count': [1, 0, 1],
        'face_center_x': [0.5, 0.0, 0.6],
        'face_center_y': [0.5, 0.0, 0.5],
        'face_size': [0.1, 0.0, 0.12],
        'face_mouth_open': [0.3, 0.0, 0.4],
        'face_eyebrow_raise': [0.2, 0.0, 0.3]
    }
    
    # Add CLIP features
    for i in range(512):
        visual_data[f'clip_{i}'] = np.random.randn(3)
    
    visual_df = pd.DataFrame(visual_data)
    
    # Test alignment
    aligner = FeatureAligner(tolerance=0.05)
    aligned_audio, aligned_visual, modality_mask, stats = aligner.align_features(
        track_times, audio_df, visual_df, video_id="test_video"
    )
    
    logger.info(f"\nAlignment Results:")
    logger.info(f"  Track times shape: {track_times.shape}")
    logger.info(f"  Aligned audio shape: {aligned_audio.shape if aligned_audio is not None else None}")
    logger.info(f"  Aligned visual shape: {aligned_visual.shape if aligned_visual is not None else None}")
    logger.info(f"  Modality mask shape: {modality_mask.shape}")
    logger.info(f"  Modality mask: {modality_mask}")
    logger.info(f"\nStatistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Verify CLIP L2 normalization
    if aligned_visual is not None:
        clip_embeddings = aligned_visual[:, 10:]  # Last 512 features
        norms = np.linalg.norm(clip_embeddings, axis=1)
        logger.info(f"\nCLIP embedding norms: {norms}")
        logger.info(f"  All norms ≈ 1.0: {np.allclose(norms, 1.0, atol=1e-5)}")
    
    logger.info("\n✅ FeatureAligner test complete!")
