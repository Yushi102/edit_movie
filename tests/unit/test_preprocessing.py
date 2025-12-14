"""
Property-based tests for data preprocessing
"""
import pytest
from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
import tempfile
import os
from data_preprocessing import DataPreprocessor, preprocess_pipeline


# ==========================================
# Property 10: Feature Normalization Bounds
# **Feature: multi-track-training-pipeline, Property 10: Feature Normalization Bounds**
# ==========================================

@given(
    num_rows=st.integers(min_value=100, max_value=1000),
    scale_values=st.lists(
        st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000
    )
)
@settings(max_examples=100, deadline=None)
def test_feature_normalization_bounds_standard(num_rows, scale_values):
    """
    For any numerical feature, after standardization (z-score),
    values should approximately fall within [-3, 3] for most data (99.7%)
    """
    # Adjust list length to match num_rows
    scale_values = scale_values[:num_rows]
    if len(scale_values) < num_rows:
        scale_values.extend([100.0] * (num_rows - len(scale_values)))
    
    # Create test dataframe
    df = pd.DataFrame({
        'video_id': ['test'] * num_rows,
        'source_video_name': ['test'] * num_rows,
        'time': np.arange(num_rows) * 0.1,
        'target_v1_active': [1] * num_rows,
        'target_v1_asset': [0] * num_rows,
        'target_v1_scale': scale_values,
        'target_v1_x': [0.0] * num_rows,
        'target_v1_y': [0.0] * num_rows,
        'target_v1_crop_l': [0.0] * num_rows
    })
    
    # Normalize
    preprocessor = DataPreprocessor(normalization_method='standard')
    df_normalized = preprocessor.normalize_features(df, fit=True)
    
    # Check bounds (99.7% should be within [-3, 3] for normal distribution)
    normalized_values = df_normalized['target_v1_scale'].values
    within_bounds = np.abs(normalized_values) <= 3.0
    percentage_within = within_bounds.sum() / len(normalized_values)
    
    # At least 95% should be within 3 standard deviations
    assert percentage_within >= 0.95, \
        f"Only {percentage_within:.1%} of values within [-3, 3], expected >= 95%"


@given(
    num_rows=st.integers(min_value=100, max_value=1000),
    scale_values=st.lists(
        st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000
    )
)
@settings(max_examples=100, deadline=None)
def test_feature_normalization_bounds_minmax(num_rows, scale_values):
    """
    For any numerical feature with variance > 0, after min-max normalization,
    all values should fall within [0, 1]
    """
    # Adjust list length
    scale_values = scale_values[:num_rows]
    if len(scale_values) < num_rows:
        scale_values.extend([100.0] * (num_rows - len(scale_values)))
    
    # Skip if all values are the same (no variance)
    if len(set(scale_values)) == 1:
        return
    
    # Create test dataframe
    df = pd.DataFrame({
        'video_id': ['test'] * num_rows,
        'source_video_name': ['test'] * num_rows,
        'time': np.arange(num_rows) * 0.1,
        'target_v1_active': [1] * num_rows,
        'target_v1_asset': [0] * num_rows,
        'target_v1_scale': scale_values,
        'target_v1_x': [0.0] * num_rows,
        'target_v1_y': [0.0] * num_rows,
        'target_v1_crop_l': [0.0] * num_rows
    })
    
    # Normalize
    preprocessor = DataPreprocessor(normalization_method='minmax')
    df_normalized = preprocessor.normalize_features(df, fit=True)
    
    # Check bounds [0, 1]
    normalized_values = df_normalized['target_v1_scale'].values
    
    assert np.all(normalized_values >= 0.0), "Some values below 0"
    assert np.all(normalized_values <= 1.0), "Some values above 1"
    
    # Check that min and max are actually 0 and 1 (or close)
    assert normalized_values.min() < 0.1, "Min should be close to 0"
    assert normalized_values.max() > 0.9, "Max should be close to 1"


@given(
    train_size=st.integers(min_value=50, max_value=200),
    val_size=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=50, deadline=None)
def test_train_val_split_no_leakage(train_size, val_size):
    """
    For any train/val split by video_id,
    there should be no video_id overlap between train and validation sets
    """
    total_size = train_size + val_size
    num_videos = min(20, total_size // 10)  # At least 10 rows per video
    
    # Create test data with multiple videos
    video_ids = []
    for i in range(num_videos):
        video_ids.extend([f'video_{i}'] * (total_size // num_videos))
    
    # Pad to exact size
    while len(video_ids) < total_size:
        video_ids.append(f'video_{num_videos - 1}')
    video_ids = video_ids[:total_size]
    
    df = pd.DataFrame({
        'video_id': video_ids,
        'source_video_name': ['test'] * total_size,
        'time': np.arange(total_size) * 0.1,
        'target_v1_active': [1] * total_size,
        'target_v1_asset': [0] * total_size,
        'target_v1_scale': [100.0] * total_size
    })
    
    # Split
    preprocessor = DataPreprocessor()
    val_ratio = val_size / total_size
    train_df, val_df = preprocessor.train_val_split(df, val_ratio=val_ratio)
    
    # Check no overlap
    train_videos = set(train_df['video_id'].unique())
    val_videos = set(val_df['video_id'].unique())
    
    overlap = train_videos & val_videos
    assert len(overlap) == 0, f"Found {len(overlap)} videos in both train and val sets"
    
    # Check all videos are accounted for
    all_videos = set(df['video_id'].unique())
    split_videos = train_videos | val_videos
    assert split_videos == all_videos, "Some videos were lost in split"


@given(
    num_rows=st.integers(min_value=100, max_value=500)
)
@settings(max_examples=50, deadline=None)
def test_data_validation_handles_missing_values(num_rows):
    """
    For any dataframe with missing values,
    validation should fill them with 0
    """
    # Create dataframe with some NaN values
    df = pd.DataFrame({
        'video_id': ['test'] * num_rows,
        'source_video_name': ['test'] * num_rows,
        'time': np.arange(num_rows) * 0.1,
        'target_v1_active': [1] * num_rows,
        'target_v1_scale': [100.0] * num_rows
    })
    
    # Introduce some NaN values
    nan_indices = np.random.choice(num_rows, size=min(10, num_rows // 10), replace=False)
    df.loc[nan_indices, 'target_v1_scale'] = np.nan
    
    # Validate
    preprocessor = DataPreprocessor()
    df_validated = preprocessor.validate_data(df)
    
    # Check no NaN values remain
    assert df_validated.isnull().sum().sum() == 0, "NaN values still present after validation"
    
    # Check that NaN positions were filled with 0
    assert all(df_validated.loc[nan_indices, 'target_v1_scale'] == 0), \
        "NaN values not filled with 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
