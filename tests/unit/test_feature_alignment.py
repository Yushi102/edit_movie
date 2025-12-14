"""
Property-based tests for Feature Alignment using Hypothesis

**Feature: multimodal-video-features-integration**
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import pandas as pd

from feature_alignment import FeatureAligner


# Strategy for generating timestamps
@st.composite
def timestamp_arrays(draw, min_len=2, max_len=100, min_time=0.0, max_time=100.0):
    """Generate sorted arrays of timestamps"""
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    times = draw(st.lists(
        st.floats(min_value=min_time, max_value=max_time, allow_nan=False, allow_infinity=False),
        min_size=length,
        max_size=length
    ))
    return np.sort(np.array(times))


# Strategy for generating audio DataFrames
@st.composite
def audio_dataframes(draw, num_rows=None):
    """Generate valid audio feature DataFrames"""
    if num_rows is None:
        num_rows = draw(st.integers(min_value=2, max_value=50))
    
    times = draw(timestamp_arrays(min_len=num_rows, max_len=num_rows))
    
    data = {
        'time': times,
        'audio_energy_rms': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'audio_is_speaking': draw(st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=num_rows, max_size=num_rows
        )),
        'silence_duration_ms': draw(st.lists(
            st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'text_is_active': draw(st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=num_rows, max_size=num_rows
        ))
    }
    
    return pd.DataFrame(data)


# Strategy for generating visual DataFrames
@st.composite
def visual_dataframes(draw, num_rows=None):
    """Generate valid visual feature DataFrames"""
    if num_rows is None:
        num_rows = draw(st.integers(min_value=2, max_value=50))
    
    times = draw(timestamp_arrays(min_len=num_rows, max_len=num_rows))
    
    data = {
        'time': times,
        'scene_change': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'visual_motion': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'saliency_x': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'saliency_y': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'face_count': draw(st.lists(
            st.integers(min_value=0, max_value=5),
            min_size=num_rows, max_size=num_rows
        )),
        'face_center_x': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'face_center_y': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'face_size': draw(st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'face_mouth_open': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        )),
        'face_eyebrow_raise': draw(st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        ))
    }
    
    # Add CLIP features
    for i in range(512):
        data[f'clip_{i}'] = draw(st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        ))
    
    return pd.DataFrame(data)


# ============================================================================
# Property 2: Timestamp alignment tolerance
# **Feature: multimodal-video-features-integration, Property 2: Timestamp alignment tolerance**
# **Validates: Requirements 1.2**
# ============================================================================

@given(
    tolerance=st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
    time_offset=st.floats(min_value=-0.005, max_value=0.005, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_property_2_timestamp_alignment_tolerance(tolerance, time_offset):
    """
    Property 2: Timestamp alignment tolerance
    
    For any pair of timestamps from track data and video features,
    if their absolute difference is ≤ tolerance seconds, they should be considered aligned.
    
    **Validates: Requirements 1.2**
    """
    aligner = FeatureAligner(tolerance=tolerance)
    
    # Create track times
    track_times = np.array([1.0, 2.0, 3.0])
    
    # Create audio features with offset
    audio_data = {
        'time': track_times + time_offset,
        'audio_energy_rms': [0.1, 0.2, 0.3],
        'audio_is_speaking': [0, 1, 0],
        'silence_duration_ms': [100, 50, 80],
        'text_is_active': [0, 1, 0]
    }
    audio_df = pd.DataFrame(audio_data)
    
    # Align features
    aligned_audio, _, modality_mask, stats = aligner.align_features(
        track_times, audio_df, None, video_id="test"
    )
    
    # Alignment should succeed
    assert aligned_audio is not None, "Alignment should succeed"
    assert modality_mask[0, 0] == True, "Audio modality should be available"
    
    # If offset is very small (within tolerance), interpolation should be low
    if abs(time_offset) < tolerance / 2:
        assert stats['audio_interpolated_pct'] < 100.0, "Some timestamps should match within tolerance"
    
    # Alignment should always produce output of correct shape
    assert aligned_audio.shape == (len(track_times), 4), "Output shape should match"


# ============================================================================
# Property 3: Interpolation correctness by feature type
# **Feature: multimodal-video-features-integration, Property 3: Interpolation correctness by feature type**
# **Validates: Requirements 1.3**
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(st.data())
def test_property_3_interpolation_correctness_by_feature_type(data):
    """
    Property 3: Interpolation correctness by feature type
    
    For any three consecutive timestamps t1 < t2 < t3 where t2 is missing:
    - Continuous features (RMS): v2 = v1 + (v3 - v1) * (t2 - t1) / (t3 - t1) (linear)
    - Binary features (is_speaking): v2 = v1 (forward-fill)
    
    **Validates: Requirements 1.3**
    """
    # Generate simple test data
    t1 = data.draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    t3 = data.draw(st.floats(min_value=t1 + 0.2, max_value=t1 + 10.0, allow_nan=False, allow_infinity=False))
    ratio = data.draw(st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False))
    t2 = t1 + (t3 - t1) * ratio
    
    v1 = data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    v3 = data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    discrete_v1 = data.draw(st.integers(min_value=0, max_value=1))
    
    # Create audio dataframe with t1 and t3
    audio_data = {
        'time': [t1, t3],
        'audio_energy_rms': [v1, v3],
        'audio_is_speaking': [discrete_v1, 0],
        'silence_duration_ms': [100.0, 200.0],
        'text_is_active': [0, 1]
    }
    audio_df = pd.DataFrame(audio_data)
    
    aligner = FeatureAligner(tolerance=0.01)
    
    # Align with t2 in the middle
    track_times = np.array([t2])
    aligned_audio, _, _, _ = aligner.align_features(
        track_times, audio_df, None, video_id="test"
    )
    
    if aligned_audio is not None:
        # Check continuous feature (audio_energy_rms) - linear interpolation
        expected_v2 = v1 + (v3 - v1) * ratio
        actual_v2 = aligned_audio[0, 0]
        
        # Allow numerical error
        assert abs(actual_v2 - expected_v2) < 0.01, \
            f"Linear interpolation failed: expected {expected_v2}, got {actual_v2}"
        
        # Check discrete feature (audio_is_speaking) - forward-fill
        actual_v2_discrete = aligned_audio[0, 1]
        assert actual_v2_discrete == discrete_v1, \
            f"Forward-fill failed: expected {discrete_v1}, got {actual_v2_discrete}"


# ============================================================================
# Property 4: Forward-fill consistency
# **Feature: multimodal-video-features-integration, Property 4: Forward-fill consistency**
# **Validates: Requirements 1.4**
# ============================================================================

@given(
    audio_df=audio_dataframes(),
    num_targets=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_property_4_forward_fill_consistency(audio_df, num_targets):
    """
    Property 4: Forward-fill consistency
    
    For any sequence of timestamps with missing features,
    forward-filled values should equal the last known non-missing value.
    
    **Validates: Requirements 1.4**
    """
    assume(len(audio_df) >= 2)
    
    aligner = FeatureAligner(tolerance=0.01)
    
    # Create target times that span the audio range
    min_time = audio_df['time'].min()
    max_time = audio_df['time'].max()
    track_times = np.linspace(min_time, max_time, num_targets)
    
    # Align
    aligned_audio, _, _, _ = aligner.align_features(
        track_times, audio_df, None, video_id="test"
    )
    
    if aligned_audio is not None:
        # Check discrete feature (audio_is_speaking) for forward-fill consistency
        discrete_values = aligned_audio[:, 1]  # audio_is_speaking
        
        # For each target time, verify it matches the last known value
        for i, t in enumerate(track_times):
            # Find last source time <= t
            valid_sources = audio_df[audio_df['time'] <= t]
            
            if len(valid_sources) > 0:
                expected_value = valid_sources['audio_is_speaking'].iloc[-1]
                actual_value = discrete_values[i]
                
                assert actual_value == expected_value, \
                    f"Forward-fill inconsistency at t={t}: expected {expected_value}, got {actual_value}"


# ============================================================================
# Property 30: Interpolation bounds validation
# **Feature: multimodal-video-features-integration, Property 30: Interpolation bounds validation**
# **Validates: Requirements 7.5**
# ============================================================================

@given(
    audio_df=audio_dataframes(),
    num_targets=st.integers(min_value=10, max_value=30)
)
@settings(max_examples=100, deadline=None)
def test_property_30_interpolation_bounds_validation(audio_df, num_targets):
    """
    Property 30: Interpolation bounds validation
    
    For any interpolated continuous value v at timestamp t between t1 and t2,
    the value should satisfy: min(v1, v2) ≤ v ≤ max(v1, v2)
    
    **Validates: Requirements 7.5**
    """
    assume(len(audio_df) >= 2)
    
    aligner = FeatureAligner(tolerance=0.01)
    
    # Create target times within the audio range
    min_time = audio_df['time'].min()
    max_time = audio_df['time'].max()
    track_times = np.linspace(min_time, max_time, num_targets)
    
    # Align
    aligned_audio, _, _, _ = aligner.align_features(
        track_times, audio_df, None, video_id="test"
    )
    
    if aligned_audio is not None:
        # Check continuous feature (audio_energy_rms)
        rms_values = aligned_audio[:, 0]
        
        # Get source RMS values
        source_rms = audio_df['audio_energy_rms'].values
        min_source = np.min(source_rms)
        max_source = np.max(source_rms)
        
        # All interpolated values should be within bounds
        assert np.all(rms_values >= min_source - 1e-6), \
            f"Interpolated values below minimum: {np.min(rms_values)} < {min_source}"
        assert np.all(rms_values <= max_source + 1e-6), \
            f"Interpolated values above maximum: {np.max(rms_values)} > {max_source}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
