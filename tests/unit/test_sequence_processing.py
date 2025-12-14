"""
Property-based tests for sequence processing
"""
import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from sequence_processing import SequenceProcessor


# Test Property 13: Sequence Segmentation Invariant
@given(
    seq_length=st.integers(min_value=50, max_value=500),
    target_length=st.integers(min_value=20, max_value=100),
    overlap=st.integers(min_value=0, max_value=19)
)
@settings(max_examples=100, deadline=None)
def test_property_13_sequence_segmentation_invariant(seq_length, target_length, overlap):
    """
    Property 13: Sequence Segmentation Invariant
    
    When a sequence is segmented into windows:
    1. All data points from original sequence must appear in at least one window
    2. Windows should have correct overlap
    3. Total coverage should equal or exceed original sequence length
    4. No data should be lost
    """
    # Create processor
    processor = SequenceProcessor(
        sequence_length=target_length,
        overlap=overlap
    )
    
    # Create synthetic sequence
    num_features = 10
    sequence = np.random.randn(seq_length, num_features)
    
    # Segment sequence
    windows = processor.segment_long_sequence(sequence)
    
    if seq_length <= target_length:
        # Should return single window without segmentation
        assert len(windows) == 1
        window, meta = windows[0]
        assert len(window) == seq_length
        assert meta['total_windows'] == 1
        assert meta['window_index'] == 0
    else:
        # Should return multiple windows
        assert len(windows) > 1
        
        # Check stride calculation
        stride = target_length - overlap
        expected_windows = int(np.ceil((seq_length - target_length) / stride)) + 1
        assert len(windows) == expected_windows
        
        # Check that all windows have correct metadata
        for idx, (window, meta) in enumerate(windows):
            assert meta['window_index'] == idx
            assert meta['total_windows'] == len(windows)
            assert meta['original_length'] == seq_length
            
            # Check window size (last window might be shorter)
            if idx < len(windows) - 1:
                assert len(window) == target_length
            else:
                assert len(window) <= target_length
        
        # Check coverage: verify all data points are covered
        covered_indices = set()
        for window, meta in windows:
            start = meta['window_start']
            end = meta['window_end']
            covered_indices.update(range(start, end))
        
        # All indices should be covered
        assert covered_indices == set(range(seq_length))
        
        # Check overlap between consecutive windows
        for i in range(len(windows) - 1):
            _, meta1 = windows[i]
            _, meta2 = windows[i + 1]
            
            # Calculate actual overlap
            overlap_start = meta2['window_start']
            overlap_end = meta1['window_end']
            actual_overlap = max(0, overlap_end - overlap_start)
            
            # Should match configured overlap (or less for last window)
            assert actual_overlap <= overlap + 1  # Allow small tolerance


# Test Property 14: Padding Length Preservation
@given(
    seq_length=st.integers(min_value=10, max_value=150),
    target_length=st.integers(min_value=50, max_value=100),
    num_features=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_property_14_padding_length_preservation(seq_length, target_length, num_features):
    """
    Property 14: Padding Length Preservation
    
    When a sequence is padded:
    1. Output length must equal target_length
    2. Original data must be preserved at the beginning
    3. Padding must be applied at the end
    4. Mask must correctly identify valid vs padded positions
    """
    # Create processor
    processor = SequenceProcessor(
        sequence_length=target_length,
        padding_value=0.0
    )
    
    # Create synthetic sequence
    sequence = np.random.randn(seq_length, num_features)
    original_sequence = sequence.copy()
    
    # Pad sequence
    padded, mask, meta = processor.pad_short_sequence(sequence)
    
    # Check output length
    assert len(padded) == target_length
    assert len(mask) == target_length
    
    if seq_length >= target_length:
        # No padding needed
        assert meta['padded'] is False
        assert meta['padding_length'] == 0
        assert np.all(mask)  # All True
        
        # Original data should be preserved (truncated if necessary)
        assert np.allclose(padded, original_sequence[:target_length])
    else:
        # Padding applied
        assert meta['padded'] is True
        assert meta['padding_length'] == target_length - seq_length
        
        # Check mask correctness
        assert np.sum(mask) == seq_length  # Number of True values
        assert np.all(mask[:seq_length])  # First seq_length are True
        assert not np.any(mask[seq_length:])  # Rest are False
        
        # Check original data preservation
        assert np.allclose(padded[:seq_length], original_sequence)
        
        # Check padding values
        assert np.allclose(padded[seq_length:], processor.padding_value)
    
    # Check metadata
    assert meta['original_length'] == seq_length


# Test Property 16: Masking Correctness
@given(
    num_videos=st.integers(min_value=3, max_value=10),
    frames_per_video=st.lists(
        st.integers(min_value=20, max_value=150),
        min_size=3,
        max_size=10
    ),
    target_length=st.integers(min_value=50, max_value=100)
)
@settings(max_examples=100, deadline=None)
def test_property_16_masking_correctness(num_videos, frames_per_video, target_length):
    """
    Property 16: Masking Correctness
    
    Masks must correctly identify valid data positions:
    1. Mask length equals sequence length
    2. Number of True values equals original sequence length (for padded sequences)
    3. True values are contiguous from the start
    4. False values (padding) are contiguous at the end
    5. Mask can be used to filter out padding in loss computation
    """
    # Ensure we have matching number of videos and frame counts
    frames_per_video = frames_per_video[:num_videos]
    if len(frames_per_video) < num_videos:
        frames_per_video.extend([50] * (num_videos - len(frames_per_video)))
    
    # Create processor
    processor = SequenceProcessor(
        sequence_length=target_length,
        padding_value=-999.0  # Use distinctive value
    )
    
    # Create synthetic dataframe
    data = []
    for video_idx, num_frames in enumerate(frames_per_video):
        for frame_idx in range(num_frames):
            row = {
                'video_id': f'video_{video_idx}',
                'source_video_name': f'source_{video_idx}',
                'time': frame_idx * 0.1
            }
            # Add feature columns
            for feat_idx in range(10):
                row[f'feature_{feat_idx}'] = np.random.randn()
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Process sequences
    processed = processor.process_video_sequences(df)
    
    # Check each processed sequence
    for seq_dict in processed:
        sequence = seq_dict['sequence']
        mask = seq_dict['mask']
        metadata = seq_dict['metadata']
        
        # Property 1: Mask length equals sequence length
        assert len(mask) == target_length
        assert len(sequence) == target_length
        
        # Property 2: Number of True values
        original_length = metadata['original_length']
        if metadata.get('window_start') is not None:
            # This is a window from a segmented sequence
            window_length = metadata['window_end'] - metadata['window_start']
            expected_true_count = min(window_length, target_length)
        else:
            # This is a complete sequence (possibly padded)
            expected_true_count = min(original_length, target_length)
        
        assert np.sum(mask) == expected_true_count
        
        # Property 3: True values are contiguous from start
        true_indices = np.where(mask)[0]
        if len(true_indices) > 0:
            assert true_indices[0] == 0
            assert np.all(np.diff(true_indices) == 1)  # Consecutive
        
        # Property 4: False values are contiguous at end
        false_indices = np.where(~mask)[0]
        if len(false_indices) > 0:
            assert false_indices[-1] == target_length - 1
            if len(false_indices) > 1:
                assert np.all(np.diff(false_indices) == 1)  # Consecutive
        
        # Property 5: Padding positions have padding value
        if metadata.get('padded', False):
            padding_start = metadata['original_length']
            if metadata.get('window_start') is not None:
                # Adjust for windowed sequences
                padding_start = metadata['window_end'] - metadata['window_start']
            
            # Check that padded positions have the padding value
            padded_region = sequence[padding_start:]
            assert np.allclose(padded_region, processor.padding_value)


# Additional test: Process video sequences integration
def test_process_video_sequences_integration():
    """
    Integration test for process_video_sequences
    """
    # Create synthetic dataframe with multiple videos
    data = []
    video_lengths = [30, 80, 150]  # Short, medium, long
    
    for video_idx, num_frames in enumerate(video_lengths):
        for frame_idx in range(num_frames):
            row = {
                'video_id': f'video_{video_idx}',
                'source_video_name': f'source_{video_idx}',
                'time': frame_idx * 0.1
            }
            # Add feature columns
            for feat_idx in range(5):
                row[f'feature_{feat_idx}'] = video_idx + frame_idx * 0.01
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create processor
    processor = SequenceProcessor(
        sequence_length=100,
        overlap=20
    )
    
    # Process sequences
    processed = processor.process_video_sequences(df)
    
    # Check results
    assert len(processed) > 0
    
    # Video 0 (30 frames): should be padded
    video_0_seqs = [s for s in processed if s['video_id'] == 'video_0']
    assert len(video_0_seqs) == 1
    assert video_0_seqs[0]['metadata']['padded'] is True
    
    # Video 1 (80 frames): should be padded
    video_1_seqs = [s for s in processed if s['video_id'] == 'video_1']
    assert len(video_1_seqs) == 1
    assert video_1_seqs[0]['metadata']['padded'] is True
    
    # Video 2 (150 frames): should be segmented
    video_2_seqs = [s for s in processed if s['video_id'] == 'video_2']
    assert len(video_2_seqs) > 1
    assert video_2_seqs[0]['metadata']['total_windows'] > 1
    
    # Get statistics
    stats = processor.get_statistics(processed)
    assert stats['total_sequences'] == len(processed)
    assert stats['unique_videos'] == 3
    assert stats['num_padded'] >= 2  # At least video_0 and video_1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
