"""
Property-based tests for MultimodalDataset

Tests Properties 1, 5, 14, 15, 31 from the design document.
"""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, HealthCheck

from multimodal_dataset import MultimodalDataset, collate_fn, create_multimodal_dataloaders
from feature_alignment import FeatureAligner
from multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_sequences_npz(temp_dir):
    """Create a sample sequences.npz file"""
    num_videos = 5
    seq_len = 50
    
    sequences = np.random.randn(num_videos, seq_len, 180).astype(np.float32)
    masks = np.ones((num_videos, seq_len), dtype=bool)
    video_ids = np.array([f"test_video_{i}" for i in range(num_videos)])
    
    npz_path = Path(temp_dir) / "test_sequences.npz"
    np.savez(npz_path, sequences=sequences, masks=masks, video_ids=video_ids)
    
    return str(npz_path)


@pytest.fixture
def sample_features_dir(temp_dir):
    """Create sample feature CSV files"""
    features_dir = Path(temp_dir) / "features"
    features_dir.mkdir()
    
    # Create features for 3 out of 5 videos
    for i in range(3):
        video_id = f"test_video_{i}"
        
        # Audio features
        audio_data = {
            'time': np.arange(0, 5.0, 0.1),
            'audio_energy_rms': np.random.rand(50),
            'audio_is_speaking': np.random.randint(0, 2, 50),
            'silence_duration_ms': np.random.rand(50) * 100,
            'text_is_active': np.random.randint(0, 2, 50),
            'speaker_id': ['speaker_1'] * 50,
            'text_word': ['word'] * 50
        }
        audio_df = pd.DataFrame(audio_data)
        audio_df.to_csv(features_dir / f"{video_id}_features.csv", index=False)
        
        # Visual features
        visual_data = {
            'time': np.arange(0, 5.0, 0.1),
            'scene_change': np.random.rand(50),
            'visual_motion': np.random.rand(50),
            'saliency_x': np.random.rand(50),
            'saliency_y': np.random.rand(50),
            'face_count': np.random.randint(0, 3, 50),
            'face_center_x': np.random.rand(50),
            'face_center_y': np.random.rand(50),
            'face_size': np.random.rand(50),
            'face_mouth_open': np.random.rand(50),
            'face_eyebrow_raise': np.random.rand(50)
        }
        
        # Add CLIP features
        for j in range(512):
            visual_data[f'clip_{j}'] = np.random.randn(50)
        
        visual_df = pd.DataFrame(visual_data)
        visual_df.to_csv(features_dir / f"{video_id}_visual_features.csv", index=False)
    
    return str(features_dir)


# Property 1: Feature file loading completeness
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    num_videos=st.integers(min_value=1, max_value=10),
    seq_len=st.integers(min_value=10, max_value=100),
    num_with_features=st.integers(min_value=0, max_value=10)
)
def test_property_1_feature_loading_completeness(num_videos, seq_len, num_with_features):
    """
    Property 1: Feature file loading completeness
    
    For any video with existing feature CSV files, loading should successfully
    return both audio and visual feature dataframes with non-zero rows.
    """
    # Ensure num_with_features doesn't exceed num_videos
    num_with_features = min(num_with_features, num_videos)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sequences
        sequences = np.random.randn(num_videos, seq_len, 180).astype(np.float32)
        masks = np.ones((num_videos, seq_len), dtype=bool)
        video_ids = np.array([f"video_{i}" for i in range(num_videos)])
        
        npz_path = Path(temp_dir) / "sequences.npz"
        np.savez(npz_path, sequences=sequences, masks=masks, video_ids=video_ids)
        
        # Create features directory
        features_dir = Path(temp_dir) / "features"
        features_dir.mkdir(exist_ok=True)
        
        # Create features for some videos
        for i in range(num_with_features):
            video_id = f"video_{i}"
            
            # Audio features
            num_frames = seq_len
            audio_data = {
                'time': np.arange(num_frames) * 0.1,
                'audio_energy_rms': np.random.rand(num_frames),
                'audio_is_speaking': np.random.randint(0, 2, num_frames),
                'silence_duration_ms': np.random.rand(num_frames) * 100,
                'text_is_active': np.random.randint(0, 2, num_frames)
            }
            audio_df = pd.DataFrame(audio_data)
            audio_df.to_csv(features_dir / f"{video_id}_features.csv", index=False)
            
            # Visual features
            visual_data = {
                'time': np.arange(num_frames) * 0.1,
                'scene_change': np.random.rand(num_frames),
                'visual_motion': np.random.rand(num_frames),
                'saliency_x': np.random.rand(num_frames),
                'saliency_y': np.random.rand(num_frames),
                'face_count': np.random.randint(0, 3, num_frames),
                'face_center_x': np.random.rand(num_frames),
                'face_center_y': np.random.rand(num_frames),
                'face_size': np.random.rand(num_frames),
                'face_mouth_open': np.random.rand(num_frames),
                'face_eyebrow_raise': np.random.rand(num_frames)
            }
            
            for j in range(512):
                visual_data[f'clip_{j}'] = np.random.randn(num_frames)
            
            visual_df = pd.DataFrame(visual_data)
            visual_df.to_csv(features_dir / f"{video_id}_visual_features.csv", index=False)
        
        # Create dataset
        dataset = MultimodalDataset(
            sequences_npz=str(npz_path),
            features_dir=str(features_dir),
            enable_multimodal=True
        )
        
        # Verify: Videos with features should load successfully
        for i in range(num_with_features):
            video_id = f"video_{i}"
            audio_df = dataset._load_audio_features(video_id)
            visual_df = dataset._load_visual_features(video_id)
            
            # Both should be loaded successfully
            assert audio_df is not None, f"Audio features should be loaded for {video_id}"
            assert visual_df is not None, f"Visual features should be loaded for {video_id}"
            assert len(audio_df) > 0, f"Audio features should have non-zero rows for {video_id}"
            assert len(visual_df) > 0, f"Visual features should have non-zero rows for {video_id}"
        
        # Verify: Videos without features should return None
        for i in range(num_with_features, num_videos):
            video_id = f"video_{i}"
            audio_df = dataset._load_audio_features(video_id)
            visual_df = dataset._load_visual_features(video_id)
            
            assert audio_df is None, f"Audio features should be None for {video_id}"
            assert visual_df is None, f"Visual features should be None for {video_id}"
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


# Property 5: Modality concatenation structure with masking
def test_property_5_modality_concatenation_structure(sample_sequences_npz, sample_features_dir):
    """
    Property 5: Modality concatenation structure with masking
    
    For any aligned audio (A), visual (V), and track (T) features with modality_mask (M),
    the output should include:
    - Concatenated features with shape (seq_len, dim_A + dim_V + dim_T) preserving order [A, V, T]
    - Modality mask with shape (seq_len, 3) indicating availability of each modality
    """
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=sample_features_dir,
        enable_multimodal=True
    )
    
    # Get a sample
    sample = dataset[0]
    
    # Verify shapes
    audio = sample['audio']
    visual = sample['visual']
    track = sample['track']
    modality_mask = sample['modality_mask']
    
    seq_len = track.shape[0]
    
    # Check audio shape
    assert audio.shape == (seq_len, 4), f"Audio shape should be (seq_len, 4), got {audio.shape}"
    
    # Check visual shape
    assert visual.shape == (seq_len, 522), f"Visual shape should be (seq_len, 522), got {visual.shape}"
    
    # Check track shape
    assert track.shape == (seq_len, 180), f"Track shape should be (seq_len, 180), got {track.shape}"
    
    # Check modality mask shape
    assert modality_mask.shape == (seq_len, 3), f"Modality mask shape should be (seq_len, 3), got {modality_mask.shape}"
    
    # Check modality mask values (should be boolean)
    assert modality_mask.dtype == torch.bool, f"Modality mask should be boolean, got {modality_mask.dtype}"
    
    # Track should always be available
    assert torch.all(modality_mask[:, 2]), "Track modality should always be available"
    
    # If audio is available, audio tensor should not be all zeros
    if torch.any(modality_mask[:, 0]):
        assert not torch.all(audio == 0), "Audio should not be all zeros when available"
    
    # If visual is available, visual tensor should not be all zeros
    if torch.any(modality_mask[:, 1]):
        assert not torch.all(visual == 0), "Visual should not be all zeros when available"


# Property 14: Video name matching
def test_property_14_video_name_matching(sample_sequences_npz, sample_features_dir):
    """
    Property 14: Video name matching
    
    For any video_id in the training data, the system should search for feature files
    matching the pattern "{video_id}_features.csv" and "{video_id}_visual_features.csv"
    in the input_features directory.
    """
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=sample_features_dir,
        enable_multimodal=True
    )
    
    # Test video name matching
    for idx in range(len(dataset)):
        video_id = dataset._get_video_id(idx)
        
        # Expected file paths
        expected_audio_path = Path(sample_features_dir) / f"{video_id}_features.csv"
        expected_visual_path = Path(sample_features_dir) / f"{video_id}_visual_features.csv"
        
        # Load features
        audio_df = dataset._load_audio_features(video_id)
        visual_df = dataset._load_visual_features(video_id)
        
        # Verify: If file exists, it should be loaded
        if expected_audio_path.exists():
            assert audio_df is not None, f"Audio features should be loaded for {video_id}"
        else:
            assert audio_df is None, f"Audio features should be None for {video_id}"
        
        if expected_visual_path.exists():
            assert visual_df is not None, f"Visual features should be loaded for {video_id}"
        else:
            assert visual_df is None, f"Visual features should be None for {video_id}"


# Property 15: Missing feature handling
def test_property_15_missing_feature_handling(sample_sequences_npz, temp_dir):
    """
    Property 15: Missing feature handling
    
    For any video without feature files, the video should still be included in the dataset
    with zero-filled features and appropriate modality mask.
    """
    # Create empty features directory
    features_dir = Path(temp_dir) / "empty_features"
    features_dir.mkdir()
    
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=str(features_dir),
        enable_multimodal=True
    )
    
    # All videos should still be accessible
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Get a sample
    sample = dataset[0]
    
    # Verify: Audio and visual should be zero-filled
    audio = sample['audio']
    visual = sample['visual']
    modality_mask = sample['modality_mask']
    
    # Check that audio and visual are zero-filled
    assert torch.all(audio == 0), "Audio should be zero-filled when missing"
    assert torch.all(visual == 0), "Visual should be zero-filled when missing"
    
    # Check modality mask
    assert not torch.any(modality_mask[:, 0]), "Audio modality should be marked as unavailable"
    assert not torch.any(modality_mask[:, 1]), "Visual modality should be marked as unavailable"
    assert torch.all(modality_mask[:, 2]), "Track modality should always be available"


# Property 31: Modality mask consistency
def test_property_31_modality_mask_consistency(sample_sequences_npz, sample_features_dir):
    """
    Property 31: Modality mask consistency
    
    For any batch of data, if modality_mask[i, j, k] = False, then the corresponding
    feature values should be zero-filled and not contribute to attention weights.
    """
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=sample_features_dir,
        enable_multimodal=True
    )
    
    # Get multiple samples
    samples = [dataset[i] for i in range(min(3, len(dataset)))]
    
    for sample in samples:
        audio = sample['audio']
        visual = sample['visual']
        track = sample['track']
        modality_mask = sample['modality_mask']
        
        # For each timestep
        for t in range(len(modality_mask)):
            # If audio is not available, audio features should be zero
            if not modality_mask[t, 0]:
                assert torch.all(audio[t] == 0), f"Audio features should be zero when modality_mask[{t}, 0] = False"
            
            # If visual is not available, visual features should be zero
            if not modality_mask[t, 1]:
                assert torch.all(visual[t] == 0), f"Visual features should be zero when modality_mask[{t}, 1] = False"
            
            # Track should always be available
            assert modality_mask[t, 2], f"Track modality should always be available at timestep {t}"


# Test batch collation
def test_batch_collation(sample_sequences_npz, sample_features_dir):
    """Test that batch collation works correctly"""
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=sample_features_dir,
        enable_multimodal=True
    )
    
    # Get multiple samples
    samples = [dataset[i] for i in range(min(4, len(dataset)))]
    
    # Collate
    batch = collate_fn(samples)
    
    # Verify batch shapes
    batch_size = len(samples)
    seq_len = samples[0]['track'].shape[0]
    
    assert batch['audio'].shape == (batch_size, seq_len, 4)
    assert batch['visual'].shape == (batch_size, seq_len, 522)
    assert batch['track'].shape == (batch_size, seq_len, 180)
    assert batch['targets'].shape == (batch_size, seq_len, 20, 9)
    assert batch['padding_mask'].shape == (batch_size, seq_len)
    assert batch['modality_mask'].shape == (batch_size, seq_len, 3)
    assert len(batch['video_ids']) == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# Property 16: Batch sequence length consistency
def test_property_16_batch_sequence_length_consistency(sample_sequences_npz, sample_features_dir):
    """
    Property 16: Batch sequence length consistency
    
    For any batch of sequences, all modalities (audio, visual, track) should have
    identical sequence length dimensions: audio.shape[1] == visual.shape[1] == track.shape[1]
    """
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=sample_features_dir,
        enable_multimodal=True
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Test multiple batches
    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio']
        visual = batch['visual']
        track = batch['track']
        targets = batch['targets']
        padding_mask = batch['padding_mask']
        modality_mask = batch['modality_mask']
        
        batch_size = audio.shape[0]
        seq_len_audio = audio.shape[1]
        seq_len_visual = visual.shape[1]
        seq_len_track = track.shape[1]
        seq_len_targets = targets.shape[1]
        seq_len_padding = padding_mask.shape[1]
        seq_len_modality = modality_mask.shape[1]
        
        # All sequence lengths should be identical
        assert seq_len_audio == seq_len_visual, \
            f"Audio seq_len ({seq_len_audio}) != Visual seq_len ({seq_len_visual})"
        assert seq_len_audio == seq_len_track, \
            f"Audio seq_len ({seq_len_audio}) != Track seq_len ({seq_len_track})"
        assert seq_len_audio == seq_len_targets, \
            f"Audio seq_len ({seq_len_audio}) != Targets seq_len ({seq_len_targets})"
        assert seq_len_audio == seq_len_padding, \
            f"Audio seq_len ({seq_len_audio}) != Padding mask seq_len ({seq_len_padding})"
        assert seq_len_audio == seq_len_modality, \
            f"Audio seq_len ({seq_len_audio}) != Modality mask seq_len ({seq_len_modality})"
        
        # Verify feature dimensions
        assert audio.shape[2] == 4, f"Audio features should be 4-dim, got {audio.shape[2]}"
        assert visual.shape[2] == 522, f"Visual features should be 522-dim, got {visual.shape[2]}"
        assert track.shape[2] == 180, f"Track features should be 180-dim, got {track.shape[2]}"
        assert targets.shape[2] == 20, f"Targets should have 20 tracks, got {targets.shape[2]}"
        assert targets.shape[3] == 9, f"Targets should have 9 params per track, got {targets.shape[3]}"
        assert modality_mask.shape[2] == 3, f"Modality mask should have 3 modalities, got {modality_mask.shape[2]}"
        
        # Test a few batches
        if batch_idx >= 2:
            break


# Property-based test for batch sequence length consistency with random batch sizes
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    batch_size=st.integers(min_value=1, max_value=8)
)
def test_property_16_batch_consistency_random_sizes(batch_size, sample_sequences_npz, sample_features_dir):
    """
    Property 16: Batch sequence length consistency (with random batch sizes)
    
    For any batch size, all modalities should have consistent sequence lengths.
    """
    dataset = MultimodalDataset(
        sequences_npz=sample_sequences_npz,
        features_dir=sample_features_dir,
        enable_multimodal=True
    )
    
    # Limit batch size to dataset size
    actual_batch_size = min(batch_size, len(dataset))
    
    if actual_batch_size == 0:
        return  # Skip if dataset is empty
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Test first batch
    batch = next(iter(dataloader))
    
    audio = batch['audio']
    visual = batch['visual']
    track = batch['track']
    
    # All sequence lengths should be identical
    assert audio.shape[1] == visual.shape[1] == track.shape[1], \
        f"Sequence lengths must match: audio={audio.shape[1]}, visual={visual.shape[1]}, track={track.shape[1]}"
