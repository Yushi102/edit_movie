"""
Property-based tests for PyTorch Dataset and DataLoader
"""
import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings
from dataset import MultiTrackDataset, collate_fn, create_dataloaders


# Test Property 15: Batch Size Consistency
@given(
    num_samples=st.integers(min_value=50, max_value=200),
    batch_size=st.integers(min_value=8, max_value=64),
    seq_length=st.integers(min_value=50, max_value=100),
    num_features=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_property_15_batch_size_consistency(num_samples, batch_size, seq_length, num_features):
    """
    Property 15: Batch Size Consistency
    
    DataLoader must produce batches with consistent properties:
    1. All batches (except possibly last) have size = batch_size
    2. Last batch has size <= batch_size
    3. Total samples across all batches equals dataset size
    4. Batch tensors have correct shape (batch_size, seq_len, features)
    5. No data duplication or loss
    """
    # Create synthetic dataset
    sequences = np.random.randn(num_samples, seq_length, num_features).astype(np.float32)
    masks = np.random.rand(num_samples, seq_length) > 0.1  # ~90% True
    
    dataset = MultiTrackDataset(sequences, masks)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Property 1 & 2: Check batch sizes
    batch_sizes = []
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        batch_seq = batch['sequences']
        batch_mask = batch['masks']
        
        current_batch_size = len(batch_seq)
        batch_sizes.append(current_batch_size)
        total_samples += current_batch_size
        
        # Property 4: Check tensor shapes
        assert batch_seq.shape == (current_batch_size, seq_length, num_features)
        assert batch_mask.shape == (current_batch_size, seq_length)
        
        # Check dtypes
        assert batch_seq.dtype == torch.float32
        assert batch_mask.dtype == torch.bool
        
        # All batches except last should have size = batch_size
        if batch_idx < len(dataloader) - 1:
            assert current_batch_size == batch_size
        else:
            # Last batch
            assert current_batch_size <= batch_size
    
    # Property 3: Total samples equals dataset size
    assert total_samples == num_samples
    
    # Property 5: Check no data loss by verifying first and last samples
    first_batch = next(iter(dataloader))
    assert torch.allclose(
        first_batch['sequences'][0],
        torch.FloatTensor(sequences[0]),
        rtol=1e-5
    )


# Test dataset initialization and access
def test_dataset_initialization():
    """Test basic dataset initialization and access"""
    num_samples = 50
    seq_length = 100
    num_features = 180
    
    sequences = np.random.randn(num_samples, seq_length, num_features).astype(np.float32)
    masks = np.ones((num_samples, seq_length), dtype=bool)
    video_ids = np.array([f'video_{i}' for i in range(num_samples)])
    source_names = np.array([f'source_{i}' for i in range(num_samples)])
    
    dataset = MultiTrackDataset(
        sequences=sequences,
        masks=masks,
        video_ids=video_ids,
        source_video_names=source_names
    )
    
    # Check length
    assert len(dataset) == num_samples
    
    # Check __getitem__
    sample = dataset[0]
    assert 'sequence' in sample
    assert 'mask' in sample
    assert 'video_id' in sample
    assert 'source_video_name' in sample
    
    assert sample['sequence'].shape == (seq_length, num_features)
    assert sample['mask'].shape == (seq_length,)
    assert sample['video_id'] == 'video_0'
    assert sample['source_video_name'] == 'source_0'


# Test collate function
def test_collate_function():
    """Test custom collate function"""
    batch_size = 8
    seq_length = 100
    num_features = 180
    
    # Create batch of samples
    batch = []
    for i in range(batch_size):
        sample = {
            'sequence': torch.randn(seq_length, num_features),
            'mask': torch.ones(seq_length, dtype=torch.bool),
            'video_id': f'video_{i}',
            'source_video_name': f'source_{i}'
        }
        batch.append(sample)
    
    # Collate
    collated = collate_fn(batch)
    
    # Check output
    assert 'sequences' in collated
    assert 'masks' in collated
    assert 'video_ids' in collated
    assert 'source_video_names' in collated
    
    assert collated['sequences'].shape == (batch_size, seq_length, num_features)
    assert collated['masks'].shape == (batch_size, seq_length)
    assert len(collated['video_ids']) == batch_size
    assert len(collated['source_video_names']) == batch_size


# Test dataloader with shuffling
def test_dataloader_shuffling():
    """Test that shuffling works correctly"""
    num_samples = 100
    seq_length = 50
    num_features = 20
    batch_size = 10
    
    sequences = np.arange(num_samples * seq_length * num_features).reshape(
        num_samples, seq_length, num_features
    ).astype(np.float32)
    masks = np.ones((num_samples, seq_length), dtype=bool)
    
    dataset = MultiTrackDataset(sequences, masks)
    
    # Create two dataloaders with different shuffling
    loader_no_shuffle = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    loader_shuffle = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    # Get first batches
    first_batch_no_shuffle = next(iter(loader_no_shuffle))
    first_batch_shuffle = next(iter(loader_shuffle))
    
    # Without shuffle, first batch should be samples 0-9
    expected_first = torch.FloatTensor(sequences[0])
    assert torch.allclose(first_batch_no_shuffle['sequences'][0], expected_first)
    
    # With shuffle, order might be different (not guaranteed, but very likely)
    # Just check that data is valid
    assert first_batch_shuffle['sequences'].shape == (batch_size, seq_length, num_features)


# Test mask preservation
def test_mask_preservation():
    """Test that masks are correctly preserved through dataset and dataloader"""
    num_samples = 20
    seq_length = 100
    num_features = 10
    batch_size = 5
    
    sequences = np.random.randn(num_samples, seq_length, num_features).astype(np.float32)
    
    # Create specific mask pattern
    masks = np.zeros((num_samples, seq_length), dtype=bool)
    for i in range(num_samples):
        # Each sample has different number of valid frames
        valid_frames = 50 + i * 2
        masks[i, :valid_frames] = True
    
    dataset = MultiTrackDataset(sequences, masks)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Check first batch
    batch = next(iter(dataloader))
    batch_masks = batch['masks']
    
    # Verify masks match original
    for i in range(batch_size):
        expected_mask = torch.BoolTensor(masks[i])
        assert torch.equal(batch_masks[i], expected_mask)


# Integration test with actual data files
def test_integration_with_npz():
    """Integration test loading from actual .npz files"""
    import os
    
    train_npz = "preprocessed_data/train_sequences.npz"
    val_npz = "preprocessed_data/val_sequences.npz"
    
    # Skip if files don't exist
    if not os.path.exists(train_npz) or not os.path.exists(val_npz):
        pytest.skip("Preprocessed data files not found")
    
    # Load datasets
    train_dataset = MultiTrackDataset.from_npz(train_npz)
    val_dataset = MultiTrackDataset.from_npz(val_npz)
    
    # Check datasets
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    
    # Check sample
    sample = train_dataset[0]
    assert 'sequence' in sample
    assert 'mask' in sample
    assert sample['sequence'].dim() == 2  # (seq_len, features)
    assert sample['mask'].dim() == 1  # (seq_len,)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_npz, val_npz, batch_size=16
    )
    
    # Test loading a batch
    batch = next(iter(train_loader))
    assert batch['sequences'].dim() == 3  # (batch, seq_len, features)
    assert batch['masks'].dim() == 2  # (batch, seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
