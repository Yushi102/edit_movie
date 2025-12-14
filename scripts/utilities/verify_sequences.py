"""
Verify processed sequence data
"""
import numpy as np

# Load processed sequences
print("Loading processed sequences...")
train_data = np.load('preprocessed_data/train_sequences.npz')
val_data = np.load('preprocessed_data/val_sequences.npz')

print("\n=== Training Data ===")
print(f"Sequences shape: {train_data['sequences'].shape}")
print(f"Masks shape: {train_data['masks'].shape}")
print(f"Number of videos: {len(np.unique(train_data['video_ids']))}")
print(f"Sequence length: {train_data['sequences'].shape[1]}")
print(f"Number of features: {train_data['sequences'].shape[2]}")

print("\n=== Validation Data ===")
print(f"Sequences shape: {val_data['sequences'].shape}")
print(f"Masks shape: {val_data['masks'].shape}")
print(f"Number of videos: {len(np.unique(val_data['video_ids']))}")
print(f"Sequence length: {val_data['sequences'].shape[1]}")
print(f"Number of features: {val_data['sequences'].shape[2]}")

# Check mask statistics
print("\n=== Mask Statistics ===")
train_mask_ratio = train_data['masks'].mean()
val_mask_ratio = val_data['masks'].mean()
print(f"Train mask ratio (valid data): {train_mask_ratio:.2%}")
print(f"Val mask ratio (valid data): {val_mask_ratio:.2%}")

# Check for NaN or Inf
print("\n=== Data Quality ===")
train_has_nan = np.isnan(train_data['sequences']).any()
train_has_inf = np.isinf(train_data['sequences']).any()
val_has_nan = np.isnan(val_data['sequences']).any()
val_has_inf = np.isinf(val_data['sequences']).any()

print(f"Train has NaN: {train_has_nan}")
print(f"Train has Inf: {train_has_inf}")
print(f"Val has NaN: {val_has_nan}")
print(f"Val has Inf: {val_has_inf}")

# Sample data
print("\n=== Sample Sequence ===")
sample_seq = train_data['sequences'][0]
sample_mask = train_data['masks'][0]
print(f"Sample sequence shape: {sample_seq.shape}")
print(f"Sample mask: {sample_mask[:20]}...")  # First 20 values
print(f"Valid frames in sample: {sample_mask.sum()}/{len(sample_mask)}")
print(f"Sample data range: [{sample_seq.min():.3f}, {sample_seq.max():.3f}]")

print("\n✅ Sequence processing verification complete!")
