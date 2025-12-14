"""
PyTorch Dataset and DataLoader for Multi-Track Transformer training
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTrackDataset(Dataset):
    """PyTorch Dataset for multi-track video editing data"""
    
    def __init__(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        video_ids: Optional[np.ndarray] = None,
        source_video_names: Optional[np.ndarray] = None,
        transform=None
    ):
        """
        Initialize dataset
        
        Args:
            sequences: Array of shape (N, seq_len, features)
            masks: Boolean array of shape (N, seq_len)
            video_ids: Optional array of video identifiers
            source_video_names: Optional array of source video names
            transform: Optional transform to apply to sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.masks = torch.BoolTensor(masks)
        self.video_ids = video_ids
        self.source_video_names = source_video_names
        self.transform = transform
        
        # Validate shapes
        assert len(self.sequences) == len(self.masks), \
            f"Sequences and masks must have same length: {len(self.sequences)} vs {len(self.masks)}"
        
        logger.info(f"Dataset initialized: {len(self)} samples, "
                   f"sequence_length={self.sequences.shape[1]}, "
                   f"features={self.sequences.shape[2]}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dict with keys:
            - 'sequence': FloatTensor of shape (seq_len, features)
            - 'mask': BoolTensor of shape (seq_len,)
            - 'video_id': str (if available)
            - 'source_video_name': str (if available)
        """
        sequence = self.sequences[idx]
        mask = self.masks[idx]
        
        # Apply transform if provided
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        sample = {
            'sequence': sequence,
            'mask': mask
        }
        
        if self.video_ids is not None:
            sample['video_id'] = str(self.video_ids[idx])
        
        if self.source_video_names is not None:
            sample['source_video_name'] = str(self.source_video_names[idx])
        
        return sample
    
    @classmethod
    def from_npz(cls, npz_path: str, transform=None):
        """
        Load dataset from .npz file
        
        Args:
            npz_path: Path to .npz file
            transform: Optional transform
        
        Returns:
            MultiTrackDataset instance
        """
        logger.info(f"Loading dataset from {npz_path}")
        data = np.load(npz_path)
        
        return cls(
            sequences=data['sequences'],
            masks=data['masks'],
            video_ids=data.get('video_ids'),
            source_video_names=data.get('source_video_names'),
            transform=transform
        )



def collate_fn(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dict with batched tensors
    """
    # Stack sequences and masks
    sequences = torch.stack([item['sequence'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    result = {
        'sequences': sequences,
        'masks': masks
    }
    
    # Add video_ids and source_video_names if available
    if 'video_id' in batch[0]:
        result['video_ids'] = [item['video_id'] for item in batch]
    
    if 'source_video_name' in batch[0]:
        result['source_video_names'] = [item['source_video_name'] for item in batch]
    
    return result


def create_dataloaders(
    train_npz: str,
    val_npz: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_npz: Path to training .npz file
        val_npz: Path to validation .npz file
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader)
    """
    logger.info("Creating dataloaders...")
    
    # Create datasets
    train_dataset = MultiTrackDataset.from_npz(train_npz)
    val_dataset = MultiTrackDataset.from_npz(val_npz)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset and dataloader")
    parser.add_argument("--train_npz", default="preprocessed_data/train_sequences.npz")
    parser.add_argument("--val_npz", default="preprocessed_data/val_sequences.npz")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.train_npz,
        args.val_npz,
        args.batch_size
    )
    
    # Test loading a batch
    logger.info("\nTesting batch loading...")
    for batch in train_loader:
        logger.info(f"Batch sequences shape: {batch['sequences'].shape}")
        logger.info(f"Batch masks shape: {batch['masks'].shape}")
        logger.info(f"Number of video_ids: {len(batch['video_ids'])}")
        logger.info(f"Sequences dtype: {batch['sequences'].dtype}")
        logger.info(f"Masks dtype: {batch['masks'].dtype}")
        logger.info(f"Sample video_id: {batch['video_ids'][0]}")
        break
    
    logger.info("\n✅ Dataset and DataLoader test complete!")
