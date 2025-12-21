"""
Multimodal Dataset Loader for Multi-Track Transformer

This module provides a PyTorch Dataset that loads and aligns audio, visual,
and track features for multimodal video editing prediction.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.utils.feature_alignment import FeatureAligner
from src.training.multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
from src.data_preparation.text_embedding import SimpleTextEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal video editing data
    
    Loads pre-integrated sequences from NPZ and splits them into
    audio, visual, and track modalities.
    """
    
    def __init__(
        self,
        sequences_npz: str,
        features_dir: str = None,  # Not used, kept for compatibility
        audio_preprocessor: Optional[AudioFeaturePreprocessor] = None,  # Not used
        visual_preprocessor: Optional[VisualFeaturePreprocessor] = None,  # Not used
        enable_multimodal: bool = True,
        tolerance: float = 0.05,  # Not used
        use_text_embedding: bool = True  # Not used
    ):
        """
        Initialize MultimodalDataset
        
        Args:
            sequences_npz: Path to .npz file with integrated sequences (audio+visual+track)
            features_dir: Not used (kept for compatibility)
            audio_preprocessor: Not used (data already normalized)
            visual_preprocessor: Not used (data already normalized)
            enable_multimodal: Whether to use multimodal features
            tolerance: Not used (kept for compatibility)
            use_text_embedding: Not used (kept for compatibility)
        """
        self.enable_multimodal = enable_multimodal
        
        # Load integrated sequences
        logger.info(f"Loading integrated sequences from {sequences_npz}")
        data = np.load(sequences_npz)
        self.sequences = data['sequences']  # (N, seq_len, 917)
        self.masks = data['masks']  # (N, seq_len)
        self.video_ids = data.get('video_ids', None)
        self.source_video_names = data.get('source_video_names', None)
        
        # Feature dimensions (from integrated data)
        # sequences shape: (N, seq_len, 977)
        # 977 = audio(215) + visual(522) + track(240)
        self.audio_dim = 215
        self.visual_dim = 522
        self.track_dim = 240
        
        total_dim = self.sequences.shape[2]
        expected_dim = self.audio_dim + self.visual_dim + self.track_dim
        
        if total_dim != expected_dim:
            logger.warning(f"Dimension mismatch: expected {expected_dim}, got {total_dim}")
            logger.warning(f"  Audio: {self.audio_dim}, Visual: {self.visual_dim}, Track: {self.track_dim}")
        
        logger.info(f"MultimodalDataset initialized:")
        logger.info(f"  Total sequences: {len(self.sequences)}")
        logger.info(f"  Sequence length: {self.sequences.shape[1]}")
        logger.info(f"  Total dimensions: {total_dim}")
        logger.info(f"  Audio dimensions: {self.audio_dim}")
        logger.info(f"  Visual dimensions: {self.visual_dim}")
        logger.info(f"  Track dimensions: {self.track_dim}")
        logger.info(f"  Enable multimodal: {self.enable_multimodal}")
    
    def _get_video_id(self, idx: int) -> str:
        """Get video ID for a given index"""
        if self.video_ids is not None:
            return str(self.video_ids[idx])
        elif self.source_video_names is not None:
            return str(self.source_video_names[idx])
        else:
            return f"video_{idx}"
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dict with keys:
            - 'audio': FloatTensor of shape (seq_len, 215)
            - 'visual': FloatTensor of shape (seq_len, 522)
            - 'track': FloatTensor of shape (seq_len, 240)
            - 'targets': FloatTensor of shape (seq_len, 20, 12)
            - 'padding_mask': BoolTensor of shape (seq_len,)
            - 'modality_mask': BoolTensor of shape (seq_len, 3) for [audio, visual, track]
            - 'video_id': str
        """
        video_id = self._get_video_id(idx)
        
        # Get integrated sequence and mask
        integrated_seq = self.sequences[idx]  # (seq_len, 977)
        padding_mask = self.masks[idx]  # (seq_len,)
        seq_len = len(integrated_seq)
        
        # Split integrated sequence into modalities
        # 977 = audio(215) + visual(522) + track(240)
        audio_features = integrated_seq[:, :self.audio_dim]  # (seq_len, 215)
        visual_features = integrated_seq[:, self.audio_dim:self.audio_dim+self.visual_dim]  # (seq_len, 522)
        track_features = integrated_seq[:, self.audio_dim+self.visual_dim:]  # (seq_len, 240)
        
        # Create modality mask (all modalities are available in integrated data)
        modality_mask = np.ones((seq_len, 3), dtype=bool)
        
        if not self.enable_multimodal:
            # Track-only mode: disable audio and visual
            modality_mask[:, 0] = False  # No audio
            modality_mask[:, 1] = False  # No visual
        
        # Reshape track sequence to (seq_len, 20, 12) for targets
        # Note: track_features is (seq_len, 240) = (seq_len, 20, 12)
        targets = track_features.reshape(seq_len, 20, 12)
        
        # Convert active values to binary (0 or 1)
        # Active values are normalized, so we need to threshold them
        # Positive values -> 1 (active), Negative values -> 0 (inactive)
        targets[:, :, 0] = (targets[:, :, 0] > 0).astype(np.float32)
        
        # Convert to tensors
        sample = {
            'audio': torch.FloatTensor(audio_features),
            'visual': torch.FloatTensor(visual_features),
            'track': torch.FloatTensor(track_features),
            'targets': torch.FloatTensor(targets),
            'padding_mask': torch.BoolTensor(padding_mask),
            'modality_mask': torch.BoolTensor(modality_mask),
            'video_id': video_id
        }
        
        return sample


def collate_fn(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dict with batched tensors
    """
    # Stack all tensors
    audio = torch.stack([item['audio'] for item in batch])
    visual = torch.stack([item['visual'] for item in batch])
    track = torch.stack([item['track'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    padding_mask = torch.stack([item['padding_mask'] for item in batch])
    modality_mask = torch.stack([item['modality_mask'] for item in batch])
    
    result = {
        'audio': audio,
        'visual': visual,
        'track': track,
        'targets': targets,
        'padding_mask': padding_mask,
        'modality_mask': modality_mask,
        'video_ids': [item['video_id'] for item in batch]
    }
    
    return result


def create_multimodal_dataloaders(
    train_npz: str,
    val_npz: str,
    features_dir: str,
    audio_preprocessor_path: Optional[str] = None,
    visual_preprocessor_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    enable_multimodal: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for multimodal data
    
    Args:
        train_npz: Path to training .npz file
        val_npz: Path to validation .npz file
        features_dir: Directory containing feature CSV files
        audio_preprocessor_path: Path to saved AudioFeaturePreprocessor
        visual_preprocessor_path: Path to saved VisualFeaturePreprocessor
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        enable_multimodal: Whether to enable multimodal features
    
    Returns:
        (train_loader, val_loader)
    """
    logger.info("Creating multimodal dataloaders...")
    
    # Load preprocessors if provided
    audio_preprocessor = None
    visual_preprocessor = None
    
    if audio_preprocessor_path and Path(audio_preprocessor_path).exists():
        audio_preprocessor = AudioFeaturePreprocessor.load(audio_preprocessor_path)
        logger.info(f"Loaded audio preprocessor from {audio_preprocessor_path}")
    
    if visual_preprocessor_path and Path(visual_preprocessor_path).exists():
        visual_preprocessor = VisualFeaturePreprocessor.load(visual_preprocessor_path)
        logger.info(f"Loaded visual preprocessor from {visual_preprocessor_path}")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        sequences_npz=train_npz,
        features_dir=features_dir,
        audio_preprocessor=audio_preprocessor,
        visual_preprocessor=visual_preprocessor,
        enable_multimodal=enable_multimodal,
        use_text_embedding=True  # Enable text embedding
    )
    
    val_dataset = MultimodalDataset(
        sequences_npz=val_npz,
        features_dir=features_dir,
        audio_preprocessor=audio_preprocessor,
        visual_preprocessor=visual_preprocessor,
        enable_multimodal=enable_multimodal,
        use_text_embedding=True  # Enable text embedding
    )
    
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
    
    parser = argparse.ArgumentParser(description="Test multimodal dataset and dataloader")
    parser.add_argument("--train_npz", default="preprocessed_data/train_sequences.npz")
    parser.add_argument("--val_npz", default="preprocessed_data/val_sequences.npz")
    parser.add_argument("--features_dir", default="input_features")
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader = create_multimodal_dataloaders(
        args.train_npz,
        args.val_npz,
        args.features_dir,
        batch_size=args.batch_size,
        enable_multimodal=True
    )
    
    # Test loading a batch
    logger.info("\nTesting batch loading...")
    for batch in train_loader:
        logger.info(f"Batch audio shape: {batch['audio'].shape}")
        logger.info(f"Batch visual shape: {batch['visual'].shape}")
        logger.info(f"Batch track shape: {batch['track'].shape}")
        logger.info(f"Batch targets shape: {batch['targets'].shape}")
        logger.info(f"Batch padding_mask shape: {batch['padding_mask'].shape}")
        logger.info(f"Batch modality_mask shape: {batch['modality_mask'].shape}")
        logger.info(f"Number of video_ids: {len(batch['video_ids'])}")
        logger.info(f"Sample video_id: {batch['video_ids'][0]}")
        logger.info(f"Modality mask sample:\n{batch['modality_mask'][0][:5]}")
        break
    
    logger.info("\n✅ Multimodal dataset and DataLoader test complete!")
