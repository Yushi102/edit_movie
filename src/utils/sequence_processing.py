"""
Sequence segmentation and padding for Multi-Track Transformer training
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceProcessor:
    """Handles sequence windowing, padding, and masking"""
    
    def __init__(
        self, 
        sequence_length: int = 100,
        overlap: int = 0,
        padding_value: float = 0.0
    ):
        """
        Initialize sequence processor
        
        Args:
            sequence_length: Target length for all sequences
            overlap: Number of frames to overlap between windows (for long sequences)
            padding_value: Value to use for padding short sequences
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.padding_value = padding_value
        
        if overlap >= sequence_length:
            raise ValueError("Overlap must be less than sequence_length")
        
        logger.info(f"SequenceProcessor initialized: length={sequence_length}, overlap={overlap}")

    
    def segment_long_sequence(
        self, 
        sequence: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Segment a long sequence into overlapping windows
        
        Args:
            sequence: Input sequence array of shape (seq_len, features)
            metadata: Optional metadata dict to attach to each window
        
        Returns:
            List of (window_array, window_metadata) tuples
        """
        seq_len = len(sequence)
        
        if seq_len <= self.sequence_length:
            # No segmentation needed
            mask = np.ones(seq_len, dtype=bool)
            meta = metadata.copy() if metadata else {}
            meta['original_length'] = seq_len
            meta['window_index'] = 0
            meta['total_windows'] = 1
            return [(sequence, meta)]
        
        # Calculate stride
        stride = self.sequence_length - self.overlap
        
        # Generate windows
        windows = []
        window_idx = 0
        start = 0
        
        while start < seq_len:
            end = min(start + self.sequence_length, seq_len)
            window = sequence[start:end]
            
            # Create metadata for this window
            meta = metadata.copy() if metadata else {}
            meta['original_length'] = seq_len
            meta['window_index'] = window_idx
            meta['window_start'] = start
            meta['window_end'] = end
            
            windows.append((window, meta))
            window_idx += 1
            
            # Move to next window
            start += stride
            
            # Stop if we've covered the entire sequence
            if end >= seq_len:
                break
        
        # Update total_windows in all metadata
        for _, meta in windows:
            meta['total_windows'] = len(windows)
        
        logger.debug(f"Segmented sequence of length {seq_len} into {len(windows)} windows")
        return windows

    
    def pad_short_sequence(
        self, 
        sequence: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Pad a short sequence to target length
        
        Args:
            sequence: Input sequence array of shape (seq_len, features)
            metadata: Optional metadata dict
        
        Returns:
            (padded_sequence, mask, metadata)
            - padded_sequence: Array of shape (sequence_length, features)
            - mask: Boolean array of shape (sequence_length,) where True = valid data
            - metadata: Updated metadata with padding info
        """
        seq_len = len(sequence)
        
        if seq_len >= self.sequence_length:
            # No padding needed, just truncate if necessary
            truncated = sequence[:self.sequence_length]
            mask = np.ones(self.sequence_length, dtype=bool)
            meta = metadata.copy() if metadata else {}
            meta['original_length'] = seq_len
            meta['padded'] = False
            meta['padding_length'] = 0
            return truncated, mask, meta
        
        # Pad sequence
        num_features = sequence.shape[1]
        padded = np.full(
            (self.sequence_length, num_features), 
            self.padding_value, 
            dtype=sequence.dtype
        )
        padded[:seq_len] = sequence
        
        # Create mask (True for valid data, False for padding)
        mask = np.zeros(self.sequence_length, dtype=bool)
        mask[:seq_len] = True
        
        # Update metadata
        meta = metadata.copy() if metadata else {}
        meta['original_length'] = seq_len
        meta['padded'] = True
        meta['padding_length'] = self.sequence_length - seq_len
        
        logger.debug(f"Padded sequence from {seq_len} to {self.sequence_length}")
        return padded, mask, meta

    
    def process_video_sequences(
        self, 
        df: pd.DataFrame,
        group_by: str = 'video_id'
    ) -> List[Dict]:
        """
        Process all video sequences in a dataframe
        
        Args:
            df: Input dataframe with video data
            group_by: Column to group sequences by (default: 'video_id')
        
        Returns:
            List of processed sequence dicts with keys:
            - 'sequence': np.ndarray of shape (sequence_length, features)
            - 'mask': np.ndarray of shape (sequence_length,) boolean mask
            - 'video_id': video identifier
            - 'source_video_name': source video name
            - 'metadata': dict with processing info
        """
        logger.info(f"Processing sequences grouped by '{group_by}'")
        
        # Get feature columns (exclude metadata columns)
        metadata_cols = ['video_id', 'source_video_name', 'time']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        logger.info(f"Using {len(feature_cols)} feature columns")
        
        processed_sequences = []
        
        # Group by video
        for video_id, group in df.groupby(group_by):
            # Sort by time
            group = group.sort_values('time')
            
            # Extract features
            sequence = group[feature_cols].values
            
            # Get metadata
            source_video_name = group['source_video_name'].iloc[0]
            metadata = {
                'video_id': video_id,
                'source_video_name': source_video_name,
                'num_frames': len(sequence)
            }
            
            # Check if sequence needs segmentation or padding
            if len(sequence) > self.sequence_length:
                # Segment long sequence
                windows = self.segment_long_sequence(sequence, metadata)
                
                for window, window_meta in windows:
                    # Pad window if needed (last window might be shorter)
                    padded, mask, final_meta = self.pad_short_sequence(window, window_meta)
                    
                    processed_sequences.append({
                        'sequence': padded,
                        'mask': mask,
                        'video_id': video_id,
                        'source_video_name': source_video_name,
                        'metadata': final_meta
                    })
            else:
                # Pad short sequence
                padded, mask, final_meta = self.pad_short_sequence(sequence, metadata)
                
                processed_sequences.append({
                    'sequence': padded,
                    'mask': mask,
                    'video_id': video_id,
                    'source_video_name': source_video_name,
                    'metadata': final_meta
                })
        
        logger.info(f"Processed {len(processed_sequences)} sequences from {df[group_by].nunique()} videos")
        return processed_sequences

    
    def get_statistics(self, processed_sequences: List[Dict]) -> Dict:
        """
        Get statistics about processed sequences
        
        Args:
            processed_sequences: List of processed sequence dicts
        
        Returns:
            Dict with statistics
        """
        stats = {
            'total_sequences': len(processed_sequences),
            'sequence_length': self.sequence_length,
            'overlap': self.overlap,
            'num_padded': sum(1 for s in processed_sequences if s['metadata'].get('padded', False)),
            'num_segmented': sum(1 for s in processed_sequences if s['metadata'].get('total_windows', 1) > 1),
            'avg_original_length': np.mean([s['metadata']['original_length'] for s in processed_sequences]),
            'avg_padding_length': np.mean([s['metadata'].get('padding_length', 0) for s in processed_sequences]),
            'unique_videos': len(set(s['video_id'] for s in processed_sequences))
        }
        
        return stats


def process_dataset(
    train_csv: str,
    val_csv: str,
    sequence_length: int = 100,
    overlap: int = 0,
    output_dir: str = 'preprocessed_data'
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process train and validation datasets
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        sequence_length: Target sequence length
        overlap: Overlap between windows
        output_dir: Directory to save processed sequences
    
    Returns:
        (train_sequences, val_sequences)
    """
    import os
    
    # Initialize processor
    processor = SequenceProcessor(
        sequence_length=sequence_length,
        overlap=overlap
    )
    
    # Load data
    logger.info(f"Loading training data from {train_csv}")
    train_df = pd.read_csv(train_csv)
    
    logger.info(f"Loading validation data from {val_csv}")
    val_df = pd.read_csv(val_csv)
    
    # Process sequences
    logger.info("Processing training sequences...")
    train_sequences = processor.process_video_sequences(train_df)
    
    logger.info("Processing validation sequences...")
    val_sequences = processor.process_video_sequences(val_df)
    
    # Get statistics
    train_stats = processor.get_statistics(train_sequences)
    val_stats = processor.get_statistics(val_sequences)
    
    logger.info("\n=== Training Set Statistics ===")
    for key, value in train_stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n=== Validation Set Statistics ===")
    for key, value in val_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Save processed sequences
    os.makedirs(output_dir, exist_ok=True)
    
    train_output = os.path.join(output_dir, 'train_sequences.npz')
    val_output = os.path.join(output_dir, 'val_sequences.npz')
    
    # Save as compressed numpy arrays
    np.savez_compressed(
        train_output,
        sequences=np.array([s['sequence'] for s in train_sequences]),
        masks=np.array([s['mask'] for s in train_sequences]),
        video_ids=np.array([s['video_id'] for s in train_sequences]),
        source_video_names=np.array([s['source_video_name'] for s in train_sequences])
    )
    
    np.savez_compressed(
        val_output,
        sequences=np.array([s['sequence'] for s in val_sequences]),
        masks=np.array([s['mask'] for s in val_sequences]),
        video_ids=np.array([s['video_id'] for s in val_sequences]),
        source_video_names=np.array([s['source_video_name'] for s in val_sequences])
    )
    
    logger.info(f"\nProcessed sequences saved:")
    logger.info(f"  Train: {train_output}")
    logger.info(f"  Val: {val_output}")
    
    return train_sequences, val_sequences


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process sequences for training")
    parser.add_argument("--train_csv", default="preprocessed_data/train_data.csv")
    parser.add_argument("--val_csv", default="preprocessed_data/val_data.csv")
    parser.add_argument("--sequence_length", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--output_dir", default="preprocessed_data")
    
    args = parser.parse_args()
    
    process_dataset(
        args.train_csv,
        args.val_csv,
        args.sequence_length,
        args.overlap,
        args.output_dir
    )
