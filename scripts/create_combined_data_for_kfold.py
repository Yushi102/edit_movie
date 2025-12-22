"""
Combine train and val data for K-Fold Cross Validation

Merges train_sequences_cut_selection.npz and val_sequences_cut_selection.npz
into a single combined_sequences_cut_selection.npz file.
"""
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Combining train and val data for K-Fold Cross Validation...")
    
    data_dir = Path('preprocessed_data')
    train_path = data_dir / 'train_sequences_cut_selection.npz'
    val_path = data_dir / 'val_sequences_cut_selection.npz'
    output_path = data_dir / 'combined_sequences_cut_selection.npz'
    
    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val data not found: {val_path}")
    
    # Load data
    logger.info(f"Loading train data from {train_path}")
    train_data = np.load(train_path, allow_pickle=True)
    
    logger.info(f"Loading val data from {val_path}")
    val_data = np.load(val_path, allow_pickle=True)
    
    # Combine
    logger.info("Combining data...")
    combined_audio = np.concatenate([train_data['audio'], val_data['audio']], axis=0)
    combined_visual = np.concatenate([train_data['visual'], val_data['visual']], axis=0)
    combined_active = np.concatenate([train_data['active'], val_data['active']], axis=0)
    
    # Combine video names if they exist
    if 'video_names' in train_data and 'video_names' in val_data:
        combined_video_names = np.concatenate([
            train_data['video_names'],
            val_data['video_names']
        ])
    else:
        combined_video_names = None
    
    logger.info(f"Combined data:")
    logger.info(f"  Audio: {combined_audio.shape}")
    logger.info(f"  Visual: {combined_visual.shape}")
    logger.info(f"  Active: {combined_active.shape}")
    logger.info(f"  Total sequences: {len(combined_audio)}")
    
    # Calculate statistics
    total_samples = combined_active.size
    active_count = np.sum(combined_active == 1)
    inactive_count = np.sum(combined_active == 0)
    logger.info(f"  Active samples: {active_count} ({active_count/total_samples*100:.2f}%)")
    logger.info(f"  Inactive samples: {inactive_count} ({inactive_count/total_samples*100:.2f}%)")
    
    # Save
    logger.info(f"\nSaving combined data to {output_path}")
    if combined_video_names is not None:
        np.savez(
            output_path,
            audio=combined_audio,
            visual=combined_visual,
            active=combined_active,
            video_names=combined_video_names
        )
    else:
        np.savez(
            output_path,
            audio=combined_audio,
            visual=combined_visual,
            active=combined_active
        )
    
    logger.info(f"\n✅ Combined data saved!")
    logger.info(f"   Path: {output_path}")
    logger.info(f"   Size: {output_path.stat().st_size / 1024**2:.2f} MB")
    logger.info(f"\nYou can now run K-Fold Cross Validation with:")
    logger.info(f"   python src/cut_selection/train_cut_selection_kfold.py")


if __name__ == '__main__':
    main()
