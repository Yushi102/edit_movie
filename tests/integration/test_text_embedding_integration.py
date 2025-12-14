"""
Test text embedding integration with multimodal dataset
"""
import numpy as np
import pandas as pd
from multimodal_dataset import MultimodalDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_embedding():
    """Test that text embeddings are properly integrated"""
    
    logger.info("Testing text embedding integration...")
    
    # Create dataset with text embedding enabled
    dataset = MultimodalDataset(
        sequences_npz='preprocessed_data/train_sequences.npz',
        features_dir='input_features',
        enable_multimodal=True,
        use_text_embedding=True
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Text embedder dimension: {dataset.text_embedder.embedding_dim if dataset.text_embedder else 'None'}")
    
    # Get a sample
    sample = dataset[0]
    
    logger.info("\nSample keys:")
    for key in sample.keys():
        if isinstance(sample[key], np.ndarray):
            logger.info(f"  {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
        else:
            logger.info(f"  {key}: {type(sample[key])}")
    
    # Check audio features dimension
    if sample['audio'] is not None:
        audio_shape = sample['audio'].shape
        logger.info(f"\n✅ Audio features shape: {audio_shape}")
        logger.info(f"   Expected: (seq_len, 10) = (seq_len, 4 base + 6 text embedding)")
        
        if audio_shape[1] == 10:
            logger.info("   ✅ Correct! Text embeddings are included!")
        else:
            logger.warning(f"   ⚠️ Expected 10 features, got {audio_shape[1]}")
    else:
        logger.warning("⚠️ No audio features for this sample")
    
    # Test a few more samples
    logger.info("\nTesting multiple samples...")
    audio_dims = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        if sample['audio'] is not None:
            audio_dims.append(sample['audio'].shape[1])
    
    if audio_dims:
        logger.info(f"Audio feature dimensions across samples: {set(audio_dims)}")
        if len(set(audio_dims)) == 1 and audio_dims[0] == 10:
            logger.info("✅ All samples have consistent 10-dimensional audio features!")
        else:
            logger.warning(f"⚠️ Inconsistent dimensions: {audio_dims}")
    
    logger.info("\n✅ Text embedding integration test complete!")

if __name__ == "__main__":
    test_text_embedding()
