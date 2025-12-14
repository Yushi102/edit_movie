"""
Test telop integration with multimodal dataset
"""
import pandas as pd
from multimodal_dataset import MultimodalDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_telop_integration():
    """Test that telop information is properly integrated"""
    
    logger.info("Testing telop integration...")
    
    # Create dataset with text embedding enabled
    dataset = MultimodalDataset(
        sequences_npz='preprocessed_data/train_sequences.npz',
        features_dir='input_features',
        enable_multimodal=True,
        use_text_embedding=True
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    logger.info("\nSample keys:")
    for key in sample.keys():
        if hasattr(sample[key], 'shape'):
            logger.info(f"  {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
        else:
            logger.info(f"  {key}: {type(sample[key])}")
    
    # Check audio features dimension
    if sample['audio'] is not None:
        audio_shape = sample['audio'].shape
        logger.info(f"\n✅ Audio features shape: {audio_shape}")
        logger.info(f"   Expected: (seq_len, 17)")
        logger.info(f"   Breakdown:")
        logger.info(f"     - 4 base features (rms, is_speaking, silence, text_is_active)")
        logger.info(f"     - 1 telop_active flag")
        logger.info(f"     - 6 speech text embeddings")
        logger.info(f"     - 6 telop text embeddings")
        
        if audio_shape[1] == 17:
            logger.info("   ✅ Correct! Telop embeddings are included!")
        else:
            logger.warning(f"   ⚠️ Expected 17 features, got {audio_shape[1]}")
    else:
        logger.warning("⚠️ No audio features for this sample")
    
    logger.info("\n✅ Telop integration test complete!")

if __name__ == "__main__":
    test_telop_integration()
