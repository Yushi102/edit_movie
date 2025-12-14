"""
Verify that text content is properly embedded
"""
import pandas as pd
import numpy as np
from multimodal_dataset import MultimodalDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_text_content():
    """Verify text content is embedded"""
    
    # Load a sample audio CSV
    audio_file = 'input_features/bandicam 2025-03-03 22-34-57-492_features.csv'
    df = pd.read_csv(audio_file)
    
    logger.info(f"Loaded {audio_file}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Show text content
    logger.info("\nText content (first 20 rows with text):")
    text_rows = df[df['text_word'].notna()].head(20)
    logger.info(text_rows[['time', 'text_is_active', 'text_word']])
    
    # Create dataset and load this video
    dataset = MultimodalDataset(
        sequences_npz='preprocessed_data/train_sequences.npz',
        features_dir='input_features',
        enable_multimodal=True,
        use_text_embedding=True
    )
    
    # Find the sample index for this video
    video_name = 'bandicam 2025-03-03 22-34-57-492'
    sample_idx = None
    for i in range(len(dataset)):
        if dataset._get_video_id(i) == video_name:
            sample_idx = i
            break
    
    if sample_idx is not None:
        logger.info(f"\nFound video at index {sample_idx}")
        
        # Load audio features directly
        audio_df = dataset._load_audio_features(video_name)
        
        if audio_df is not None:
            logger.info(f"\nAudio DataFrame shape: {audio_df.shape}")
            logger.info(f"Columns: {audio_df.columns.tolist()}")
            
            # Check if text embeddings were added
            text_emb_cols = [col for col in audio_df.columns if col.startswith('text_emb_')]
            logger.info(f"\nText embedding columns: {text_emb_cols}")
            
            if text_emb_cols:
                logger.info("\n✅ Text embeddings successfully added!")
                logger.info("\nSample text embeddings (first 5 rows with text):")
                text_rows = audio_df[audio_df['text_word'].notna()].head(5)
                logger.info(text_rows[['time', 'text_word'] + text_emb_cols])
            else:
                logger.warning("⚠️ No text embedding columns found!")
        else:
            logger.warning("Could not load audio features")
    else:
        logger.warning(f"Could not find video: {video_name}")

if __name__ == "__main__":
    verify_text_content()
