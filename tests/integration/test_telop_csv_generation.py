"""
Test telop CSV generation without full feature extraction
"""
import pandas as pd
import numpy as np
from telop_extractor import TelopExtractor
from text_embedding import SimpleTextEmbedder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test telop extraction and embedding
xml_path = 'editxml/bandicam 2025-03-03 22-34-57-492.xml'
total_duration = 55.0  # Approximate duration

logger.info("Step 1: Extract telops from XML")
extractor = TelopExtractor(fps=10.0)
df_telop = extractor.extract_and_convert(xml_path, total_duration)

logger.info(f"\nTelop DataFrame:")
logger.info(f"  Shape: {df_telop.shape}")
logger.info(f"  Columns: {df_telop.columns.tolist()}")

# Show sample
logger.info(f"\nSample telop data (first 20 active frames):")
active = df_telop[df_telop['telop_active'] == 1].head(20)
logger.info(active)

logger.info("\nStep 2: Add telop text embeddings")
embedder = SimpleTextEmbedder()
telop_embeddings = embedder.embed_series(df_telop['telop_text'])

logger.info(f"\nTelop embeddings shape: {telop_embeddings.shape}")

# Add to DataFrame
for i in range(embedder.embedding_dim):
    df_telop[f'telop_emb_{i}'] = telop_embeddings[:, i]

logger.info(f"\nFinal DataFrame:")
logger.info(f"  Shape: {df_telop.shape}")
logger.info(f"  Columns: {df_telop.columns.tolist()}")

# Show sample with embeddings
logger.info(f"\nSample with embeddings (first 5 active frames):")
active_with_emb = df_telop[df_telop['telop_active'] == 1].head(5)
logger.info(active_with_emb)

# Simulate full audio features
logger.info("\nStep 3: Simulate full audio feature CSV")

# Create base audio features (dummy data)
df_audio = pd.DataFrame({
    'time': df_telop['time'],
    'audio_energy_rms': np.random.rand(len(df_telop)) * 0.1,
    'audio_is_speaking': np.random.randint(0, 2, len(df_telop)),
    'silence_duration_ms': np.random.randint(0, 1000, len(df_telop)),
    'speaker_id': np.nan,
    'text_is_active': np.random.randint(0, 2, len(df_telop)),
    'text_word': 'テスト'
})

# Add speech embeddings (dummy)
speech_embeddings = embedder.embed_series(df_audio['text_word'])
for i in range(embedder.embedding_dim):
    df_audio[f'speech_emb_{i}'] = speech_embeddings[:, i]

# Add telop data
df_audio['telop_active'] = df_telop['telop_active']
df_audio['telop_text'] = df_telop['telop_text']
for i in range(embedder.embedding_dim):
    df_audio[f'telop_emb_{i}'] = df_telop[f'telop_emb_{i}']

logger.info(f"\nFinal audio features CSV:")
logger.info(f"  Shape: {df_audio.shape}")
logger.info(f"  Columns ({len(df_audio.columns)}): {df_audio.columns.tolist()}")

# Count features
base_features = 5  # rms, is_speaking, silence, text_is_active, telop_active
speech_emb = 6
telop_emb = 6
total_numeric = base_features + speech_emb + telop_emb

logger.info(f"\nFeature breakdown:")
logger.info(f"  Base features: {base_features}")
logger.info(f"  Speech embeddings: {speech_emb}")
logger.info(f"  Telop embeddings: {telop_emb}")
logger.info(f"  Total numeric features: {total_numeric}")

# Save sample
output_path = 'test_features/sample_with_telop.csv'
df_audio.to_csv(output_path, index=False)
logger.info(f"\nSaved sample CSV to: {output_path}")

logger.info("\n✅ Telop CSV generation test complete!")
