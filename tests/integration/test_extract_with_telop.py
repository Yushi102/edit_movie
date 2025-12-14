"""
Test feature extraction with telop information
"""
import sys
sys.path.insert(0, '.')

from extract_video_features_parallel import extract_features_worker
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test with one video
video_path = r"D:\切り抜き\2025-3\2025-3-03\bandicam 2025-03-03 22-34-57-492.mp4"
output_dir = "test_features"

logger.info(f"Testing feature extraction with telop for: {video_path}")

result = extract_features_worker(video_path, output_dir)

logger.info(f"\nResult: {result}")

if result['status'] == 'Success':
    # Check the output CSV
    import os
    from pathlib import Path
    
    video_stem = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_stem}_features.csv")
    
    df = pd.read_csv(output_path)
    
    logger.info(f"\nOutput CSV:")
    logger.info(f"  Path: {output_path}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # Check for telop columns
    if 'telop_active' in df.columns and 'telop_text' in df.columns:
        logger.info("\n✅ Telop columns found!")
        
        # Show telop data
        telop_rows = df[df['telop_active'] == 1]
        logger.info(f"\nTelop active frames: {len(telop_rows)}")
        
        if len(telop_rows) > 0:
            logger.info("\nSample telop data:")
            logger.info(telop_rows[['time', 'telop_active', 'telop_text']].head(20))
    else:
        logger.warning("⚠️ Telop columns not found!")
