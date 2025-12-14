"""
Test script for MultimodalTransformer model
"""
import torch
import logging
from model import create_multimodal_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_multimodal_model():
    """Test multimodal model creation and forward pass"""
    logger.info("Testing MultimodalTransformer model...")
    
    # Create model
    model = create_multimodal_model(
        audio_features=4,
        visual_features=522,
        track_features=180,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        enable_multimodal=True,
        fusion_type='gated'
    )
    
    # Create dummy inputs
    batch_size = 4
    seq_len = 100
    
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    
    # Simulate padding in last 20 frames
    padding_mask[:, -20:] = False
    
    # Simulate missing audio for first half
    modality_mask[:, :seq_len//2, 0] = False
    
    logger.info(f"\nInput shapes:")
    logger.info(f"  Audio: {audio.shape}")
    logger.info(f"  Visual: {visual.shape}")
    logger.info(f"  Track: {track.shape}")
    logger.info(f"  Padding mask: {padding_mask.shape}")
    logger.info(f"  Modality mask: {modality_mask.shape}")
    
    # Forward pass
    logger.info("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(audio, visual, track, padding_mask, modality_mask)
    
    # Check outputs
    logger.info("\nOutput shapes:")
    for key, value in outputs.items():
        logger.info(f"  {key}: {value.shape}")
    
    # Verify output shapes
    expected_shapes = {
        'active': (batch_size, seq_len, 20, 2),
        'asset': (batch_size, seq_len, 20, 10),
        'scale': (batch_size, seq_len, 20, 1),
        'pos_x': (batch_size, seq_len, 20, 1),
        'pos_y': (batch_size, seq_len, 20, 1),
        'crop_l': (batch_size, seq_len, 20, 1),
        'crop_r': (batch_size, seq_len, 20, 1),
        'crop_t': (batch_size, seq_len, 20, 1),
        'crop_b': (batch_size, seq_len, 20, 1)
    }
    
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(outputs[key].shape)
        if actual_shape != expected_shape:
            logger.error(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
            all_correct = False
    
    if all_correct:
        logger.info("\n✅ All output shapes are correct!")
    else:
        logger.error("\n❌ Some output shapes are incorrect!")
        return False
    
    return True


def test_track_only_mode():
    """Test model in track-only mode (enable_multimodal=False)"""
    logger.info("\n\nTesting track-only mode...")
    
    # Create model with multimodal disabled
    model = create_multimodal_model(
        audio_features=4,
        visual_features=522,
        track_features=180,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        enable_multimodal=False
    )
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 50
    
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass (audio and visual should be ignored)
    with torch.no_grad():
        outputs = model(audio, visual, track, padding_mask)
    
    logger.info(f"Track-only mode output shape (active): {outputs['active'].shape}")
    assert outputs['active'].shape == (batch_size, seq_len, 20, 2)
    
    logger.info("✅ Track-only mode test passed!")
    return True


def test_different_fusion_types():
    """Test different fusion strategies"""
    logger.info("\n\nTesting different fusion types...")
    
    batch_size = 2
    seq_len = 50
    
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    
    for fusion_type in ['concat', 'add', 'gated']:
        logger.info(f"\nTesting fusion_type='{fusion_type}'...")
        
        model = create_multimodal_model(
            d_model=128,
            num_encoder_layers=2,
            enable_multimodal=True,
            fusion_type=fusion_type
        )
        
        with torch.no_grad():
            outputs = model(audio, visual, track)
        
        logger.info(f"  Output shape: {outputs['active'].shape}")
        assert outputs['active'].shape == (batch_size, seq_len, 20, 2)
    
    logger.info("\n✅ All fusion types test passed!")
    return True


if __name__ == "__main__":
    success = True
    
    success = success and test_multimodal_model()
    success = success and test_track_only_mode()
    success = success and test_different_fusion_types()
    
    if success:
        logger.info("\n\n✅ All multimodal model tests passed!")
    else:
        logger.error("\n\n❌ Some tests failed!")
        exit(1)
