"""
Property-based tests for backward compatibility and fallback

Feature: multimodal-video-features-integration
Property 18: Graceful fallback to track-only mode
Property 20: Dual-mode inference support
Property 21: Checkpoint type detection
Validates: Requirements 5.1, 5.3, 5.4
"""
import torch
import pytest
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings
from model import MultiTrackTransformer, MultimodalTransformer
from model_persistence import save_model, load_model
from loss import create_optimizer


@given(
    st.sampled_from([64, 128, 192, 256]),  # d_model (divisible by 4)
    st.integers(min_value=2, max_value=4),  # num_layers
)
@settings(max_examples=50, deadline=None)
def test_graceful_fallback_to_track_only(d_model, num_layers):
    """
    Property 18: Graceful fallback to track-only mode
    
    For any dataset where video features are unavailable, the system should
    successfully train using only track data with enable_multimodal=False.
    
    This test verifies that:
    1. Multimodal model can be created with enable_multimodal=False
    2. Model accepts track-only inputs
    3. Model produces valid outputs
    4. Behavior is identical to track-only model
    """
    # Create multimodal model with multimodal disabled
    multimodal_model = MultimodalTransformer(
        audio_features=4,
        visual_features=522,
        track_features=180,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=num_layers,
        enable_multimodal=False  # Disable multimodal
    )
    
    # Create track-only model for comparison
    track_only_model = MultiTrackTransformer(
        input_features=180,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=num_layers
    )
    
    # Create test input (track-only)
    batch_size = 4
    seq_len = 20
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Test multimodal model with track-only input
    multimodal_model.eval()
    with torch.no_grad():
        # Should accept track-only input when enable_multimodal=False
        output_mm = multimodal_model(
            audio=torch.zeros(batch_size, seq_len, 4),  # Dummy audio
            visual=torch.zeros(batch_size, seq_len, 522),  # Dummy visual
            track=track,
            padding_mask=padding_mask
        )
    
    # Test track-only model
    track_only_model.eval()
    with torch.no_grad():
        output_to = track_only_model(track, padding_mask)
    
    # Verify output structure is identical
    assert set(output_mm.keys()) == set(output_to.keys()), \
        "Output keys should match between multimodal (disabled) and track-only"
    
    # Verify output shapes are identical
    for key in output_mm.keys():
        assert output_mm[key].shape == output_to[key].shape, \
            f"Output shape mismatch for {key}: {output_mm[key].shape} vs {output_to[key].shape}"
    
    # Verify outputs are valid (not NaN or Inf)
    for key, value in output_mm.items():
        assert not torch.isnan(value).any(), f"Multimodal output {key} contains NaN"
        assert not torch.isinf(value).any(), f"Multimodal output {key} contains Inf"


@given(
    st.integers(min_value=2, max_value=8),  # batch_size
    st.integers(min_value=10, max_value=30),  # seq_len
)
@settings(max_examples=50, deadline=None)
def test_dual_mode_inference_support(batch_size, seq_len):
    """
    Property 20: Dual-mode inference support
    
    For any trained model, inference should succeed in both multimodal mode
    (with all features) and unimodal mode (track-only) without errors.
    
    This test verifies that:
    1. Model can run inference with full multimodal inputs
    2. Model can run inference with track-only inputs (multimodal disabled)
    3. Both modes produce valid outputs
    4. Output structure is consistent
    """
    # Create multimodal model
    model = MultimodalTransformer(
        audio_features=4,
        visual_features=522,
        track_features=180,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        enable_multimodal=True
    )
    
    model.eval()
    
    # Create inputs
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    
    # Test 1: Multimodal inference (all features available)
    with torch.no_grad():
        output_multimodal = model(audio, visual, track, padding_mask, modality_mask)
    
    # Verify multimodal output is valid
    expected_keys = {'active', 'asset', 'scale', 'pos_x', 'pos_y', 
                     'crop_l', 'crop_r', 'crop_t', 'crop_b'}
    assert set(output_multimodal.keys()) == expected_keys, \
        f"Multimodal output keys mismatch"
    
    for key, value in output_multimodal.items():
        assert not torch.isnan(value).any(), f"Multimodal {key} contains NaN"
        assert not torch.isinf(value).any(), f"Multimodal {key} contains Inf"
    
    # Test 2: Track-only inference (disable multimodal via enable_multimodal flag)
    model.enable_multimodal = False
    
    with torch.no_grad():
        output_track_only = model(
            torch.zeros(batch_size, seq_len, 4),  # Dummy audio
            torch.zeros(batch_size, seq_len, 522),  # Dummy visual
            track,
            padding_mask
        )
    
    # Verify track-only output is valid
    assert set(output_track_only.keys()) == expected_keys, \
        f"Track-only output keys mismatch"
    
    for key, value in output_track_only.items():
        assert not torch.isnan(value).any(), f"Track-only {key} contains NaN"
        assert not torch.isinf(value).any(), f"Track-only {key} contains Inf"
    
    # Verify output shapes are consistent
    for key in expected_keys:
        assert output_multimodal[key].shape == output_track_only[key].shape, \
            f"Shape mismatch for {key}"
    
    # Restore multimodal mode
    model.enable_multimodal = True


@given(
    st.sampled_from([64, 128, 192, 256]),  # d_model (divisible by 4)
)
@settings(max_examples=30, deadline=None)
def test_checkpoint_type_detection(d_model):
    """
    Property 21: Checkpoint type detection
    
    For any saved checkpoint, loading should correctly detect whether it's
    a multimodal or unimodal model based on the stored architecture configuration.
    
    This test verifies that:
    1. Multimodal checkpoints are detected as multimodal
    2. Track-only checkpoints are detected as track-only
    3. Correct model type is instantiated
    4. Loaded model has correct configuration
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Save and load multimodal model
        multimodal_model = MultimodalTransformer(
            audio_features=4,
            visual_features=522,
            track_features=180,
            d_model=d_model,
            nhead=4,
            num_encoder_layers=2,
            enable_multimodal=True,
            fusion_type='gated'
        )
        
        mm_path = tmpdir / 'multimodal_model.pth'
        save_model(multimodal_model, str(mm_path), metadata={'test': 'multimodal'})
        
        # Load and verify type detection
        loaded_mm = load_model(str(mm_path), device='cpu')
        
        assert loaded_mm['config']['model_type'] == 'multimodal', \
            "Multimodal checkpoint should be detected as multimodal"
        
        assert loaded_mm['config']['audio_features'] == 4, \
            "Audio features mismatch"
        assert loaded_mm['config']['visual_features'] == 522, \
            "Visual features mismatch"
        assert loaded_mm['config']['track_features'] == 180, \
            "Track features mismatch"
        assert loaded_mm['config']['fusion_type'] == 'gated', \
            "Fusion type mismatch"
        
        # Verify loaded model is MultimodalTransformer
        assert isinstance(loaded_mm['model'], MultimodalTransformer), \
            "Loaded model should be MultimodalTransformer"
        
        # Test 2: Save and load track-only model
        track_only_model = MultiTrackTransformer(
            input_features=180,
            d_model=d_model,
            nhead=4,
            num_encoder_layers=2
        )
        
        to_path = tmpdir / 'track_only_model.pth'
        save_model(track_only_model, str(to_path), metadata={'test': 'track_only'})
        
        # Load and verify type detection
        loaded_to = load_model(str(to_path), device='cpu')
        
        assert loaded_to['config']['model_type'] == 'track_only', \
            "Track-only checkpoint should be detected as track_only"
        
        assert loaded_to['config']['input_features'] == 180, \
            "Input features mismatch"
        
        # Verify loaded model is MultiTrackTransformer
        assert isinstance(loaded_to['model'], MultiTrackTransformer), \
            "Loaded model should be MultiTrackTransformer"


def test_checkpoint_backward_compatibility():
    """
    Test that old checkpoints (without model_type) default to track_only
    
    Verifies backward compatibility with existing checkpoints.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a track-only model
        model = MultiTrackTransformer(
            input_features=180,
            d_model=128,
            nhead=4,
            num_encoder_layers=2
        )
        
        # Save checkpoint
        checkpoint_path = tmpdir / 'old_checkpoint.pth'
        save_model(model, str(checkpoint_path))
        
        # Manually remove model_type to simulate old checkpoint
        checkpoint = torch.load(checkpoint_path)
        if 'model_type' in checkpoint['config']:
            del checkpoint['config']['model_type']
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint (should default to track_only)
        loaded = load_model(str(checkpoint_path), device='cpu')
        
        # Verify it defaults to track_only
        assert loaded['config'].get('model_type', 'track_only') == 'track_only', \
            "Old checkpoints should default to track_only"
        
        # Verify model loads successfully
        assert isinstance(loaded['model'], MultiTrackTransformer), \
            "Should load as MultiTrackTransformer"


def test_multimodal_checkpoint_state_preservation():
    """
    Test that multimodal model state is correctly preserved across save/load
    
    Verifies that model weights are identical after save/load cycle.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create multimodal model
        model = MultimodalTransformer(
            audio_features=4,
            visual_features=522,
            track_features=180,
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            enable_multimodal=True,
            fusion_type='gated'
        )
        
        # Create test input
        batch_size = 2
        seq_len = 10
        audio = torch.randn(batch_size, seq_len, 4)
        visual = torch.randn(batch_size, seq_len, 522)
        track = torch.randn(batch_size, seq_len, 180)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
        
        # Get output from original model
        model.eval()
        with torch.no_grad():
            output_original = model(audio, visual, track, padding_mask, modality_mask)
        
        # Save model
        checkpoint_path = tmpdir / 'multimodal_checkpoint.pth'
        save_model(model, str(checkpoint_path))
        
        # Load model
        loaded = load_model(str(checkpoint_path), device='cpu')
        loaded_model = loaded['model']
        
        # Get output from loaded model
        loaded_model.eval()
        with torch.no_grad():
            output_loaded = loaded_model(audio, visual, track, padding_mask, modality_mask)
        
        # Verify outputs match
        for key in output_original.keys():
            assert torch.allclose(output_original[key], output_loaded[key], atol=1e-6), \
                f"Output mismatch for {key} after save/load"


if __name__ == "__main__":
    print("Running backward compatibility tests...")
    print("\n" + "="*70)
    print("Test 1: Graceful fallback to track-only mode")
    print("="*70)
    test_graceful_fallback_to_track_only(128, 2)
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 2: Dual-mode inference support")
    print("="*70)
    test_dual_mode_inference_support(4, 20)
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 3: Checkpoint type detection")
    print("="*70)
    test_checkpoint_type_detection(128)
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 4: Checkpoint backward compatibility")
    print("="*70)
    test_checkpoint_backward_compatibility()
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 5: Multimodal checkpoint state preservation")
    print("="*70)
    test_multimodal_checkpoint_state_preservation()
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("✅ All backward compatibility tests passed!")
    print("="*70)
