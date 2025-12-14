"""
Property-based tests for MultimodalTransformer model

Tests Property 19 from the design document.
"""
import pytest
import torch
from hypothesis import given, strategies as st, settings

from model import create_multimodal_model


# Property 19: Multimodal flag respect
@settings(max_examples=50, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=10, max_value=100),
    d_model=st.integers(min_value=8, max_value=32).map(lambda x: x * 8)  # Ensure divisible by 8
)
def test_property_19_multimodal_flag_respect(batch_size, seq_len, d_model):
    """
    Property 19: Multimodal flag respect
    
    For any model configuration, setting enable_multimodal=False should result
    in the model ignoring audio and visual inputs and processing only track data.
    """
    # Create two models: one with multimodal enabled, one disabled
    model_multimodal = create_multimodal_model(
        d_model=d_model,
        num_encoder_layers=2,
        enable_multimodal=True,
        fusion_type='gated'
    )
    
    model_track_only = create_multimodal_model(
        d_model=d_model,
        num_encoder_layers=2,
        enable_multimodal=False
    )
    
    # Create inputs
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    
    # Forward pass for both models
    with torch.no_grad():
        outputs_multimodal = model_multimodal(audio, visual, track)
        outputs_track_only = model_track_only(audio, visual, track)
    
    # Both should produce valid outputs
    assert outputs_multimodal['active'].shape == (batch_size, seq_len, 20, 2)
    assert outputs_track_only['active'].shape == (batch_size, seq_len, 20, 2)
    
    # Track-only model should have fewer parameters (no fusion module)
    params_multimodal = model_multimodal.count_parameters()
    params_track_only = model_track_only.count_parameters()
    
    assert params_track_only < params_multimodal, \
        f"Track-only model should have fewer parameters: {params_track_only} vs {params_multimodal}"


def test_multimodal_flag_with_different_audio_visual():
    """
    Test that track-only mode produces same output regardless of audio/visual inputs
    """
    batch_size = 2
    seq_len = 50
    
    # Create track-only model in eval mode (to disable dropout)
    model = create_multimodal_model(
        d_model=128,
        num_encoder_layers=2,
        enable_multimodal=False,
        dropout=0.0  # Disable dropout for deterministic output
    )
    model.eval()
    
    # Same track input
    track = torch.randn(batch_size, seq_len, 180)
    
    # Different audio/visual inputs
    audio1 = torch.randn(batch_size, seq_len, 4)
    visual1 = torch.randn(batch_size, seq_len, 522)
    
    audio2 = torch.randn(batch_size, seq_len, 4)
    visual2 = torch.randn(batch_size, seq_len, 522)
    
    # Forward pass with different audio/visual
    with torch.no_grad():
        outputs1 = model(audio1, visual1, track)
        outputs2 = model(audio2, visual2, track)
    
    # Outputs should be identical (since audio/visual are ignored)
    for key in outputs1.keys():
        assert torch.allclose(outputs1[key], outputs2[key], atol=1e-5), \
            f"Track-only mode should produce identical outputs for {key}"


def test_multimodal_mode_uses_all_modalities():
    """
    Test that multimodal mode actually uses audio and visual inputs
    """
    batch_size = 2
    seq_len = 50
    
    # Create multimodal model
    model = create_multimodal_model(
        d_model=128,
        num_encoder_layers=2,
        enable_multimodal=True,
        fusion_type='gated'
    )
    
    # Same track input
    track = torch.randn(batch_size, seq_len, 180)
    
    # Different audio/visual inputs
    audio1 = torch.randn(batch_size, seq_len, 4)
    visual1 = torch.randn(batch_size, seq_len, 522)
    
    audio2 = torch.randn(batch_size, seq_len, 4)
    visual2 = torch.randn(batch_size, seq_len, 522)
    
    # Forward pass with different audio/visual
    with torch.no_grad():
        outputs1 = model(audio1, visual1, track)
        outputs2 = model(audio2, visual2, track)
    
    # Outputs should be different (since audio/visual are used)
    outputs_differ = False
    for key in outputs1.keys():
        if not torch.allclose(outputs1[key], outputs2[key], atol=1e-5):
            outputs_differ = True
            break
    
    assert outputs_differ, \
        "Multimodal mode should produce different outputs when audio/visual differ"


def test_modality_mask_effect():
    """
    Test that modality mask correctly affects the output
    """
    batch_size = 2
    seq_len = 50
    
    # Create multimodal model
    model = create_multimodal_model(
        d_model=128,
        num_encoder_layers=2,
        enable_multimodal=True,
        fusion_type='gated'
    )
    
    # Create inputs
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    
    # Mask 1: All modalities available
    modality_mask1 = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    
    # Mask 2: Audio unavailable
    modality_mask2 = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    modality_mask2[:, :, 0] = False
    
    # Forward pass with different masks
    with torch.no_grad():
        outputs1 = model(audio, visual, track, modality_mask=modality_mask1)
        outputs2 = model(audio, visual, track, modality_mask=modality_mask2)
    
    # Outputs should be different (since audio is masked out in second case)
    outputs_differ = False
    for key in outputs1.keys():
        if not torch.allclose(outputs1[key], outputs2[key], atol=1e-5):
            outputs_differ = True
            break
    
    assert outputs_differ, \
        "Modality mask should affect the output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
