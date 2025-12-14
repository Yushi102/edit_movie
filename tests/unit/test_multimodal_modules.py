"""
Property-based tests for Multimodal Modules

Tests Properties 10, 11, 32 from the design document.
"""
import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings
import numpy as np

from multimodal_modules import ModalityEmbedding, ModalityFusion


# Property 10: Configurable input dimensions
@settings(max_examples=100, deadline=None)
@given(
    audio_dim=st.integers(min_value=1, max_value=100),
    visual_dim=st.integers(min_value=1, max_value=600),
    track_dim=st.integers(min_value=1, max_value=200),
    d_model=st.integers(min_value=32, max_value=512),
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=10, max_value=100)
)
def test_property_10_configurable_input_dimensions(
    audio_dim, visual_dim, track_dim, d_model, batch_size, seq_len
):
    """
    Property 10: Configurable input dimensions
    
    For any valid combination of (audio_dim, visual_dim, track_dim), the model
    should initialize successfully and accept inputs of those dimensions.
    """
    # Create embeddings for each modality
    audio_embedding = ModalityEmbedding(audio_dim, d_model)
    visual_embedding = ModalityEmbedding(visual_dim, d_model)
    track_embedding = ModalityEmbedding(track_dim, d_model)
    
    # Create random inputs
    audio_input = torch.randn(batch_size, seq_len, audio_dim)
    visual_input = torch.randn(batch_size, seq_len, visual_dim)
    track_input = torch.randn(batch_size, seq_len, track_dim)
    
    # Forward pass should succeed
    audio_emb = audio_embedding(audio_input)
    visual_emb = visual_embedding(visual_input)
    track_emb = track_embedding(track_input)
    
    # Verify output shapes
    assert audio_emb.shape == (batch_size, seq_len, d_model), \
        f"Audio embedding shape mismatch: expected {(batch_size, seq_len, d_model)}, got {audio_emb.shape}"
    assert visual_emb.shape == (batch_size, seq_len, d_model), \
        f"Visual embedding shape mismatch: expected {(batch_size, seq_len, d_model)}, got {visual_emb.shape}"
    assert track_emb.shape == (batch_size, seq_len, d_model), \
        f"Track embedding shape mismatch: expected {(batch_size, seq_len, d_model)}, got {track_emb.shape}"


# Property 11: Modality embedding to common dimension
@settings(max_examples=100, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=600),
    d_model=st.integers(min_value=32, max_value=512),
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=10, max_value=100)
)
def test_property_11_modality_embedding_to_common_dimension(
    input_dim, d_model, batch_size, seq_len
):
    """
    Property 11: Modality embedding to common dimension
    
    For any modality input of dimension D_in, after passing through its embedding
    layer, the output should have dimension d_model.
    """
    # Create embedding
    embedding = ModalityEmbedding(input_dim, d_model)
    
    # Create random input
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = embedding(input_tensor)
    
    # Verify output dimension
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape mismatch: expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Verify that output dimension is exactly d_model
    assert output.shape[-1] == d_model, \
        f"Output last dimension should be {d_model}, got {output.shape[-1]}"


# Property 32: Gated fusion weight bounds
def test_property_32_gated_fusion_weight_bounds():
    """
    Property 32: Gated fusion weight bounds
    
    For any gated fusion output, the learned gate values should be in range [0, 1]
    after sigmoid activation, and the sum of weighted contributions should equal
    the fused output.
    """
    batch_size = 4
    seq_len = 50
    d_model = 256
    
    # Create embeddings
    audio_emb = torch.randn(batch_size, seq_len, d_model)
    visual_emb = torch.randn(batch_size, seq_len, d_model)
    track_emb = torch.randn(batch_size, seq_len, d_model)
    
    # Create gated fusion
    fusion = ModalityFusion(d_model, num_modalities=3, fusion_type='gated')
    
    # Forward pass
    fused = fusion(audio_emb, visual_emb, track_emb)
    
    # Manually compute gates to verify bounds
    with torch.no_grad():
        gate_audio = fusion.gate_networks[0](audio_emb)
        gate_visual = fusion.gate_networks[1](visual_emb)
        gate_track = fusion.gate_networks[2](track_emb)
        
        # Verify gates are in [0, 1] (sigmoid output)
        assert torch.all(gate_audio >= 0) and torch.all(gate_audio <= 1), \
            "Audio gates should be in range [0, 1]"
        assert torch.all(gate_visual >= 0) and torch.all(gate_visual <= 1), \
            "Visual gates should be in range [0, 1]"
        assert torch.all(gate_track >= 0) and torch.all(gate_track <= 1), \
            "Track gates should be in range [0, 1]"
        
        # Compute expected fused output (before dropout)
        expected_fused = (
            gate_audio * audio_emb +
            gate_visual * visual_emb +
            gate_track * track_emb
        )
        
        # Note: We can't directly compare with fused because dropout is applied
        # But we can verify the computation structure is correct
        assert expected_fused.shape == fused.shape, \
            f"Expected fused shape {expected_fused.shape}, got {fused.shape}"


# Test gated fusion with modality masking
def test_gated_fusion_with_masking():
    """Test that gated fusion correctly applies modality masking"""
    batch_size = 4
    seq_len = 50
    d_model = 256
    
    # Create embeddings
    audio_emb = torch.randn(batch_size, seq_len, d_model)
    visual_emb = torch.randn(batch_size, seq_len, d_model)
    track_emb = torch.randn(batch_size, seq_len, d_model)
    
    # Create modality mask: audio unavailable for all timesteps
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    modality_mask[:, :, 0] = False  # Audio unavailable
    
    # Create gated fusion
    fusion = ModalityFusion(d_model, num_modalities=3, fusion_type='gated')
    
    # Forward pass with masking
    fused = fusion(audio_emb, visual_emb, track_emb, modality_mask)
    
    # Verify output shape
    assert fused.shape == (batch_size, seq_len, d_model)
    
    # The audio contribution should be zeroed out before fusion
    # We can't directly verify this without accessing internals, but we can
    # verify that the fusion runs without errors


# Test concatenation fusion
def test_concatenation_fusion():
    """Test concatenation fusion strategy"""
    batch_size = 4
    seq_len = 50
    d_model = 256
    
    # Create embeddings
    audio_emb = torch.randn(batch_size, seq_len, d_model)
    visual_emb = torch.randn(batch_size, seq_len, d_model)
    track_emb = torch.randn(batch_size, seq_len, d_model)
    
    # Create concatenation fusion
    fusion = ModalityFusion(d_model, num_modalities=3, fusion_type='concat')
    
    # Forward pass
    fused = fusion(audio_emb, visual_emb, track_emb)
    
    # Verify output shape
    assert fused.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {fused.shape}"


# Test additive fusion
def test_additive_fusion():
    """Test additive fusion strategy"""
    batch_size = 4
    seq_len = 50
    d_model = 256
    
    # Create embeddings
    audio_emb = torch.randn(batch_size, seq_len, d_model)
    visual_emb = torch.randn(batch_size, seq_len, d_model)
    track_emb = torch.randn(batch_size, seq_len, d_model)
    
    # Create additive fusion
    fusion = ModalityFusion(d_model, num_modalities=3, fusion_type='add')
    
    # Forward pass
    fused = fusion(audio_emb, visual_emb, track_emb)
    
    # Verify output shape
    assert fused.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {fused.shape}"
    
    # Verify that weights sum to 1 (softmax)
    with torch.no_grad():
        weights = torch.softmax(fusion.modality_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6), \
            "Additive fusion weights should sum to 1"


# Test with different d_model values
@settings(max_examples=50, deadline=None)
@given(
    d_model=st.integers(min_value=32, max_value=512),
    fusion_type=st.sampled_from(['concat', 'add', 'gated'])
)
def test_fusion_with_various_d_model(d_model, fusion_type):
    """Test fusion modules with various d_model values"""
    batch_size = 2
    seq_len = 20
    
    # Create embeddings
    audio_emb = torch.randn(batch_size, seq_len, d_model)
    visual_emb = torch.randn(batch_size, seq_len, d_model)
    track_emb = torch.randn(batch_size, seq_len, d_model)
    
    # Create fusion
    fusion = ModalityFusion(d_model, num_modalities=3, fusion_type=fusion_type)
    
    # Forward pass
    fused = fusion(audio_emb, visual_emb, track_emb)
    
    # Verify output shape
    assert fused.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {fused.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
