"""
Property-based tests for Multi-Track Transformer model
"""
import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from model import MultiTrackTransformer, create_model


# Test Property 11: Model Output Structure Completeness
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=20, max_value=150),
    d_model=st.integers(min_value=64, max_value=256).filter(lambda x: x % 8 == 0),
    num_tracks=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=50, deadline=None)
def test_property_11_model_output_structure_completeness(batch_size, seq_len, d_model, num_tracks):
    """
    Property 11: Model Output Structure Completeness
    
    Model must output predictions for all 9 parameters for all tracks:
    1. Output dict contains all 9 parameter keys
    2. Each output has correct shape (batch, seq_len, num_tracks, output_dim)
    3. Classification outputs have correct number of classes
    4. Regression outputs have dimension 1
    5. All outputs are valid tensors (no NaN, no Inf)
    """
    # Create model
    model = MultiTrackTransformer(
        input_features=180,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=2,  # Smaller for faster testing
        dim_feedforward=d_model * 2,
        dropout=0.1,
        num_tracks=num_tracks,
        max_asset_classes=10
    )
    model.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, 180)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, mask)
    
    # Property 1: All 9 parameter keys present
    expected_keys = {'active', 'asset', 'scale', 'pos_x', 'pos_y', 'crop_l', 'crop_r', 'crop_t', 'crop_b'}
    assert set(outputs.keys()) == expected_keys, f"Missing keys: {expected_keys - set(outputs.keys())}"
    
    # Property 2: Correct shapes
    for key in expected_keys:
        assert outputs[key].dim() == 4, f"{key} should have 4 dimensions"
        assert outputs[key].shape[0] == batch_size, f"{key} batch size mismatch"
        assert outputs[key].shape[1] == seq_len, f"{key} sequence length mismatch"
        assert outputs[key].shape[2] == num_tracks, f"{key} num_tracks mismatch"
    
    # Property 3: Classification outputs have correct number of classes
    assert outputs['active'].shape[3] == 2, "active should have 2 classes"
    assert outputs['asset'].shape[3] == 10, "asset should have 10 classes"
    
    # Property 4: Regression outputs have dimension 1
    regression_keys = ['scale', 'pos_x', 'pos_y', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
    for key in regression_keys:
        assert outputs[key].shape[3] == 1, f"{key} should have output dimension 1"
    
    # Property 5: No NaN or Inf
    for key, value in outputs.items():
        assert not torch.isnan(value).any(), f"{key} contains NaN"
        assert not torch.isinf(value).any(), f"{key} contains Inf"


# Test Property 20: Logical Track Activation Consistency
@given(
    batch_size=st.integers(min_value=2, max_value=8),
    seq_len=st.integers(min_value=50, max_value=100)
)
@settings(max_examples=50, deadline=None)
def test_property_20_logical_track_activation_consistency(batch_size, seq_len):
    """
    Property 20: Logical Track Activation Consistency
    
    Track activation predictions should be logically consistent:
    1. Active logits should produce valid probabilities after softmax
    2. Probabilities should sum to 1 for each track
    3. Model should handle masked (padded) positions correctly
    4. Predictions for padded positions should still be valid (not NaN)
    """
    # Create model
    model = create_model(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256
    )
    model.eval()
    
    # Create input with some padding
    x = torch.randn(batch_size, seq_len, 180)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Add padding to last 20% of sequence
    padding_start = int(seq_len * 0.8)
    mask[:, padding_start:] = False
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, mask)
    
    active_logits = outputs['active']  # (batch, seq_len, num_tracks, 2)
    
    # Property 1: Valid probabilities after softmax
    active_probs = torch.softmax(active_logits, dim=-1)
    assert not torch.isnan(active_probs).any(), "Active probabilities contain NaN"
    assert not torch.isinf(active_probs).any(), "Active probabilities contain Inf"
    assert (active_probs >= 0).all(), "Probabilities should be non-negative"
    assert (active_probs <= 1).all(), "Probabilities should be <= 1"
    
    # Property 2: Probabilities sum to 1
    prob_sums = active_probs.sum(dim=-1)  # (batch, seq_len, num_tracks)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "Probabilities should sum to 1"
    
    # Property 3 & 4: Predictions for padded positions are still valid
    padded_logits = active_logits[:, padding_start:, :, :]
    assert not torch.isnan(padded_logits).any(), "Padded positions contain NaN"
    assert not torch.isinf(padded_logits).any(), "Padded positions contain Inf"


# Test model parameter count
def test_model_parameter_count():
    """Test that model has reasonable number of parameters"""
    model = create_model(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024
    )
    
    num_params = model.count_parameters()
    
    # Should have millions of parameters but not too many
    assert num_params > 1_000_000, "Model should have at least 1M parameters"
    assert num_params < 100_000_000, "Model should have less than 100M parameters"


# Test model with different input sizes
@pytest.mark.parametrize("input_features,num_tracks", [
    (180, 20),  # Standard: 20 tracks × 9 params
    (90, 10),   # Half: 10 tracks × 9 params
    (270, 30),  # More: 30 tracks × 9 params
])
def test_model_different_input_sizes(input_features, num_tracks):
    """Test model with different input feature sizes"""
    model = MultiTrackTransformer(
        input_features=input_features,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        num_tracks=num_tracks
    )
    model.eval()
    
    batch_size = 2
    seq_len = 50
    
    x = torch.randn(batch_size, seq_len, input_features)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    with torch.no_grad():
        outputs = model(x, mask)
    
    # Check output shapes
    assert outputs['active'].shape == (batch_size, seq_len, num_tracks, 2)
    assert outputs['scale'].shape == (batch_size, seq_len, num_tracks, 1)


# Test model without mask
def test_model_without_mask():
    """Test that model works without providing a mask"""
    model = create_model(d_model=128, nhead=4, num_encoder_layers=2)
    model.eval()
    
    batch_size = 2
    seq_len = 50
    
    x = torch.randn(batch_size, seq_len, 180)
    
    # Forward pass without mask
    with torch.no_grad():
        outputs = model(x, mask=None)
    
    # Should still produce valid outputs
    assert outputs['active'].shape == (batch_size, seq_len, 20, 2)
    assert not torch.isnan(outputs['active']).any()


# Test gradient flow
def test_gradient_flow():
    """Test that gradients flow through the model"""
    model = create_model(d_model=128, nhead=4, num_encoder_layers=2)
    model.train()
    
    batch_size = 2
    seq_len = 50
    
    x = torch.randn(batch_size, seq_len, 180, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    outputs = model(x, mask)
    
    # Compute dummy loss using all outputs to ensure all parameters get gradients
    loss = sum(output.sum() for output in outputs.values())
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Input should have gradients"
    
    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} should have gradients"


# Test model save and load
def test_model_save_load():
    """Test that model can be saved and loaded"""
    import tempfile
    import os
    
    # Create model
    model1 = create_model(d_model=128, nhead=4, num_encoder_layers=2)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model1.state_dict(), save_path)
        
        # Load model
        model2 = create_model(d_model=128, nhead=4, num_encoder_layers=2)
        model2.load_state_dict(torch.load(save_path))
    
    # Compare outputs
    model1.eval()
    model2.eval()
    
    x = torch.randn(2, 50, 180)
    mask = torch.ones(2, 50, dtype=torch.bool)
    
    with torch.no_grad():
        outputs1 = model1(x, mask)
        outputs2 = model2(x, mask)
    
    # Outputs should be identical
    for key in outputs1.keys():
        assert torch.allclose(outputs1[key], outputs2[key], atol=1e-6), \
            f"Outputs for {key} don't match after save/load"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
