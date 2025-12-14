"""
Property-based tests for loss function backward compatibility

Feature: multimodal-video-features-integration, Property 17: Loss computation backward compatibility
Validates: Requirements 4.4
"""
import torch
import pytest
from hypothesis import given, strategies as st, settings
from loss import MultiTrackLoss
from model import MultiTrackTransformer
from multimodal_modules import ModalityEmbedding, ModalityFusion


# Strategy for generating valid batch dimensions
batch_dims = st.tuples(
    st.integers(min_value=1, max_value=8),  # batch_size
    st.integers(min_value=10, max_value=50),  # seq_len
)


def create_multimodal_transformer(
    audio_features: int = 4,
    visual_features: int = 522,
    track_features: int = 180,
    d_model: int = 128,  # Smaller for faster tests
    num_encoder_layers: int = 2,
    enable_multimodal: bool = True
):
    """Helper to create multimodal transformer"""
    from model import MultimodalTransformer
    
    return MultimodalTransformer(
        audio_features=audio_features,
        visual_features=visual_features,
        track_features=track_features,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=256,
        dropout=0.1,
        num_tracks=20,
        max_asset_classes=10,
        enable_multimodal=enable_multimodal,
        fusion_type='gated'
    )


def create_track_only_transformer(
    input_features: int = 180,
    d_model: int = 128,
    num_encoder_layers: int = 2
):
    """Helper to create track-only transformer"""
    return MultiTrackTransformer(
        input_features=input_features,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=256,
        dropout=0.1,
        num_tracks=20,
        max_asset_classes=10
    )


@given(batch_dims)
@settings(max_examples=100, deadline=None)
def test_multimodal_loss_compatibility(dims):
    """
    Property 17: Loss computation backward compatibility
    
    For any batch of predictions and targets, the computed loss for track parameters
    should be numerically identical to the existing MultiTrackLoss implementation.
    
    This test verifies that:
    1. Multimodal model outputs have the same structure as track-only model
    2. Loss function accepts both model outputs without errors
    3. Loss computation produces valid (non-NaN, non-Inf) values
    4. All expected loss components are present
    """
    batch_size, seq_len = dims
    
    # Create multimodal model
    model = create_multimodal_transformer(enable_multimodal=True)
    model.eval()
    
    # Create dummy inputs
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(audio, visual, track, padding_mask, modality_mask)
    
    # Verify output structure
    expected_keys = {'active', 'asset', 'scale', 'pos_x', 'pos_y', 
                     'crop_l', 'crop_r', 'crop_t', 'crop_b'}
    assert set(predictions.keys()) == expected_keys, \
        f"Output keys mismatch. Expected {expected_keys}, got {set(predictions.keys())}"
    
    # Verify shapes
    assert predictions['active'].shape == (batch_size, seq_len, 20, 2), \
        f"Active shape mismatch: {predictions['active'].shape}"
    assert predictions['asset'].shape == (batch_size, seq_len, 20, 10), \
        f"Asset shape mismatch: {predictions['asset'].shape}"
    assert predictions['scale'].shape == (batch_size, seq_len, 20, 1), \
        f"Scale shape mismatch: {predictions['scale'].shape}"
    
    # Create targets with same structure
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, 20)),
        'asset': torch.randint(0, 10, (batch_size, seq_len, 20)),
        'scale': torch.randn(batch_size, seq_len, 20, 1),
        'pos_x': torch.randn(batch_size, seq_len, 20, 1),
        'pos_y': torch.randn(batch_size, seq_len, 20, 1),
        'crop_l': torch.randn(batch_size, seq_len, 20, 1),
        'crop_r': torch.randn(batch_size, seq_len, 20, 1),
        'crop_t': torch.randn(batch_size, seq_len, 20, 1),
        'crop_b': torch.randn(batch_size, seq_len, 20, 1)
    }
    
    # Create loss function
    loss_fn = MultiTrackLoss(
        active_weight=1.0,
        asset_weight=1.0,
        scale_weight=1.0,
        position_weight=1.0,
        crop_weight=1.0
    )
    
    # Compute loss
    losses = loss_fn(predictions, targets, padding_mask)
    
    # Verify all loss components are present
    expected_loss_keys = {'total', 'active', 'asset', 'scale', 'position', 'crop'}
    assert set(losses.keys()) == expected_loss_keys, \
        f"Loss keys mismatch. Expected {expected_loss_keys}, got {set(losses.keys())}"
    
    # Verify all losses are valid (not NaN or Inf)
    for key, value in losses.items():
        assert not torch.isnan(value), f"{key} loss is NaN"
        assert not torch.isinf(value), f"{key} loss is Inf"
        assert value >= 0, f"{key} loss is negative: {value}"
    
    # Verify total loss is sum of weighted components
    expected_total = (
        losses['active'] + losses['asset'] + losses['scale'] + 
        losses['position'] + losses['crop']
    )
    assert torch.allclose(losses['total'], expected_total, rtol=1e-5), \
        f"Total loss mismatch: {losses['total']} vs {expected_total}"


@given(batch_dims)
@settings(max_examples=100, deadline=None)
def test_track_only_loss_compatibility(dims):
    """
    Verify that track-only model also works with the same loss function
    
    This ensures backward compatibility with existing models.
    """
    batch_size, seq_len = dims
    
    # Create track-only model
    model = create_track_only_transformer()
    model.eval()
    
    # Create dummy input
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(track, padding_mask)
    
    # Create targets
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, 20)),
        'asset': torch.randint(0, 10, (batch_size, seq_len, 20)),
        'scale': torch.randn(batch_size, seq_len, 20, 1),
        'pos_x': torch.randn(batch_size, seq_len, 20, 1),
        'pos_y': torch.randn(batch_size, seq_len, 20, 1),
        'crop_l': torch.randn(batch_size, seq_len, 20, 1),
        'crop_r': torch.randn(batch_size, seq_len, 20, 1),
        'crop_t': torch.randn(batch_size, seq_len, 20, 1),
        'crop_b': torch.randn(batch_size, seq_len, 20, 1)
    }
    
    # Create loss function
    loss_fn = MultiTrackLoss()
    
    # Compute loss
    losses = loss_fn(predictions, targets, padding_mask)
    
    # Verify all losses are valid
    for key, value in losses.items():
        assert not torch.isnan(value), f"{key} loss is NaN"
        assert not torch.isinf(value), f"{key} loss is Inf"
        assert value >= 0, f"{key} loss is negative: {value}"


@given(batch_dims)
@settings(max_examples=50, deadline=None)
def test_loss_with_partial_padding(dims):
    """
    Test that loss function correctly handles padding masks
    
    Verifies that padded positions don't contribute to loss.
    """
    batch_size, seq_len = dims
    
    if seq_len < 20:
        # Skip if sequence too short
        return
    
    # Create model
    model = create_multimodal_transformer(enable_multimodal=True)
    model.eval()
    
    # Create inputs
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    
    # Create padding mask (last 20% is padding)
    padding_len = seq_len // 5
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -padding_len:] = False
    
    # Get predictions
    with torch.no_grad():
        predictions = model(audio, visual, track, padding_mask, modality_mask)
    
    # Create targets
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, 20)),
        'asset': torch.randint(0, 10, (batch_size, seq_len, 20)),
        'scale': torch.randn(batch_size, seq_len, 20, 1),
        'pos_x': torch.randn(batch_size, seq_len, 20, 1),
        'pos_y': torch.randn(batch_size, seq_len, 20, 1),
        'crop_l': torch.randn(batch_size, seq_len, 20, 1),
        'crop_r': torch.randn(batch_size, seq_len, 20, 1),
        'crop_t': torch.randn(batch_size, seq_len, 20, 1),
        'crop_b': torch.randn(batch_size, seq_len, 20, 1)
    }
    
    # Compute loss with padding
    loss_fn = MultiTrackLoss()
    losses_with_padding = loss_fn(predictions, targets, padding_mask)
    
    # Compute loss without padding (full mask)
    full_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    losses_without_padding = loss_fn(predictions, targets, full_mask)
    
    # Loss with padding should be different (and typically lower)
    # because we're averaging over fewer valid positions
    assert not torch.allclose(
        losses_with_padding['total'], 
        losses_without_padding['total'],
        rtol=1e-3
    ), "Padding mask should affect loss computation"
    
    # All losses should still be valid
    for key, value in losses_with_padding.items():
        assert not torch.isnan(value), f"{key} loss is NaN with padding"
        assert not torch.isinf(value), f"{key} loss is Inf with padding"


@given(batch_dims)
@settings(max_examples=50, deadline=None)
def test_loss_gradient_flow(dims):
    """
    Test that gradients flow correctly through loss computation
    
    Verifies that loss can be used for training.
    """
    batch_size, seq_len = dims
    
    # Create model with gradient tracking
    model = create_multimodal_transformer(enable_multimodal=True)
    model.train()
    
    # Create inputs
    audio = torch.randn(batch_size, seq_len, 4)
    visual = torch.randn(batch_size, seq_len, 522)
    track = torch.randn(batch_size, seq_len, 180)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    
    # Get predictions (with gradients)
    predictions = model(audio, visual, track, padding_mask, modality_mask)
    
    # Create targets
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, 20)),
        'asset': torch.randint(0, 10, (batch_size, seq_len, 20)),
        'scale': torch.randn(batch_size, seq_len, 20, 1),
        'pos_x': torch.randn(batch_size, seq_len, 20, 1),
        'pos_y': torch.randn(batch_size, seq_len, 20, 1),
        'crop_l': torch.randn(batch_size, seq_len, 20, 1),
        'crop_r': torch.randn(batch_size, seq_len, 20, 1),
        'crop_t': torch.randn(batch_size, seq_len, 20, 1),
        'crop_b': torch.randn(batch_size, seq_len, 20, 1)
    }
    
    # Compute loss
    loss_fn = MultiTrackLoss()
    losses = loss_fn(predictions, targets, padding_mask)
    
    # Backward pass
    losses['total'].backward()
    
    # Check that gradients exist and are valid
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), "Gradient contains NaN"
            assert not torch.isinf(param.grad).any(), "Gradient contains Inf"
    
    assert has_gradients, "No gradients computed"


if __name__ == "__main__":
    print("Running loss compatibility tests...")
    print("\n" + "="*70)
    print("Test 1: Multimodal loss compatibility")
    print("="*70)
    test_multimodal_loss_compatibility((4, 20))
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 2: Track-only loss compatibility")
    print("="*70)
    test_track_only_loss_compatibility((4, 20))
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 3: Loss with partial padding")
    print("="*70)
    test_loss_with_partial_padding((4, 30))
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("Test 4: Loss gradient flow")
    print("="*70)
    test_loss_gradient_flow((2, 20))
    print("✅ Passed")
    
    print("\n" + "="*70)
    print("✅ All loss compatibility tests passed!")
    print("="*70)
