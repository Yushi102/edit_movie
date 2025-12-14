# Design Document

## Overview

This design extends the existing Multi-Track Transformer to incorporate video content features (audio and visual) alongside editing track data. The system will implement a multimodal architecture that fuses temporal audio features, high-dimensional visual features (including CLIP embeddings), and editing patterns to enable content-aware video editing prediction.

The architecture follows a modality-specific embedding approach where each input modality (audio, visual, track) is processed through dedicated projection layers before fusion. This design maintains backward compatibility with the existing track-only model while enabling richer content-aware predictions.

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Loading Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Audio CSV    │  │ Visual CSV   │  │ Track NPZ    │      │
│  │ Loader       │  │ Loader       │  │ Loader       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Alignment Layer                      │
│         (Timestamp matching + Interpolation)                 │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing & Normalization                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Audio        │  │ Visual       │  │ Track        │      │
│  │ Normalizer   │  │ Normalizer   │  │ Normalizer   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Multimodal Transformer Model                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Audio        │  │ Visual       │  │ Track        │      │
│  │ Embedding    │  │ Embedding    │  │ Embedding    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │ Modality Fusion │                        │
│                   └────────┬────────┘                        │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │ Positional      │                        │
│                   │ Encoding        │                        │
│                   └────────┬────────┘                        │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │ Transformer     │                        │
│                   │ Encoder Layers  │                        │
│                   └────────┬────────┘                        │
│                            ▼                                 │
│                   ┌─────────────────┐                        │
│                   │ Track-Specific  │                        │
│                   │ Output Heads    │                        │
│                   └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Component Descriptions

1. **Data Loading Layer**: Loads audio features (_features.csv), visual features (_visual_features.csv), and track data (sequences.npz) for each video
2. **Feature Alignment Layer**: Synchronizes timestamps across modalities using interpolation
3. **Preprocessing Layer**: Applies modality-specific normalization
4. **Multimodal Transformer**: Processes fused features through transformer encoder
5. **Output Heads**: Predicts 9 parameters for each of 20 tracks

## Components and Interfaces

### 1. MultimodalDataset

```python
class MultimodalDataset(Dataset):
    def __init__(
        self,
        sequences_npz: str,
        features_dir: str,
        audio_scaler: Optional[StandardScaler] = None,
        visual_scaler: Optional[StandardScaler] = None,
        track_scaler: Optional[StandardScaler] = None,
        enable_multimodal: bool = True
    )
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Returns: {
        #   'audio': (seq_len, audio_dim),
        #   'visual': (seq_len, visual_dim),
        #   'track': (seq_len, track_dim),
        #   'targets': (seq_len, num_tracks, 9),
        #   'mask': (seq_len,)
        # }
```

**Responsibilities:**
- Load and align multimodal features
- Apply normalization
- Handle missing features gracefully
- Support fallback to track-only mode

### 2. FeatureAligner

```python
class FeatureAligner:
    def __init__(self, tolerance: float = 0.05)
    
    def align_features(
        self,
        track_times: np.ndarray,
        audio_df: pd.DataFrame,
        visual_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns: (aligned_audio, aligned_visual, modality_mask)
        # modality_mask: (seq_len, 3) boolean indicating availability of [audio, visual, track]
```

**Responsibilities:**
- Match timestamps with tolerance
- Interpolate continuous numerical features (RMS, motion, etc.) using linear interpolation
- Forward-fill discrete/binary features (is_speaking, text_active, face_count)
- Handle missing modalities with zero-filling and mask generation
- Validate temporal consistency

### 3. MultimodalTransformer

```python
class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        audio_features: int = 5,  # Updated: 5 numerical features only
        visual_features: int = 522,
        track_features: int = 180,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_tracks: int = 20,
        max_asset_classes: int = 10,
        enable_multimodal: bool = True,
        fusion_type: str = 'concat',  # 'concat', 'add', 'gated', 'attention'
        use_modality_attention_mask: bool = True  # Mask unavailable modalities
    )
    
    def forward(
        self,
        audio: torch.Tensor,  # (batch, seq_len, audio_dim)
        visual: torch.Tensor,  # (batch, seq_len, visual_dim)
        track: torch.Tensor,  # (batch, seq_len, track_dim)
        padding_mask: Optional[torch.Tensor] = None,  # (batch, seq_len) for sequence padding
        modality_mask: Optional[torch.Tensor] = None  # (batch, seq_len, 3) for modality availability
    ) -> Dict[str, torch.Tensor]
```

**Responsibilities:**
- Embed each modality to d_model dimension with modality-specific projection layers
- Apply modality attention masking to ignore unavailable modalities
- Fuse modalities using specified strategy (with gated fusion for imbalanced information density)
- Process through transformer encoder with proper masking
- Generate track-specific predictions

### 4. ModalityEmbedding

```python
class ModalityEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1
    )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projects input_dim -> d_model
```

**Responsibilities:**
- Project modality-specific features to common dimension
- Apply dropout for regularization
- Learn modality-specific representations

### 5. ModalityFusion

```python
class ModalityFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_modalities: int = 3,
        fusion_type: str = 'concat'
    )
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor,
        track_emb: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None  # (batch, seq_len, 3)
    ) -> torch.Tensor
```

**Fusion Strategies:**
- **Concatenation**: `[audio; visual; track]` → Linear(3*d_model, d_model)
- **Addition**: `audio + visual + track` with learned weights
- **Gated Fusion**: `gate_a * audio + gate_v * visual + gate_t * track` where gates are learned from input
- **Attention**: Cross-modal attention before summation

**Modality Masking:**
When modality_mask is provided, unavailable modalities are zeroed out before fusion to prevent noise from zero-filled features.

## Data Models

### Audio Features (5 dimensions - numerical only)
```python
{
    'time': float,  # Timestamp in seconds
    'audio_energy_rms': float,  # RMS energy (normalized)
    'audio_is_speaking': int,  # Binary: 0 or 1
    'silence_duration_ms': float,  # Milliseconds of silence
    'text_is_active': int,  # Binary: 0 or 1
    # Note: speaker_id and text_word are excluded from model input
    # They are categorical/string data unsuitable for interpolation
    # and would require separate embedding layers
}
```

### Visual Features (522 dimensions)
```python
{
    'time': float,  # Timestamp in seconds
    'scene_change': float,  # Scene change score [0, 1]
    'visual_motion': float,  # Motion magnitude
    'saliency_x': float,  # X coordinate of visual saliency
    'saliency_y': float,  # Y coordinate of visual saliency
    'face_count': int,  # Number of detected faces
    'face_center_x': float,  # Face center X (empty if no face)
    'face_center_y': float,  # Face center Y (empty if no face)
    'face_size': float,  # Face bounding box size
    'face_mouth_open': float,  # Mouth openness score
    'face_eyebrow_raise': float,  # Eyebrow raise score
    'clip_0' to 'clip_511': float  # CLIP embeddings (512-dim)
}
```

### Track Features (180 dimensions)
```python
# 20 tracks × 9 parameters = 180 dimensions
# Per track: [active, asset_id, scale, x, y, crop_l, crop_r, crop_t, crop_b]
```

### Aligned Multimodal Features
```python
{
    'audio': np.ndarray,  # (seq_len, 5) - numerical features only
    'visual': np.ndarray,  # (seq_len, 522)
    'track': np.ndarray,  # (seq_len, 180)
    'targets': np.ndarray,  # (seq_len, 20, 9)
    'padding_mask': np.ndarray,  # (seq_len,) boolean - True for valid, False for padding
    'modality_mask': np.ndarray,  # (seq_len, 3) boolean - availability of [audio, visual, track]
    'video_id': str,
    'timestamps': np.ndarray  # (seq_len,)
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Data Loading and Alignment Properties

**Property 1: Feature file loading completeness**
*For any* video with existing feature CSV files, loading should successfully return both audio and visual feature dataframes with non-zero rows
**Validates: Requirements 1.1**

**Property 2: Timestamp alignment tolerance**
*For any* pair of timestamps from track data and video features, if their absolute difference is ≤ 0.05 seconds, they should be considered aligned
**Validates: Requirements 1.2**

**Property 3: Interpolation correctness by feature type**
*For any* three consecutive timestamps t1 < t2 < t3 where t2 is missing:
- Continuous features (RMS, motion): v2 = v1 + (v3 - v1) * (t2 - t1) / (t3 - t1) (linear)
- Binary features (is_speaking): v2 = v1 (forward-fill)
- CLIP embeddings: Apply linear interpolation then L2 renormalize
**Validates: Requirements 1.3**

**Property 4: Forward-fill consistency**
*For any* sequence of timestamps with missing features, forward-filled values should equal the last known non-missing value
**Validates: Requirements 1.4**

**Property 5: Modality concatenation structure with masking**
*For any* aligned audio (A), visual (V), and track (T) features with modality_mask (M), the output should include:
- Concatenated features with shape (seq_len, dim_A + dim_V + dim_T) preserving order [A, V, T]
- Modality mask with shape (seq_len, 3) indicating availability of each modality
**Validates: Requirements 1.5**

### Preprocessing and Normalization Properties

**Property 6: Normalization round-trip consistency**
*For any* audio features, after normalization and storing parameters, applying the inverse transformation using stored (mean, std) should recover the original values within numerical precision
**Validates: Requirements 2.1, 2.5**

**Property 7: Independent normalization**
*For any* visual features, normalizing motion values should not change the mean and std of saliency values, and vice versa
**Validates: Requirements 2.2**

**Property 8: L2 normalization unit length**
*For any* CLIP embedding vector after L2 normalization, the L2 norm should equal 1.0 within numerical tolerance (1e-6)
**Validates: Requirements 2.3**

**Property 9: Missing face data zero-filling**
*For any* input with missing face detection data (face_count = 0), all face-related features (center_x, center_y, size, mouth_open, eyebrow_raise) should be filled with zeros
**Validates: Requirements 2.4**

### Model Architecture Properties

**Property 10: Configurable input dimensions**
*For any* valid combination of (audio_dim, visual_dim, track_dim), the model should initialize successfully and accept inputs of those dimensions
**Validates: Requirements 3.1**

**Property 11: Modality embedding to common dimension**
*For any* modality input of dimension D_in, after passing through its embedding layer, the output should have dimension d_model
**Validates: Requirements 3.2, 3.3**

**Property 12: Additive fusion with learned weights**
*For any* set of modality embeddings (audio_emb, visual_emb, track_emb), the fused output should equal: w_a * audio_emb + w_v * visual_emb + w_t * track_emb, where w_* are learnable parameters
**Validates: Requirements 3.4**

**Property 13: Cross-modal attention application**
*For any* model with cross-modal attention enabled, the attention mechanism should be applied before positional encoding, and attention weights should sum to 1.0 across the key dimension
**Validates: Requirements 3.5**

### Training Data Loading Properties

**Property 14: Video name matching**
*For any* video_id in the training data, the system should search for feature files matching the pattern "{video_id}_features.csv" and "{video_id}_visual_features.csv" in the input_features directory
**Validates: Requirements 4.1**

**Property 15: Missing feature handling**
*For any* video without feature files, the video should be excluded from the dataset and a warning should be logged containing the video_id
**Validates: Requirements 4.2**

**Property 16: Batch sequence length consistency**
*For any* batch of sequences, all modalities (audio, visual, track) should have identical sequence length dimensions: audio.shape[1] == visual.shape[1] == track.shape[1]
**Validates: Requirements 4.3**

**Property 17: Loss computation backward compatibility**
*For any* batch of predictions and targets, the computed loss for track parameters should be numerically identical to the existing MultiTrackLoss implementation
**Validates: Requirements 4.4**

### Backward Compatibility Properties

**Property 18: Graceful fallback to track-only mode**
*For any* dataset where video features are unavailable, the system should successfully train using only track data with enable_multimodal=False
**Validates: Requirements 5.1**

**Property 19: Multimodal flag respect**
*For any* model configuration, setting enable_multimodal=False should result in the model ignoring audio and visual inputs and processing only track data
**Validates: Requirements 5.2**

**Property 20: Dual-mode inference support**
*For any* trained model, inference should succeed in both multimodal mode (with all features) and unimodal mode (track-only) without errors
**Validates: Requirements 5.3**

**Property 21: Checkpoint type detection**
*For any* saved checkpoint, loading should correctly detect whether it's a multimodal or unimodal model based on the stored architecture configuration
**Validates: Requirements 5.4**

### Logging and Validation Properties

**Property 22: Feature loading logging**
*For any* data loading operation, the log should contain an entry with the count of successfully loaded feature files
**Validates: Requirements 6.1**

**Property 23: Alignment failure logging**
*For any* alignment failure, the log should contain the video_id and the timestamp ranges that failed to align
**Validates: Requirements 6.2**

**Property 24: Interpolation percentage logging**
*For any* alignment operation that performs interpolation, the log should report the percentage of interpolated values relative to total values
**Validates: Requirements 6.3**

**Property 25: Missing feature type logging**
*For any* video with missing features, the log should explicitly list which feature types (audio, visual, or both) are unavailable
**Validates: Requirements 6.4**

**Property 26: Monotonic timestamp ordering**
*For any* aligned feature sequence, timestamps should be strictly monotonically increasing: t[i] < t[i+1] for all valid indices i
**Validates: Requirements 7.1**

**Property 27: Coverage percentage computation**
*For any* dataset, the coverage percentage should equal (number of track timestamps with matching features) / (total track timestamps) * 100
**Validates: Requirements 7.2**

**Property 28: Gap detection threshold**
*For any* sequence of timestamps, gaps larger than 1.0 seconds should be identified and reported
**Validates: Requirements 7.3**

**Property 29: Feature dimension validation**
*For any* loaded feature vector, the dimensions should match expected values: audio=5 (numerical only), visual=522, track=180
**Validates: Requirements 7.4**

**Property 30: Interpolation bounds validation**
*For any* interpolated continuous value v at timestamp t between t1 and t2, the value should satisfy: min(v1, v2) ≤ v ≤ max(v1, v2)
**Validates: Requirements 7.5**

**Property 31: Modality mask consistency**
*For any* batch of data, if modality_mask[i, j, k] = False, then the corresponding feature values should be zero-filled and not contribute to attention weights
**Validates: Requirements 1.4, 2.4**

**Property 32: Gated fusion weight bounds**
*For any* gated fusion output, the learned gate values should be in range [0, 1] after sigmoid activation, and the sum of weighted contributions should equal the fused output
**Validates: Requirements 3.4**

## Error Handling

### Feature Loading Errors

1. **Missing Feature Files**: Log warning, skip video, continue with remaining videos
2. **Corrupted CSV Files**: Log error with file path, skip video, continue
3. **Dimension Mismatch**: Log error with expected vs actual dimensions, raise exception
4. **Empty Feature Files**: Log warning, treat as missing features

### Alignment Errors

1. **No Overlapping Timestamps**: Log error with timestamp ranges, skip video
2. **Excessive Interpolation** (>50% of values): Log warning, continue but flag for review
3. **Timestamp Ordering Violation**: Log error, attempt to sort, raise exception if fails
4. **Large Gaps** (>5 seconds): Log warning, use forward-fill, flag sequence

### Model Errors

1. **Dimension Mismatch in Forward Pass**: Raise ValueError with detailed shape information
2. **NaN in Embeddings**: Log error, apply gradient clipping, continue training
3. **Memory Overflow**: Reduce batch size automatically, log adjustment
4. **Checkpoint Loading Failure**: Log error, provide fallback to random initialization

### Training Errors

1. **All Videos Missing Features**: Raise RuntimeError, cannot proceed
2. **Batch Size Mismatch**: Raise ValueError with batch composition details
3. **Loss Computation NaN**: Log error, skip batch, continue training
4. **Validation Set Empty**: Log warning, skip validation for that epoch

## Testing Strategy

### Unit Testing

Unit tests will verify specific examples and edge cases:

1. **Feature Loading**: Test loading valid CSVs, handling missing files, corrupted data
2. **Timestamp Alignment**: Test exact matches, tolerance boundaries, interpolation edge cases
3. **Normalization**: Test zero mean/unit variance, L2 norm, zero-filling
4. **Model Initialization**: Test various dimension configurations, flag settings
5. **Fusion Mechanisms**: Test concatenation, addition, attention with known inputs

### Property-Based Testing

Property-based tests will use Hypothesis library (minimum 100 iterations per property) to verify universal properties:

1. **Data Loading Properties** (Properties 1-5): Generate random timestamps, feature values, test alignment and concatenation
2. **Normalization Properties** (Properties 6-9): Generate random feature distributions, verify statistical properties
3. **Model Architecture Properties** (Properties 10-13): Generate random input dimensions and tensors, verify output shapes and fusion
4. **Training Properties** (Properties 14-17): Generate random datasets with varying feature availability
5. **Compatibility Properties** (Properties 18-21): Test mode switching with random configurations
6. **Logging Properties** (Properties 22-25): Verify log entries with various failure scenarios
7. **Validation Properties** (Properties 26-30): Generate random sequences, verify constraints

### Integration Testing

Integration tests will verify end-to-end workflows:

1. **Full Pipeline Test**: Load real feature files, align, preprocess, train for 1 epoch
2. **Fallback Mode Test**: Train with and without features, compare outputs
3. **Checkpoint Compatibility Test**: Save multimodal model, load in unimodal mode
4. **Batch Processing Test**: Process multiple videos with mixed feature availability

### Performance Testing

Performance tests will ensure scalability:

1. **Large Batch Processing**: Test with batch_size=32, seq_len=200
2. **High-Dimensional Features**: Verify memory usage with full CLIP embeddings
3. **Long Sequences**: Test with sequences up to 500 frames
4. **Many Videos**: Test dataset with 100+ videos

## Implementation Notes

### Feature Dimension Summary

- Audio features: 5 dimensions (RMS, is_speaking, silence_duration, text_is_active) - numerical only
  - Excluded: speaker_id (categorical), text_word (string) - unsuitable for interpolation
- Visual features: 522 dimensions (11 scalar + 512 CLIP)
  - Stored as float16 to reduce memory usage
- Track features: 180 dimensions (20 tracks × 9 parameters)
- Total input: 707 dimensions (when concatenated)

### Interpolation Strategy by Feature Type

| Feature Type | Interpolation Method | Rationale |
|--------------|---------------------|-----------|
| Continuous (RMS, motion, saliency) | Linear | Smooth transitions between values |
| Binary (is_speaking, text_active) | Forward-fill | Discrete states don't interpolate |
| Count (face_count) | Nearest neighbor | Integer values |
| CLIP embeddings | Linear + L2 renorm | Preserve semantic space |
| Categorical (speaker_id) | Excluded | Cannot interpolate meaningfully |

### Fusion Strategy Comparison

| Strategy | Pros | Cons | Parameters | Use Case |
|----------|------|------|------------|----------|
| Concatenation | Simple, preserves all info | High dimensional | Linear(3*d_model, d_model) | Baseline |
| Addition | Low parameters, efficient | May lose modality-specific info | 3 scalar weights | Fast experiments |
| Gated Fusion | Balances imbalanced modalities | Moderate complexity | 3 gating networks | **Recommended for imbalanced info density** |
| Attention | Learns importance dynamically | Most complex, slower | Multi-head attention layers | If gated fusion insufficient |

**Recommendation**: Start with **Gated Fusion** to address the information density imbalance (Audio: 5-dim vs Visual: 522-dim). The gating mechanism will learn to weight modalities appropriately, preventing visual features from dominating.

**Gated Fusion Formula:**
```
gate_a = sigmoid(W_a * audio_emb + b_a)
gate_v = sigmoid(W_v * visual_emb + b_v)
gate_t = sigmoid(W_t * track_emb + b_t)
fused = gate_a ⊙ audio_emb + gate_v ⊙ visual_emb + gate_t ⊙ track_emb
```

### Backward Compatibility Strategy

1. **Model Versioning**: Add `model_version` field to checkpoints
2. **Feature Detection**: Check for `enable_multimodal` flag in config
3. **Graceful Degradation**: If features missing, automatically disable multimodal
4. **Separate Configs**: Maintain separate YAML configs for multimodal vs unimodal

### Memory Optimization

1. **Lazy Loading**: Load features on-demand rather than all at once (REQUIRED for large datasets)
2. **Feature Caching**: Cache aligned features to avoid repeated interpolation
3. **Mixed Precision**: Use fp16 for CLIP embeddings to reduce memory (REQUIRED - saves 50% memory)
4. **Gradient Checkpointing**: Enable for transformer layers if memory constrained
5. **Feature Sampling Rate**: Assume 10 FPS (1 feature vector per 0.1 seconds) for balance between temporal resolution and memory

### Modality Masking Strategy

To prevent zero-filled features from introducing noise:

1. **Modality Availability Mask**: Track which modalities are actually present at each timestep
2. **Attention Masking**: In transformer, mask out unavailable modalities so they don't contribute to attention
3. **Loss Masking**: Don't compute gradients for predictions based on missing modalities
4. **Graceful Degradation**: Model should learn to make reasonable predictions even when some modalities are missing

**Example Scenarios:**
- Video with no audio track: modality_mask[:, 0] = False
- Frames with no face detected: Zero-fill face features, but modality_mask remains True (face_count=0 is valid information)
- Missing CLIP features: modality_mask[:, 1] = False for those frames
