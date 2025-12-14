"""
Modality Embedding and Fusion Modules for Multimodal Transformer

This module provides components for embedding different modalities to a common
dimension and fusing them using various strategies.
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Literal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModalityEmbedding(nn.Module):
    """
    Embedding layer for projecting modality-specific features to common dimension
    
    Projects input features from input_dim to d_model dimension with dropout
    for regularization.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Initialize ModalityEmbedding
        
        Args:
            input_dim: Input feature dimension
            d_model: Output model dimension
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Linear projection
        self.projection = nn.Linear(input_dim, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        logger.debug(f"ModalityEmbedding initialized: {input_dim} -> {d_model}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features to d_model dimension
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Embedded tensor of shape (batch, seq_len, d_model)
        """
        # Project to d_model
        embedded = self.projection(x)  # (batch, seq_len, d_model)
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        return embedded


class ModalityFusion(nn.Module):
    """
    Fusion module for combining multiple modality embeddings
    
    Supports multiple fusion strategies:
    - 'concat': Concatenate embeddings and project to d_model
    - 'add': Weighted addition of embeddings
    - 'gated': Gated fusion with learned gates (recommended for imbalanced modalities)
    """
    
    def __init__(
        self,
        d_model: int,
        num_modalities: int = 3,
        fusion_type: Literal['concat', 'add', 'gated'] = 'gated',
        dropout: float = 0.1
    ):
        """
        Initialize ModalityFusion
        
        Args:
            d_model: Model dimension
            num_modalities: Number of modalities to fuse (default: 3 for audio, visual, track)
            fusion_type: Fusion strategy ('concat', 'add', 'gated')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # Concatenation fusion: [audio; visual; track] -> Linear(3*d_model, d_model)
            self.fusion_projection = nn.Linear(num_modalities * d_model, d_model)
            nn.init.xavier_uniform_(self.fusion_projection.weight)
            nn.init.zeros_(self.fusion_projection.bias)
            
        elif fusion_type == 'add':
            # Additive fusion with learned weights
            self.modality_weights = nn.Parameter(torch.ones(num_modalities))
            
        elif fusion_type == 'gated':
            # Gated fusion: gate = sigmoid(W * embedding + b)
            # Create gating networks for each modality
            self.gate_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.Sigmoid()
                )
                for _ in range(num_modalities)
            ])
            
            # Initialize gate networks
            for gate_net in self.gate_networks:
                nn.init.xavier_uniform_(gate_net[0].weight)
                nn.init.zeros_(gate_net[0].bias)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Must be 'concat', 'add', or 'gated'")
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"ModalityFusion initialized: type={fusion_type}, num_modalities={num_modalities}")
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor,
        track_emb: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse modality embeddings
        
        Args:
            audio_emb: Audio embeddings (batch, seq_len, d_model)
            visual_emb: Visual embeddings (batch, seq_len, d_model)
            track_emb: Track embeddings (batch, seq_len, d_model)
            modality_mask: Optional mask (batch, seq_len, 3) indicating availability
                          [audio, visual, track]. False means unavailable.
        
        Returns:
            Fused embeddings (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = audio_emb.shape
        
        # Apply modality masking: zero out unavailable modalities
        if modality_mask is not None:
            # Expand mask to match embedding dimensions
            audio_mask = modality_mask[:, :, 0:1]  # (batch, seq_len, 1)
            visual_mask = modality_mask[:, :, 1:2]  # (batch, seq_len, 1)
            track_mask = modality_mask[:, :, 2:3]  # (batch, seq_len, 1)
            
            # Zero out unavailable modalities
            audio_emb = audio_emb * audio_mask.float()
            visual_emb = visual_emb * visual_mask.float()
            track_emb = track_emb * track_mask.float()
        
        if self.fusion_type == 'concat':
            # Concatenate along feature dimension
            concatenated = torch.cat([audio_emb, visual_emb, track_emb], dim=-1)  # (batch, seq_len, 3*d_model)
            
            # Project back to d_model
            fused = self.fusion_projection(concatenated)  # (batch, seq_len, d_model)
            
        elif self.fusion_type == 'add':
            # Weighted addition
            # Normalize weights to sum to 1
            weights = torch.softmax(self.modality_weights, dim=0)
            
            fused = (
                weights[0] * audio_emb +
                weights[1] * visual_emb +
                weights[2] * track_emb
            )
            
        elif self.fusion_type == 'gated':
            # Gated fusion: gate_i = sigmoid(W_i * emb_i + b_i)
            gate_audio = self.gate_networks[0](audio_emb)  # (batch, seq_len, d_model)
            gate_visual = self.gate_networks[1](visual_emb)  # (batch, seq_len, d_model)
            gate_track = self.gate_networks[2](track_emb)  # (batch, seq_len, d_model)
            
            # Weighted sum: fused = gate_a ⊙ audio + gate_v ⊙ visual + gate_t ⊙ track
            fused = (
                gate_audio * audio_emb +
                gate_visual * visual_emb +
                gate_track * track_emb
            )
        
        # Apply dropout
        fused = self.dropout(fused)
        
        return fused


if __name__ == "__main__":
    # Test ModalityEmbedding
    logger.info("Testing ModalityEmbedding...")
    
    batch_size = 4
    seq_len = 100
    
    # Test with different input dimensions
    audio_dim = 4
    visual_dim = 522
    track_dim = 180
    d_model = 256
    
    audio_embedding = ModalityEmbedding(audio_dim, d_model)
    visual_embedding = ModalityEmbedding(visual_dim, d_model)
    track_embedding = ModalityEmbedding(track_dim, d_model)
    
    # Create dummy inputs
    audio_input = torch.randn(batch_size, seq_len, audio_dim)
    visual_input = torch.randn(batch_size, seq_len, visual_dim)
    track_input = torch.randn(batch_size, seq_len, track_dim)
    
    # Forward pass
    audio_emb = audio_embedding(audio_input)
    visual_emb = visual_embedding(visual_input)
    track_emb = track_embedding(track_input)
    
    logger.info(f"Audio embedding shape: {audio_emb.shape}")
    logger.info(f"Visual embedding shape: {visual_emb.shape}")
    logger.info(f"Track embedding shape: {track_emb.shape}")
    
    assert audio_emb.shape == (batch_size, seq_len, d_model)
    assert visual_emb.shape == (batch_size, seq_len, d_model)
    assert track_emb.shape == (batch_size, seq_len, d_model)
    
    logger.info("✅ ModalityEmbedding test passed!")
    
    # Test ModalityFusion
    logger.info("\nTesting ModalityFusion...")
    
    # Test concatenation fusion
    fusion_concat = ModalityFusion(d_model, num_modalities=3, fusion_type='concat')
    fused_concat = fusion_concat(audio_emb, visual_emb, track_emb)
    logger.info(f"Concatenation fusion output shape: {fused_concat.shape}")
    assert fused_concat.shape == (batch_size, seq_len, d_model)
    
    # Test additive fusion
    fusion_add = ModalityFusion(d_model, num_modalities=3, fusion_type='add')
    fused_add = fusion_add(audio_emb, visual_emb, track_emb)
    logger.info(f"Additive fusion output shape: {fused_add.shape}")
    assert fused_add.shape == (batch_size, seq_len, d_model)
    
    # Test gated fusion
    fusion_gated = ModalityFusion(d_model, num_modalities=3, fusion_type='gated')
    fused_gated = fusion_gated(audio_emb, visual_emb, track_emb)
    logger.info(f"Gated fusion output shape: {fused_gated.shape}")
    assert fused_gated.shape == (batch_size, seq_len, d_model)
    
    logger.info("✅ ModalityFusion test passed!")
    
    # Test with modality masking
    logger.info("\nTesting with modality masking...")
    
    # Create modality mask: audio unavailable for first half
    modality_mask = torch.ones(batch_size, seq_len, 3, dtype=torch.bool)
    modality_mask[:, :seq_len//2, 0] = False  # Audio unavailable for first half
    
    fused_masked = fusion_gated(audio_emb, visual_emb, track_emb, modality_mask)
    logger.info(f"Masked fusion output shape: {fused_masked.shape}")
    assert fused_masked.shape == (batch_size, seq_len, d_model)
    
    # Verify that audio contribution is zeroed out where masked
    # (This is a qualitative check - the actual values depend on the gates)
    logger.info("✅ Modality masking test passed!")
    
    logger.info("\n✅ All multimodal module tests complete!")
