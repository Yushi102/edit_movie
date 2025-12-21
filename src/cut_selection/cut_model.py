"""
Cut Selection Model

Audio + Visual → Active prediction (binary: used/not used in edit)
"""
import torch
import torch.nn as nn
from src.model.multimodal_modules import ModalityEmbedding
from src.cut_selection.fusion import TwoModalityFusion
from src.cut_selection.positional_encoding import PositionalEncoding


class CutSelectionModel(nn.Module):
    """
    Model for cut selection (Stage 1)
    
    Input: Audio + Visual features from source video
    Output: Binary active prediction (0=not used, 1=used in edit)
    """
    
    def __init__(
        self,
        audio_features: int,
        visual_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        fusion_type: str = 'gated'
    ):
        super().__init__()
        
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.d_model = d_model
        
        # Modality embeddings
        self.audio_embedding = ModalityEmbedding(audio_features, d_model, dropout)
        self.visual_embedding = ModalityEmbedding(visual_features, d_model, dropout)
        
        # Positional encoding (essential for temporal understanding)
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # Fusion (2 modalities: audio + visual)
        self.fusion = TwoModalityFusion(d_model, fusion_type=fusion_type, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Active prediction head (binary classification)
        self.active_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 2 classes: inactive (0) or active (1)
        )
        
        print(f"CutSelectionModel initialized:")
        print(f"  Audio features: {audio_features}")
        print(f"  Visual features: {visual_features}")
        print(f"  Model dimension: {d_model}")
        print(f"  Attention heads: {nhead}")
        print(f"  Encoder layers: {num_encoder_layers}")
        print(f"  Fusion type: {fusion_type}")
    
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        """
        Forward pass
        
        Args:
            audio: (batch, seq_len, audio_features)
            visual: (batch, seq_len, visual_features)
            padding_mask: (batch, seq_len) - True for padding positions
        
        Returns:
            dict with:
                - active: (batch, seq_len, 2) - logits for binary classification
        """
        # Embed modalities
        audio_emb = self.audio_embedding(audio)  # (batch, seq_len, d_model)
        visual_emb = self.visual_embedding(visual)  # (batch, seq_len, d_model)
        
        # Fuse modalities
        fused = self.fusion(audio_emb, visual_emb)  # (batch, seq_len, d_model)
        
        # Add positional encoding (critical for temporal understanding)
        fused = self.positional_encoding(fused)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)
        
        # Active prediction
        active_logits = self.active_head(encoded)  # (batch, seq_len, 2)
        
        return {
            'active': active_logits
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
