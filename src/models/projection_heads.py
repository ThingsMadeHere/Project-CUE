"""
Universal Projection Heads
Maps modality-specific outputs to shared semantic space and produces cone representations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cone_embeddings import ConeProjectionHead


class UniversalProjectionHeads(nn.Module):
    """
    Container for all modality-specific projection heads that map to shared semantic space
    """
    def __init__(self, text_encoder_dim=384, image_encoder_dim=768, audio_encoder_dim=384, video_encoder_dim=768, 
                 output_dim=256, up_project_dim=None):
        super().__init__()
        
        self.text_projection = ConeProjectionHead(
            input_dim=text_encoder_dim, 
            output_dim=output_dim, 
            up_project_dim=up_project_dim
        )
        
        self.image_projection = ConeProjectionHead(
            input_dim=image_encoder_dim, 
            output_dim=output_dim, 
            up_project_dim=up_project_dim
        )
        
        self.audio_projection = ConeProjectionHead(
            input_dim=audio_encoder_dim, 
            output_dim=output_dim, 
            up_project_dim=up_project_dim
        )
        
        self.video_projection = ConeProjectionHead(
            input_dim=video_encoder_dim, 
            output_dim=output_dim, 
            up_project_dim=up_project_dim
        )
    
    def forward(self, modality_type, embeddings):
        """
        Args:
            modality_type: string indicating the modality ('text', 'image', 'audio', 'video')
            embeddings: tensor of shape (batch_size, encoder_dim)
        Returns:
            mu: unit vector of shape (batch_size, output_dim or up_project_dim)
            theta: angular radius of shape (batch_size, 1) in (0, π/2]
        """
        if modality_type == 'text':
            return self.text_projection(embeddings)
        elif modality_type == 'image':
            return self.image_projection(embeddings)
        elif modality_type == 'audio':
            return self.audio_projection(embeddings)
        elif modality_type == 'video':
            return self.video_projection(embeddings)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")
    
    def project_text(self, text_embeddings):
        """Project text embeddings to cone representation"""
        return self.text_projection(text_embeddings)
    
    def project_image(self, image_embeddings):
        """Project image embeddings to cone representation"""
        return self.image_projection(image_embeddings)
    
    def project_audio(self, audio_embeddings):
        """Project audio embeddings to cone representation"""
        return self.audio_projection(audio_embeddings)
    
    def project_video(self, video_embeddings):
        """Project video embeddings to cone representation"""
        return self.video_projection(video_embeddings)


class MultiHeadAttentionPooling(nn.Module):
    """
    Learnable attention pooling for aggregating multiple embeddings (e.g., video frames)
    """
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            pooled_embedding: tensor of shape (batch_size, embedding_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(embeddings)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize along sequence dimension
        
        # Apply attention weights and sum
        pooled = torch.sum(attn_weights * embeddings, dim=1)  # (batch_size, embedding_dim)
        
        return pooled