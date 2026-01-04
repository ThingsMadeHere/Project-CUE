"""
Main Universal Compositional Embedder with Cone-Based Semantics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder
from .projection_heads import UniversalProjectionHeads
from .compositional_layer import CompositionalLayer, PrimitiveConceptExtractor
from .cone_embeddings import ConeEmbedding


class UniversalCompositionalEmbedder(nn.Module):
    """
    Main model that combines all components:
    - Modality-specific encoders (frozen)
    - Universal projection heads (trainable)
    - Compositional layer (optional but recommended)
    """
    def __init__(self, 
                 text_encoder_model='sentence-transformers/all-MiniLM-L6-v2',
                 image_encoder_model='google/vit-base-patch16-224',
                 audio_encoder_model='openai/whisper-tiny',
                 output_dim=256,
                 up_project_dim=None,
                 enable_composition=True,
                 num_primitives=1000):
        super().__init__()
        
        # Initialize modality-specific encoders (frozen)
        self.text_encoder = TextEncoder(model_name=text_encoder_model, freeze=True)
        self.image_encoder = ImageEncoder(model_name=image_encoder_model, freeze=True)
        self.audio_encoder = AudioEncoder(model_name=audio_encoder_model, freeze=True)
        self.video_encoder = VideoEncoder(image_encoder=self.image_encoder)  # Reuse image encoder
        
        # Initialize universal projection heads
        self.projection_heads = UniversalProjectionHeads(
            text_encoder_dim=384,      # E5/SBERT typically outputs 384-dim
            image_encoder_dim=768,     # ViT-Base outputs 768-dim
            audio_encoder_dim=384,     # Whisper encoder outputs 384-dim
            video_encoder_dim=768,     # Video encoder uses image encoder output
            output_dim=output_dim,
            up_project_dim=up_project_dim
        )
        
        # Initialize compositional layer if enabled
        self.enable_composition = enable_composition
        if enable_composition:
            self.compositional_layer = CompositionalLayer(dim=output_dim if up_project_dim is None else up_project_dim)
            self.primitive_extractor = PrimitiveConceptExtractor(
                dim=output_dim if up_project_dim is None else up_project_dim,
                num_primitives=num_primitives
            )
        
        # Cone operations
        self.cone_ops = ConeEmbedding(dim=output_dim if up_project_dim is None else up_project_dim)
        
        # Store dimensions
        self.output_dim = output_dim
        self.up_project_dim = up_project_dim
    
    def encode_text(self, texts):
        """Encode text to cone representation"""
        text_embeddings = self.text_encoder(texts)
        mu, theta = self.projection_heads.project_text(text_embeddings)
        return mu, theta
    
    def encode_image(self, images):
        """Encode image to cone representation"""
        image_embeddings = self.image_encoder(images)
        mu, theta = self.projection_heads.project_image(image_embeddings)
        return mu, theta
    
    def encode_audio(self, audios, sampling_rate=16000):
        """Encode audio to cone representation"""
        audio_embeddings = self.audio_encoder(audios, sampling_rate=sampling_rate)
        mu, theta = self.projection_heads.project_audio(audio_embeddings)
        return mu, theta
    
    def encode_video(self, video_paths):
        """Encode video to cone representation"""
        video_embeddings = self.video_encoder(video_paths)
        mu, theta = self.projection_heads.project_video(video_embeddings)
        return mu, theta
    
    def forward(self, input_data, modality_type):
        """
        Main forward pass for single modality input
        Args:
            input_data: Input data appropriate for the modality type
            modality_type: String indicating the modality ('text', 'image', 'audio', 'video')
        Returns:
            mu: unit vector of shape (batch_size, output_dim or up_project_dim)
            theta: angular radius of shape (batch_size, 1) in (0, π/2]
        """
        if modality_type == 'text':
            return self.encode_text(input_data)
        elif modality_type == 'image':
            return self.encode_image(input_data)
        elif modality_type == 'audio':
            return self.encode_audio(input_data)
        elif modality_type == 'video':
            return self.encode_video(input_data)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")
    
    def compose_concepts(self, cones_list):
        """
        Compose multiple cone representations into a compound concept
        Args:
            cones_list: List of tuples [(mu1, theta1), (mu2, theta2), ...]
        Returns:
            compound_cone: tuple (mu_compound, theta_compound)
        """
        if not self.enable_composition:
            raise RuntimeError("Composition is not enabled for this model")
        
        return self.compositional_layer(cones_list)
    
    def extract_primitives(self, query_embedding, k=5):
        """
        Extract top k primitive concepts for a given embedding
        Args:
            query_embedding: tensor of shape (batch_size, dim)
            k: number of top primitives to return
        Returns:
            top_similarities: tensor of shape (batch_size, k)
            top_indices: tensor of shape (batch_size, k)
        """
        if not self.enable_composition:
            raise RuntimeError("Composition is not enabled for this model")
        
        return self.primitive_extractor.get_top_primitives(query_embedding, k=k)
    
    def check_cone_containment(self, cone1, cone2):
        """
        Check if cone1 ⊆ cone2 (cone1 is contained in cone2)
        Args:
            cone1: tuple (mu1, theta1) - inner cone
            cone2: tuple (mu2, theta2) - outer cone
        Returns:
            containment: boolean tensor indicating if cone1 ⊆ cone2
        """
        return self.cone_ops.cone_containment(cone1, cone2)
    
    def compute_cone_similarity(self, cone1, cone2):
        """
        Compute similarity between two cones
        Args:
            cone1: tuple (mu1, theta1)
            cone2: tuple (mu2, theta2)
        Returns:
            similarity: cosine similarity between centroids
        """
        return self.cone_ops.cone_similarity(cone1, cone2)
    
    def test_membership(self, embedding, cone):
        """
        Test if an embedding is inside a cone
        Args:
            embedding: tensor of shape (..., dim), should be unit vector
            cone: tuple (mu, theta) representing the cone
        Returns:
            membership: boolean tensor indicating membership
        """
        return self.cone_ops.membership_test(embedding, cone)


def create_model(config=None):
    """
    Factory function to create the model with specified configuration
    """
    if config is None:
        config = {}
    
    model = UniversalCompositionalEmbedder(
        text_encoder_model=config.get('text_encoder_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        image_encoder_model=config.get('image_encoder_model', 'google/vit-base-patch16-224'),
        audio_encoder_model=config.get('audio_encoder_model', 'openai/whisper-tiny'),
        output_dim=config.get('output_dim', 256),
        up_project_dim=config.get('up_project_dim', None),
        enable_composition=config.get('enable_composition', True),
        num_primitives=config.get('num_primitives', 1000)
    )
    
    return model