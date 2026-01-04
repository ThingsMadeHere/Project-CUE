"""
Tests for the Universal Compositional Embedder models
"""
import pytest
import torch
import numpy as np
from PIL import Image

from src.models import UniversalCompositionalEmbedder
from src.models.encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder
from src.models.projection_heads import UniversalProjectionHeads, ConeProjectionHead
from src.models.compositional_layer import CompositionalLayer, PrimitiveConceptExtractor
from src.models.cone_embeddings import ConeEmbedding, ConeProjectionHead as CPConeProjectionHead


class TestUniversalCompositionalEmbedder:
    """Test the main UniversalCompositionalEmbedder class"""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with default parameters"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Check that all components are initialized
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'image_encoder')
        assert hasattr(model, 'audio_encoder')
        assert hasattr(model, 'video_encoder')
        assert hasattr(model, 'projection_heads')
        assert hasattr(model, 'cone_ops')
        
        # Check that composition components are initialized when enabled
        assert hasattr(model, 'compositional_layer')
        assert hasattr(model, 'primitive_extractor')
        
        # Check dimensions
        assert model.output_dim == 64
    
    def test_model_forward_pass_text(self, sample_text):
        """Test forward pass with text input"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Test text encoding
        mu, theta = model.encode_text(sample_text)
        
        # Check output shapes
        assert mu.shape[0] == len(sample_text)  # batch dimension
        assert mu.shape[1] == 64  # output dimension
        assert theta.shape[0] == len(sample_text)  # batch dimension
        assert theta.shape[1] == 1  # single angular radius per sample
        
        # Check that mu vectors are unit vectors
        mu_norms = torch.norm(mu, p=2, dim=-1)
        assert torch.allclose(mu_norms, torch.ones_like(mu_norms), atol=1e-5)
        
        # Check that theta values are in valid range (0, π/2]
        assert torch.all(theta > 0)
        assert torch.all(theta <= torch.pi/2)
    
    def test_model_forward_pass_image(self, sample_image):
        """Test forward pass with image input"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Test image encoding with single image
        mu, theta = model.encode_image([sample_image])
        
        # Check output shapes
        assert mu.shape[0] == 1  # batch dimension
        assert mu.shape[1] == 64  # output dimension
        assert theta.shape[0] == 1  # batch dimension
        assert theta.shape[1] == 1  # single angular radius per sample
        
        # Check that mu vectors are unit vectors
        mu_norms = torch.norm(mu, p=2, dim=-1)
        assert torch.allclose(mu_norms, torch.ones_like(mu_norms), atol=1e-5)
        
        # Check that theta values are in valid range (0, π/2]
        assert torch.all(theta > 0)
        assert torch.all(theta <= torch.pi/2)
    
    def test_model_forward_pass_audio(self, sample_audio):
        """Test forward pass with audio input"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Test audio encoding
        mu, theta = model.encode_audio([sample_audio], sampling_rate=16000)
        
        # Check output shapes
        assert mu.shape[0] == 1  # batch dimension
        assert mu.shape[1] == 64  # output dimension
        assert theta.shape[0] == 1  # batch dimension
        assert theta.shape[1] == 1  # single angular radius per sample
        
        # Check that mu vectors are unit vectors
        mu_norms = torch.norm(mu, p=2, dim=-1)
        assert torch.allclose(mu_norms, torch.ones_like(mu_norms), atol=1e-5)
        
        # Check that theta values are in valid range (0, π/2]
        assert torch.all(theta > 0)
        assert torch.all(theta <= torch.pi/2)
    
    def test_cone_operations(self):
        """Test cone operations methods"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Create dummy cones
        mu1 = torch.randn(1, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)  # Make unit vector
        theta1 = torch.tensor([[0.5]])  # Angular radius in (0, π/2]
        
        mu2 = torch.randn(1, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)  # Make unit vector
        theta2 = torch.tensor([[0.3]])  # Angular radius in (0, π/2]
        
        cone1 = (mu1, theta1)
        cone2 = (mu2, theta2)
        
        # Test cone similarity
        similarity = model.compute_cone_similarity(cone1, cone2)
        assert isinstance(similarity, torch.Tensor)
        assert similarity.shape == (1,)
        assert -1 <= similarity.item() <= 1  # Cosine similarity range
        
        # Test cone containment
        containment = model.check_cone_containment(cone1, cone2)
        assert isinstance(containment, torch.Tensor)
        assert containment.shape == (1,)
        assert containment.dtype == torch.bool
        
        # Test membership test
        membership = model.test_membership(mu1, cone2)
        assert isinstance(membership, torch.Tensor)
        assert membership.shape == (1,)
        assert membership.dtype == torch.bool
    
    @pytest.mark.skip(reason="Composition requires more complex setup")
    def test_composition(self):
        """Test concept composition"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Create dummy cones for composition
        mu1 = torch.randn(1, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5]])
        
        mu2 = torch.randn(1, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3]])
        
        cones_list = [(mu1, theta1), (mu2, theta2)]
        
        # Test composition
        mu_compound, theta_compound = model.compose_concepts(cones_list)
        
        # Check output shapes
        assert mu_compound.shape[0] == 1
        assert mu_compound.shape[1] == 64
        assert theta_compound.shape[0] == 1
        assert theta_compound.shape[1] == 1
        
        # Check that mu is unit vector
        mu_norm = torch.norm(mu_compound, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)


class TestEncoders:
    """Test individual encoder components"""
    
    def test_text_encoder(self, sample_text):
        """Test TextEncoder functionality"""
        encoder = TextEncoder(model_name='sentence-transformers/all-MiniLM-L6-v2', freeze=True)
        
        # Test single text
        embedding = encoder(sample_text[0])
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 2  # (batch_size, hidden_size)
        assert embedding.shape[0] == 1  # batch size 1
        
        # Test multiple texts
        embeddings = encoder(sample_text)
        assert embeddings.shape[0] == len(sample_text)  # batch size matches input
        
        # Check that parameters are frozen
        for param in encoder.parameters():
            assert not param.requires_grad
    
    def test_image_encoder(self, sample_image):
        """Test ImageEncoder functionality"""
        encoder = ImageEncoder(model_name='google/vit-base-patch16-224', freeze=True)
        
        # Test single image
        embedding = encoder(sample_image)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 2  # (batch_size, hidden_size)
        assert embedding.shape[0] == 1  # batch size 1
        
        # Test multiple images
        embeddings = encoder([sample_image, sample_image])
        assert embeddings.shape[0] == 2  # batch size 2
        
        # Check that parameters are frozen
        for param in encoder.parameters():
            assert not param.requires_grad
    
    def test_audio_encoder(self, sample_audio):
        """Test AudioEncoder functionality"""
        encoder = AudioEncoder(model_name='openai/whisper-tiny', freeze=True)
        
        # Test single audio
        embedding = encoder(sample_audio, sampling_rate=16000)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 2  # (batch_size, hidden_size)
        assert embedding.shape[0] == 1  # batch size 1
        
        # Test multiple audios
        embeddings = encoder([sample_audio, sample_audio], sampling_rate=16000)
        assert embeddings.shape[0] == 2  # batch size 2
        
        # Check that parameters are frozen
        for param in encoder.parameters():
            assert not param.requires_grad


class TestProjectionHeads:
    """Test projection head components"""
    
    def test_universal_projection_heads(self):
        """Test UniversalProjectionHeads functionality"""
        heads = UniversalProjectionHeads(
            text_encoder_dim=384,
            image_encoder_dim=768,
            audio_encoder_dim=384,
            video_encoder_dim=768,
            output_dim=64
        )
        
        # Test text projection
        text_embedding = torch.randn(2, 384)
        mu_text, theta_text = heads.project_text(text_embedding)
        assert mu_text.shape == (2, 64)
        assert theta_text.shape == (2, 1)
        
        # Test image projection
        image_embedding = torch.randn(2, 768)
        mu_img, theta_img = heads.project_image(image_embedding)
        assert mu_img.shape == (2, 64)
        assert theta_img.shape == (2, 1)
        
        # Test audio projection
        audio_embedding = torch.randn(2, 384)
        mu_audio, theta_audio = heads.project_audio(audio_embedding)
        assert mu_audio.shape == (2, 64)
        assert theta_audio.shape == (2, 1)
        
        # Test video projection
        video_embedding = torch.randn(2, 768)
        mu_video, theta_video = heads.project_video(video_embedding)
        assert mu_video.shape == (2, 64)
        assert theta_video.shape == (2, 1)
    
    def test_cone_projection_head(self):
        """Test ConeProjectionHead functionality"""
        head = CPConeProjectionHead(input_dim=128, output_dim=64)
        
        # Test with input tensor
        x = torch.randn(3, 128)
        mu, theta = head(x)
        
        # Check output shapes
        assert mu.shape == (3, 64)
        assert theta.shape == (3, 1)
        
        # Check that mu vectors are unit vectors
        mu_norms = torch.norm(mu, p=2, dim=-1)
        assert torch.allclose(mu_norms, torch.ones_like(mu_norms), atol=1e-5)
        
        # Check that theta values are in valid range (0, π/2]
        assert torch.all(theta > 0)
        assert torch.all(theta <= torch.pi/2)


class TestCompositionalLayer:
    """Test compositional layer components"""
    
    def test_compositional_layer_initialization(self):
        """Test CompositionalLayer initialization"""
        layer = CompositionalLayer(dim=64)
        
        assert layer.dim == 64
        assert hasattr(layer, 'cone_ops')
        assert hasattr(layer, 'composition_weights')
    
    def test_compositional_layer_forward(self):
        """Test CompositionalLayer forward pass"""
        layer = CompositionalLayer(dim=64)
        
        # Create dummy cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5], [0.4]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.2]])
        
        cones_list = [(mu1, theta1), (mu2, theta2)]
        
        # Test composition
        mu_compound, theta_compound = layer(cones_list)
        
        # Check output shapes
        assert mu_compound.shape == (2, 64)
        assert theta_compound.shape == (2, 1)
        
        # Check that mu is unit vector
        mu_norm = torch.norm(mu_compound, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)
    
    def test_primitive_concept_extractor(self):
        """Test PrimitiveConceptExtractor functionality"""
        extractor = PrimitiveConceptExtractor(dim=64, num_primitives=10)
        
        # Test forward pass
        query_embedding = torch.randn(3, 64)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        
        similarities = extractor(query_embedding)
        assert similarities.shape == (3, 10)  # (batch_size, num_primitives)
        
        # Test get_top_primitives
        top_similarities, top_indices = extractor.get_top_primitives(query_embedding, k=3)
        assert top_similarities.shape == (3, 3)  # (batch_size, k)
        assert top_indices.shape == (3, 3)  # (batch_size, k)
        
        # Check that indices are within valid range
        assert torch.all(top_indices >= 0)
        assert torch.all(top_indices < 10)


class TestConeEmbeddings:
    """Test cone embedding operations"""
    
    def test_cone_embedding_initialization(self):
        """Test ConeEmbedding initialization"""
        cone_emb = ConeEmbedding(dim=64)
        
        assert cone_emb.dim == 64
    
    def test_cone_embedding_forward(self):
        """Test ConeEmbedding forward pass"""
        cone_emb = ConeEmbedding(dim=64)
        
        # Create test inputs
        mu = torch.randn(3, 64)
        theta = torch.tensor([[0.5], [0.3], [0.7]])
        
        # Forward pass
        mu_out, theta_out = cone_emb(mu, theta)
        
        # Check that mu is normalized to unit vector
        mu_norms = torch.norm(mu_out, p=2, dim=-1)
        assert torch.allclose(mu_norms, torch.ones_like(mu_norms), atol=1e-5)
        
        # Check that theta is in valid range
        assert torch.all(theta_out > 0)
        assert torch.all(theta_out <= torch.pi/2)
    
    def test_cone_intersection(self):
        """Test cone intersection operation"""
        # Create test cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5], [0.4]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.2]])
        
        cone1 = (mu1, theta1)
        cone2 = (mu2, theta2)
        
        # Test intersection
        mu_int, theta_int = ConeEmbedding.cone_intersection(cone1, cone2)
        
        # Check output shapes
        assert mu_int.shape == (2, 64)
        assert theta_int.shape == (2, 1)
        
        # Check that mu is unit vector
        mu_norm = torch.norm(mu_int, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)
        
        # Check that theta is positive
        assert torch.all(theta_int > 0)
    
    def test_cone_containment(self):
        """Test cone containment operation"""
        # Create test cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.1], [0.2]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.5], [0.6]])  # Larger radii for outer cone
        
        cone1 = (mu1, theta1)  # Inner cone
        cone2 = (mu2, theta2)  # Outer cone
        
        # Test containment: cone1 ⊆ cone2
        containment = ConeEmbedding.cone_containment(cone1, cone2)
        
        # Should be boolean tensor
        assert containment.dtype == torch.bool
        assert containment.shape == (2,)
    
    def test_cone_similarity(self):
        """Test cone similarity operation"""
        # Create test cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5], [0.4]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.2]])
        
        cone1 = (mu1, theta1)
        cone2 = (mu2, theta2)
        
        # Test similarity
        similarity = ConeEmbedding.cone_similarity(cone1, cone2)
        
        # Should be in range [-1, 1]
        assert similarity.shape == (2,)
        assert torch.all(similarity >= -1.0)
        assert torch.all(similarity <= 1.0)
    
    def test_membership_test(self):
        """Test membership test operation"""
        # Create a cone
        mu = torch.randn(1, 64)
        mu = torch.nn.functional.normalize(mu, p=2, dim=-1)
        theta = torch.tensor([[0.5]])
        
        cone = (mu, theta)
        
        # Create test embeddings
        test_embedding = torch.randn(3, 64)
        test_embedding = torch.nn.functional.normalize(test_embedding, p=2, dim=-1)
        
        # Test membership
        membership = ConeEmbedding.membership_test(test_embedding, cone)
        
        # Should be boolean tensor
        assert membership.dtype == torch.bool
        assert membership.shape == (3,)