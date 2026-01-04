"""
Integration tests for the Universal Compositional Embedder
"""
import pytest
import torch
import numpy as np
from PIL import Image

from src.models import UniversalCompositionalEmbedder


class TestIntegration:
    """Integration tests for the full system"""
    
    def test_full_model_pipeline(self, sample_text, sample_image, sample_audio):
        """Test the complete pipeline from input to output"""
        # Create model with smaller dimensions for faster testing
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Test text encoding
        mu_text, theta_text = model.encode_text(sample_text)
        assert mu_text.shape[0] == len(sample_text)
        assert mu_text.shape[1] == 64
        assert theta_text.shape[0] == len(sample_text)
        assert theta_text.shape[1] == 1
        
        # Test image encoding
        mu_img, theta_img = model.encode_image([sample_image])
        assert mu_img.shape[0] == 1
        assert mu_img.shape[1] == 64
        assert theta_img.shape[0] == 1
        assert theta_img.shape[1] == 1
        
        # Test audio encoding
        mu_audio, theta_audio = model.encode_audio([sample_audio], sampling_rate=16000)
        assert mu_audio.shape[0] == 1
        assert mu_audio.shape[1] == 64
        assert theta_audio.shape[0] == 1
        assert theta_audio.shape[1] == 1
        
        # Test cone operations between different modalities
        similarity = model.compute_cone_similarity((mu_text[0], theta_text[0]), (mu_img[0], theta_img[0]))
        assert isinstance(similarity, torch.Tensor)
        assert similarity.shape == (1,)
        
        # Test membership
        membership = model.test_membership(mu_text[0], (mu_img[0], theta_img[0]))
        assert isinstance(membership, torch.Tensor)
        assert membership.shape == (1,)
        assert membership.dtype == torch.bool
    
    def test_composition_with_different_modalities(self, sample_text, sample_image):
        """Test composing concepts from different modalities"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Encode different modalities
        mu_text, theta_text = model.encode_text(sample_text[:1])
        mu_img, theta_img = model.encode_image([sample_image])
        
        # Create cones list for composition
        cones_list = [
            (mu_text, theta_text),
            (mu_img, theta_img)
        ]
        
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
    
    def test_primitive_extraction(self, sample_text):
        """Test primitive concept extraction"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=20  # Larger number for better testing
        )
        
        # Encode text
        mu_text, _ = model.encode_text(sample_text)
        
        # Extract primitives
        top_similarities, top_indices = model.extract_primitives(mu_text[0:1], k=5)
        
        # Check shapes
        assert top_similarities.shape == (1, 5)  # (batch_size, k)
        assert top_indices.shape == (1, 5)  # (batch_size, k)
        
        # Check that indices are valid
        assert torch.all(top_indices >= 0)
        assert torch.all(top_indices < 20)  # num_primitives
    
    def test_cone_containment_relationships(self):
        """Test cone containment relationships"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Create two cones where one should contain the other
        mu_general = torch.randn(1, 64)
        mu_general = torch.nn.functional.normalize(mu_general, p=2, dim=-1)
        theta_general = torch.tensor([[0.8]])  # Larger radius for general concept
        
        mu_specific = mu_general.clone() + 0.1 * torch.randn(1, 64)
        mu_specific = torch.nn.functional.normalize(mu_specific, p=2, dim=-1)
        theta_specific = torch.tensor([[0.2]])  # Smaller radius for specific concept
        
        general_cone = (mu_general, theta_general)
        specific_cone = (mu_specific, theta_specific)
        
        # Test containment: specific_cone should be contained in general_cone
        containment = model.check_cone_containment(specific_cone, general_cone)
        
        assert isinstance(containment, torch.Tensor)
        assert containment.shape == (1,)
        assert containment.dtype == torch.bool
    
    def test_cross_modality_similarity(self, sample_text, sample_image, sample_audio):
        """Test similarity between different modalities"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Encode different modalities
        mu_text, theta_text = model.encode_text(sample_text[:1])
        mu_img, theta_img = model.encode_image([sample_image])
        mu_audio, theta_audio = model.encode_audio([sample_audio], sampling_rate=16000)
        
        # Test similarities between all pairs
        sim_text_img = model.compute_cone_similarity((mu_text, theta_text), (mu_img, theta_img))
        sim_text_audio = model.compute_cone_similarity((mu_text, theta_text), (mu_audio, theta_audio))
        sim_img_audio = model.compute_cone_similarity((mu_img, theta_img), (mu_audio, theta_audio))
        
        # All similarities should be valid tensors
        assert isinstance(sim_text_img, torch.Tensor)
        assert isinstance(sim_text_audio, torch.Tensor)
        assert isinstance(sim_img_audio, torch.Tensor)
        
        assert sim_text_img.shape == (1,)
        assert sim_text_audio.shape == (1,)
        assert sim_img_audio.shape == (1,)
    
    def test_model_forward_method(self, sample_text):
        """Test the main forward method with different modalities"""
        model = UniversalCompositionalEmbedder(
            output_dim=64,
            enable_composition=True,
            num_primitives=10
        )
        
        # Test text forward
        mu_text, theta_text = model(sample_text, 'text')
        assert mu_text.shape[0] == len(sample_text)
        assert theta_text.shape[0] == len(sample_text)
        
        # Test invalid modality
        with pytest.raises(ValueError):
            model(sample_text, 'invalid_modality')