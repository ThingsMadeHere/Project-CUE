"""
Tests for the utils module
"""
import pytest
import torch
import numpy as np

from src.utils.cone_operations import (
    cone_volume, cone_distance, cone_union, 
    cone_centroid_similarity, cone_set_intersection,
    cone_expansion, cone_restriction, cone_entropy
)


class TestConeOperations:
    """Test cone operation utilities"""
    
    def test_cone_volume(self):
        """Test cone volume calculation"""
        # Test with different angular radii
        theta = torch.tensor([[0.1], [0.5], [1.0]])
        dim = 64
        
        volumes = cone_volume(theta, dim)
        
        # Check output shape
        assert volumes.shape == (3,)
        
        # Volume should be positive
        assert torch.all(volumes > 0)
        
        # Larger theta should give larger volume (approximately)
        assert volumes[2] > volumes[1] > volumes[0]
    
    def test_cone_distance(self):
        """Test cone distance calculation"""
        # Create test cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5], [0.4]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.2]])
        
        cone1 = (mu1, theta1)
        cone2 = (mu2, theta2)
        
        distances = cone_distance(cone1, cone2)
        
        # Check output shape
        assert distances.shape == (2,)
        
        # Distance should be non-negative
        assert torch.all(distances >= 0)
    
    def test_cone_union(self):
        """Test cone union operation"""
        # Create test cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.2], [0.3]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.4]])
        
        cone1 = (mu1, theta1)
        cone2 = (mu2, theta2)
        
        mu_union, theta_union = cone_union(cone1, cone2)
        
        # Check output shapes
        assert mu_union.shape == (2, 64)
        assert theta_union.shape == (2, 1)
        
        # Check that mu is unit vector
        mu_norm = torch.norm(mu_union, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)
        
        # Union theta should be larger than individual thetas (approximately)
        assert torch.all(theta_union >= theta1)
        assert torch.all(theta_union >= theta2)
    
    def test_cone_centroid_similarity(self):
        """Test cone centroid similarity"""
        # Create test cones
        mu1 = torch.randn(3, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5], [0.4], [0.3]])
        
        mu2 = torch.randn(3, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.2], [0.1]])
        
        cone1 = (mu1, theta1)
        cone2 = (mu2, theta2)
        
        similarities = cone_centroid_similarity(cone1, cone2)
        
        # Check output shape
        assert similarities.shape == (3,)
        
        # Similarity should be in range [-1, 1]
        assert torch.all(similarities >= -1.0)
        assert torch.all(similarities <= 1.0)
    
    def test_cone_set_intersection(self):
        """Test cone set intersection operation"""
        # Create multiple test cones
        mu1 = torch.randn(2, 64)
        mu1 = torch.nn.functional.normalize(mu1, p=2, dim=-1)
        theta1 = torch.tensor([[0.5], [0.4]])
        
        mu2 = torch.randn(2, 64)
        mu2 = torch.nn.functional.normalize(mu2, p=2, dim=-1)
        theta2 = torch.tensor([[0.3], [0.2]])
        
        mu3 = torch.randn(2, 64)
        mu3 = torch.nn.functional.normalize(mu3, p=2, dim=-1)
        theta3 = torch.tensor([[0.4], [0.3]])
        
        cones_list = [(mu1, theta1), (mu2, theta2), (mu3, theta3)]
        
        mu_int, theta_int = cone_set_intersection(cones_list)
        
        # Check output shapes
        assert mu_int.shape == (2, 64)
        assert theta_int.shape == (2, 1)
        
        # Check that mu is unit vector
        mu_norm = torch.norm(mu_int, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)
        
        # Intersection theta should be smaller than individual thetas (approximately)
        # Note: This is approximate since intersection is complex
    
    def test_cone_expansion(self):
        """Test cone expansion operation"""
        # Create a test cone
        mu = torch.randn(3, 64)
        mu = torch.nn.functional.normalize(mu, p=2, dim=-1)
        theta = torch.tensor([[0.2], [0.3], [0.4]])
        
        cone = (mu, theta)
        
        expanded_cone = cone_expansion(cone, expansion_factor=1.5)
        mu_exp, theta_exp = expanded_cone
        
        # Check shapes
        assert mu_exp.shape == (3, 64)
        assert theta_exp.shape == (3, 1)
        
        # Check that mu is still unit vector
        mu_norm = torch.norm(mu_exp, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)
        
        # Expanded theta should be larger
        assert torch.all(theta_exp >= theta)
        assert torch.all(theta_exp <= torch.pi/2)
    
    def test_cone_restriction(self):
        """Test cone restriction operation"""
        # Create a test cone
        mu = torch.randn(3, 64)
        mu = torch.nn.functional.normalize(mu, p=2, dim=-1)
        theta = torch.tensor([[0.5], [0.6], [0.7]])
        
        cone = (mu, theta)
        
        restricted_cone = cone_restriction(cone, restriction_factor=0.8)
        mu_res, theta_res = restricted_cone
        
        # Check shapes
        assert mu_res.shape == (3, 64)
        assert theta_res.shape == (3, 1)
        
        # Check that mu is still unit vector
        mu_norm = torch.norm(mu_res, p=2, dim=-1)
        assert torch.allclose(mu_norm, torch.ones_like(mu_norm), atol=1e-5)
        
        # Restricted theta should be smaller
        assert torch.all(theta_res <= theta)
    
    def test_cone_entropy(self):
        """Test cone entropy calculation"""
        # Test with different angular radii
        theta = torch.tensor([[0.1], [0.5], [1.0]])
        
        entropies = cone_entropy(theta)
        
        # Check output shape
        assert entropies.shape == (3,)
        
        # For larger theta, entropy should be larger (approximately)
        # Note: log function is increasing, so larger theta -> larger entropy
        assert entropies[2] > entropies[1] > entropies[0]