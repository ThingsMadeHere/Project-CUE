"""
Cone Representation System
Implements spherical cone representation (μ, θ) where:
- μ ∈ ℝᴰ: unit vector (centroid direction)
- θ ∈ (0, π/2]: angular radius (specificity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConeEmbedding(nn.Module):
    """
    Represents a concept as a spherical cone defined by:
    - μ: centroid direction (unit vector in R^D)
    - θ: angular radius (specificity, in (0, π/2])
    """
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
    
    def forward(self, mu, theta):
        """
        Args:
            mu: centroid directions, shape (batch_size, dim), should be unit vectors
            theta: angular radii, shape (batch_size,) or (batch_size, 1), in (0, π/2]
        Returns:
            cone: tuple of (mu, theta) with proper shapes
        """
        # Ensure mu is unit vector
        mu = F.normalize(mu, p=2, dim=-1)
        
        # Ensure theta is in valid range (0, π/2]
        theta = torch.clamp(theta, min=1e-6, max=math.pi/2 - 1e-6)
        
        return mu, theta
    
    @staticmethod
    def cone_intersection(cone1, cone2, eps=1e-8):
        """
        Approximate intersection of two cones using centroid averaging and radius reduction.
        Args:
            cone1: tuple (mu1, theta1)
            cone2: tuple (mu2, theta2)
        Returns:
            intersected_cone: tuple (mu_int, theta_int)
        """
        mu1, theta1 = cone1
        mu2, theta2 = cone2
        
        # Compute the average of the centroids and normalize
        mu_int = F.normalize(mu1 + mu2, p=2, dim=-1)
        
        # Compute angular distance between original centroids
        cos_sim = torch.clamp(torch.sum(mu1 * mu2, dim=-1), -1.0, 1.0)
        angular_dist = torch.acos(cos_sim)
        
        # Reduce the radius based on the intersection
        # The intersection radius is based on the minimum of the original radii
        # reduced by the angular distance between the centroids
        theta_int = torch.clamp(torch.min(theta1, theta2) - angular_dist/2, min=eps)
        
        return mu_int, theta_int
    
    @staticmethod
    def cone_containment(cone1, cone2):
        """
        Check if cone1 ⊆ cone2 (cone1 is contained in cone2)
        This is equivalent to arccos(μ1·μ2) + θ1 ≤ θ2
        Args:
            cone1: tuple (mu1, theta1) - inner cone
            cone2: tuple (mu2, theta2) - outer cone
        Returns:
            containment: boolean tensor indicating if cone1 ⊆ cone2
        """
        mu1, theta1 = cone1
        mu2, theta2 = cone2
        
        # Compute cosine similarity
        cos_sim = torch.clamp(torch.sum(mu1 * mu2, dim=-1), -1.0, 1.0)
        angular_dist = torch.acos(cos_sim)
        
        # Check containment condition
        containment = angular_dist + theta1 <= theta2
        
        return containment
    
    @staticmethod
    def cone_similarity(cone1, cone2):
        """
        Compute similarity between two cones based on angular distance between centroids
        Args:
            cone1: tuple (mu1, theta1)
            cone2: tuple (mu2, theta2)
        Returns:
            similarity: cosine similarity between centroids (higher = more similar)
        """
        mu1, _ = cone1
        mu2, _ = cone2
        
        cos_sim = torch.sum(mu1 * mu2, dim=-1)
        return cos_sim
    
    @staticmethod
    def membership_test(embedding, cone, eps=1e-8):
        """
        Check if an embedding is inside a cone
        Args:
            embedding: tensor of shape (..., dim), should be unit vector
            cone: tuple (mu, theta) representing the cone
        Returns:
            membership: boolean tensor of shape (...) indicating membership
        """
        mu, theta = cone
        
        # Ensure embedding is unit vector
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        # Compute angular distance between embedding and cone centroid
        cos_sim = torch.clamp(torch.sum(embedding * mu, dim=-1), -1.0, 1.0)
        angular_dist = torch.acos(cos_sim)
        
        # Check if angular distance is within the cone's angular radius
        membership = angular_dist <= theta - eps
        
        return membership


class ConeProjectionHead(nn.Module):
    """
    Lightweight trainable projection layer that maps modality-specific outputs 
    to a shared semantic space and outputs cone representation (μ, θ).
    """
    def __init__(self, input_dim, output_dim=256, up_project_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.up_project_dim = up_project_dim
        
        # Projection to get the base embedding
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Separate heads for mu (direction) and theta (radius)
        self.mu_head = nn.Linear(output_dim, output_dim)
        self.theta_head = nn.Linear(output_dim, 1)
        
        # Optional up-projection layer
        if up_project_dim and up_project_dim != output_dim:
            self.up_projection = nn.Linear(output_dim, up_project_dim)
        else:
            self.up_projection = None
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, input_dim)
        Returns:
            mu: unit vector of shape (batch_size, output_dim or up_project_dim)
            theta: angular radius of shape (batch_size, 1) in (0, π/2]
        """
        # Project to shared space
        h = torch.relu(self.projection(x))
        
        # Get cone parameters
        mu_raw = self.mu_head(h)  # Directions (not yet normalized)
        theta_raw = self.theta_head(h)  # Raw angular radius values
        
        # Normalize mu to unit vector
        mu = F.normalize(mu_raw, p=2, dim=-1)
        
        # Transform theta_raw to (0, π/2] range using sigmoid and scaling
        theta = (math.pi/2) * torch.sigmoid(theta_raw) + 1e-6
        
        # Apply up-projection if specified
        if self.up_projection is not None:
            mu = self.up_projection(mu)
            # Normalize again after up-projection to maintain unit vector
            mu = F.normalize(mu, p=2, dim=-1)
        
        return mu, theta