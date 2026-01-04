"""
Compositional Layer
Implements compound embedding synthesis from primitive cone embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cone_embeddings import ConeEmbedding


class CompositionalLayer(nn.Module):
    """
    Handles composition of primitive concepts to form compound concepts
    """
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.cone_ops = ConeEmbedding(dim=dim)
        
        # Learnable parameters for composition operations
        self.composition_weights = nn.Parameter(torch.ones(1))  # Learnable weight for composition
    
    def forward(self, cones_list):
        """
        Synthesize compound embedding from a list of primitive cone embeddings
        Args:
            cones_list: List of tuples [(mu1, theta1), (mu2, theta2), ...]
                       where each mu is (batch_size, dim) and theta is (batch_size, 1)
        Returns:
            compound_cone: tuple (mu_compound, theta_compound) representing the compound concept
        """
        if len(cones_list) == 0:
            raise ValueError("cones_list cannot be empty")
        if len(cones_list) == 1:
            return cones_list[0]  # Return the single cone if only one provided
        
        # Start with the first cone
        mu_compound, theta_compound = cones_list[0]
        
        # Sequentially compose with each additional cone
        for i in range(1, len(cones_list)):
            mu_next, theta_next = cones_list[i]
            mu_compound, theta_compound = self.cone_ops.cone_intersection(
                (mu_compound, theta_compound), 
                (mu_next, theta_next)
            )
        
        return mu_compound, theta_compound
    
    def compose_multiple(self, cones_tensor, mask=None):
        """
        Compose multiple cones simultaneously using tensor operations
        Args:
            cones_tensor: tuple of (mu_tensor, theta_tensor) 
                         where mu_tensor is (batch_size, num_cones, dim)
                         and theta_tensor is (batch_size, num_cones, 1)
            mask: Optional binary mask of shape (batch_size, num_cones) indicating valid cones
        Returns:
            compound_cone: tuple (mu_compound, theta_compound) of shape (batch_size, dim) and (batch_size, 1)
        """
        mu_tensor, theta_tensor = cones_tensor
        
        batch_size, num_cones, dim = mu_tensor.shape
        
        # Normalize all mu vectors
        mu_tensor = F.normalize(mu_tensor, p=2, dim=-1)
        
        if mask is not None:
            # Apply mask to ignore invalid cones
            mu_tensor = mu_tensor * mask.unsqueeze(-1)
            theta_tensor = theta_tensor * mask.unsqueeze(-1)
        
        # Sum all mu vectors and normalize (centroid averaging)
        mu_sum = torch.sum(mu_tensor, dim=1)  # (batch_size, dim)
        mu_compound = F.normalize(mu_sum, p=2, dim=-1)  # (batch_size, dim)
        
        # For theta, take the minimum (most specific) but adjust based on number of components
        # Using a more sophisticated approach: reduce theta based on number of components
        if mask is not None:
            valid_counts = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
            valid_counts = torch.clamp(valid_counts, min=1.0)
        else:
            valid_counts = torch.ones(batch_size, 1, device=theta_tensor.device) * num_cones
        
        # Average theta but reduce it based on number of components to reflect increased specificity
        theta_avg = torch.sum(theta_tensor, dim=1) / valid_counts  # (batch_size, 1)
        # Further reduce based on number of components to increase specificity
        theta_compound = theta_avg / torch.sqrt(valid_counts)
        
        # Ensure theta is within valid range
        theta_compound = torch.clamp(theta_compound, min=1e-6, max=3.14159/2 - 1e-6)
        
        return mu_compound, theta_compound


class PrimitiveConceptExtractor(nn.Module):
    """
    Identifies and extracts primitive concepts from embeddings
    """
    def __init__(self, dim=256, num_primitives=1000):
        super().__init__()
        self.dim = dim
        self.num_primitives = num_primitives
        
        # Learnable primitive concept embeddings
        self.primitive_embeddings = nn.Parameter(torch.randn(num_primitives, dim))
        nn.init.xavier_uniform_(self.primitive_embeddings)
        
        # Normalize primitive embeddings to unit vectors
        self.primitive_embeddings.data = F.normalize(self.primitive_embeddings.data, p=2, dim=-1)
    
    def forward(self, query_embedding):
        """
        Find the most relevant primitive concepts for a given embedding
        Args:
            query_embedding: tensor of shape (batch_size, dim)
        Returns:
            similarities: tensor of shape (batch_size, num_primitives) with cosine similarities
            top_indices: tensor of shape (batch_size, k) with indices of top k primitives
        """
        # Normalize query embedding
        query_normalized = F.normalize(query_embedding, p=2, dim=-1)
        
        # Compute similarities with all primitive concepts
        similarities = torch.matmul(query_normalized, self.primitive_embeddings.t())  # (batch_size, num_primitives)
        
        return similarities
    
    def get_top_primitives(self, query_embedding, k=5):
        """
        Get top k most relevant primitive concepts for a query embedding
        Args:
            query_embedding: tensor of shape (batch_size, dim)
            k: number of top primitives to return
        Returns:
            top_similarities: tensor of shape (batch_size, k)
            top_indices: tensor of shape (batch_size, k)
        """
        similarities = self.forward(query_embedding)
        top_similarities, top_indices = torch.topk(similarities, k=k, dim=-1)
        
        return top_similarities, top_indices


class ConceptComposabilityPredictor(nn.Module):
    """
    Predicts how well concepts can be composed together
    """
    def __init__(self, dim=256):
        super().__init__()
        self.composition_predictor = nn.Sequential(
            nn.Linear(2 * dim, dim),  # Concatenated embeddings
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()  # Output: probability of successful composition
        )
    
    def forward(self, mu1, mu2):
        """
        Predict how well two concepts can be composed
        Args:
            mu1: first concept embedding, shape (batch_size, dim)
            mu2: second concept embedding, shape (batch_size, dim)
        Returns:
            composability_score: tensor of shape (batch_size, 1)
        """
        # Concatenate the two embeddings
        combined = torch.cat([mu1, mu2], dim=-1)
        
        # Predict composability
        score = self.composition_predictor(combined)
        
        return score