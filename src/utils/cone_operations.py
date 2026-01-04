"""
Cone Operations Utilities
Additional mathematical operations for cone representations
"""
import torch
import torch.nn.functional as F
import math


def cone_volume(theta, dim):
    """
    Calculate the volume of a spherical cone in D-dimensional space
    This is proportional to the angular radius
    Args:
        theta: Angular radius, shape (batch_size,) or (batch_size, 1)
        dim: Dimension of the space
    Returns:
        volume: Volume of the cone (proportional value)
    """
    if theta.dim() == 2:
        theta = theta.squeeze(-1)
    
    # Volume is proportional to sin^(D-1)(theta) for small theta
    # For simplicity, using theta^(D-1) as approximation
    volume = torch.pow(theta, dim - 1)
    return volume


def cone_distance(cone1, cone2):
    """
    Compute a distance metric between two cones
    Args:
        cone1: tuple (mu1, theta1)
        cone2: tuple (mu2, theta2)
    Returns:
        distance: Distance between the cones
    """
    mu1, theta1 = cone1
    mu2, theta2 = cone2
    
    # Compute angular distance between centroids
    cos_sim = torch.clamp(torch.sum(mu1 * mu2, dim=-1), -1.0, 1.0)
    angular_dist = torch.acos(cos_sim)
    
    # Combine angular distance and difference in angular radii
    radius_diff = torch.abs(theta1 - theta2)
    
    # Weighted combination of centroid distance and radius difference
    distance = angular_dist + 0.5 * radius_diff.squeeze()
    
    return distance


def cone_union(cone1, cone2, eps=1e-8):
    """
    Approximate union of two cones (opposite of intersection)
    Args:
        cone1: tuple (mu1, theta1)
        cone2: tuple (mu2, theta2)
    Returns:
        union_cone: tuple (mu_union, theta_union)
    """
    mu1, theta1 = cone1
    mu2, theta2 = cone2
    
    # For union, we want to encompass both cones
    # Take the midpoint of centroids and expand the radius
    mu_union = F.normalize(mu1 + mu2, p=2, dim=-1)
    
    # Compute angular distance between original centroids
    cos_sim = torch.clamp(torch.sum(mu1 * mu2, dim=-1), -1.0, 1.0)
    angular_dist = torch.acos(cos_sim)
    
    # The union radius should encompass both original cones
    # It's at least the max of original radii plus half the angular distance between centroids
    theta_union = torch.max(theta1, theta2) + angular_dist/2 + eps
    
    # Ensure theta is in valid range
    max_theta = torch.tensor(math.pi/2 - eps, device=theta_union.device)
    theta_union = torch.min(theta_union, max_theta)
    
    return mu_union, theta_union


def cone_centroid_similarity(cone1, cone2):
    """
    Compute similarity between cone centroids using cosine similarity
    Args:
        cone1: tuple (mu1, theta1)
        cone2: tuple (mu2, theta2)
    Returns:
        similarity: Cosine similarity between centroids
    """
    mu1, _ = cone1
    mu2, _ = cone2
    
    similarity = torch.sum(mu1 * mu2, dim=-1)
    return similarity


def cone_set_intersection(cones_list):
    """
    Compute intersection of multiple cones
    Args:
        cones_list: List of tuples [(mu1, theta1), (mu2, theta2), ...]
    Returns:
        intersected_cone: tuple (mu_int, theta_int)
    """
    if len(cones_list) == 0:
        raise ValueError("cones_list cannot be empty")
    if len(cones_list) == 1:
        return cones_list[0]
    
    from src.models.cone_embeddings import ConeEmbedding
    cone_ops = ConeEmbedding()
    
    # Start with the first cone
    result_cone = cones_list[0]
    
    # Intersect with each subsequent cone
    for i in range(1, len(cones_list)):
        result_cone = cone_ops.cone_intersection(result_cone, cones_list[i])
    
    return result_cone


def cone_expansion(cone, expansion_factor=1.2):
    """
    Expand a cone by increasing its angular radius
    Args:
        cone: tuple (mu, theta)
        expansion_factor: Factor by which to expand the angular radius
    Returns:
        expanded_cone: tuple (mu, theta_expanded)
    """
    mu, theta = cone
    
    # Expand the angular radius
    theta_expanded = torch.clamp(theta * expansion_factor, max=math.pi/2 - 1e-6)
    
    return mu, theta_expanded


def cone_restriction(cone, restriction_factor=0.8):
    """
    Restrict a cone by decreasing its angular radius
    Args:
        cone: tuple (mu, theta)
        restriction_factor: Factor by which to restrict the angular radius
    Returns:
        restricted_cone: tuple (mu, theta_restricted)
    """
    mu, theta = cone
    
    # Restrict the angular radius
    theta_restricted = theta * restriction_factor
    
    return mu, theta_restricted


def cone_entropy(theta):
    """
    Compute entropy-like measure for a cone (larger cones have higher entropy)
    Args:
        theta: Angular radius, shape (batch_size,) or (batch_size, 1)
    Returns:
        entropy: Entropy measure
    """
    if theta.dim() == 2:
        theta = theta.squeeze(-1)
    
    # Entropy is proportional to the log of the angular radius
    entropy = torch.log(theta + 1e-8)  # Add small epsilon to avoid log(0)
    return entropy