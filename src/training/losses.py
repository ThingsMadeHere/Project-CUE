"""
Training Objectives Implementation
Implements the various loss functions needed for training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.cone_embeddings import ConeEmbedding


class CrossModalAlignmentLoss(nn.Module):
    """
    Contrastive loss to align embeddings of the same concept from different modalities
    """
    def __init__(self, temperature=0.07, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cone_ops = ConeEmbedding()
    
    def forward(self, mu_list, theta_list, labels):
        """
        Args:
            mu_list: List of mu tensors from different modalities, each of shape (batch_size, dim)
            theta_list: List of theta tensors from different modalities, each of shape (batch_size, 1)
            labels: Labels indicating which samples are the same concept across modalities
        Returns:
            loss: Scalar contrastive loss value
        """
        # Combine all embeddings from all modalities
        all_mu = torch.cat(mu_list, dim=0)  # (total_batch_size, dim)
        all_theta = torch.cat(theta_list, dim=0)  # (total_batch_size, 1)
        all_labels = torch.cat([labels] * len(mu_list), dim=0)  # (total_batch_size,)
        
        # Compute similarity matrix based on cone centroids
        similarity_matrix = torch.matmul(all_mu, all_mu.t())  # (total_batch_size, total_batch_size)
        
        # Create mask for positive pairs (same concept)
        labels = all_labels.unsqueeze(1)
        pos_mask = torch.eq(labels, labels.t()).float()  # (total_batch_size, total_batch_size)
        
        # Exclude self-similarity
        pos_mask = pos_mask - torch.diag_embed(torch.diag(pos_mask))
        
        # Compute the contrastive loss
        similarity_matrix = similarity_matrix / self.temperature
        exp_sim = torch.exp(similarity_matrix)
        
        # For each sample, sum the exponentiated similarities of positive pairs
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1)  # (total_batch_size,)
        
        # For each sample, sum all exponentiated similarities
        all_sum = torch.sum(exp_sim, dim=1)  # (total_batch_size,)
        
        # Compute the loss
        loss = -torch.log(pos_sum / all_sum + 1e-8)  # Add small epsilon to avoid log(0)
        loss = torch.mean(loss)
        
        return loss


class ConeContainmentLoss(nn.Module):
    """
    Differentiable loss to enforce cone inclusion for known parent-child relationships
    """
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
        self.cone_ops = ConeEmbedding()
    
    def forward(self, parent_cones, child_cones):
        """
        Args:
            parent_cones: Tuples of (mu_parent, theta_parent) for parent concepts
            child_cones: Tuples of (mu_child, theta_child) for child concepts
        Returns:
            loss: Scalar loss value enforcing parent ⊇ child containment
        """
        mu_parent, theta_parent = parent_cones
        mu_child, theta_child = child_cones
        
        # Compute angular distance between parent and child centroids
        cos_sim = torch.clamp(torch.sum(mu_parent * mu_child, dim=-1), -1.0, 1.0)
        angular_dist = torch.acos(cos_sim)
        
        # The containment constraint: arccos(μ_parent·μ_child) + θ_child <= θ_parent
        # We want: angular_dist + theta_child <= theta_parent
        # So the violation is: max(0, angular_dist + theta_child - theta_parent + margin)
        violations = torch.clamp(angular_dist + theta_child.squeeze() - theta_parent.squeeze() + self.margin, min=0.0)
        
        loss = torch.mean(violations)
        
        return loss


class PrimitiveGroundingLoss(nn.Module):
    """
    Loss to enforce stability and irreducibility of primitive concepts
    """
    def __init__(self, stability_weight=1.0, irreducibility_weight=1.0):
        super().__init__()
        self.stability_weight = stability_weight
        self.irreducibility_weight = irreducibility_weight
    
    def forward(self, primitive_embeddings, reconstructed_embeddings):
        """
        Args:
            primitive_embeddings: Tensor of primitive concept embeddings (batch_size, dim)
            reconstructed_embeddings: Tensor of embeddings reconstructed from other concepts (batch_size, dim)
        Returns:
            loss: Scalar loss value
        """
        # Stability: primitive embeddings should be stable and not change much
        stability_loss = 0  # For now, we'll focus on irreducibility
        
        # Irreducibility: primitive concepts should not be accurately reconstructible from other concepts
        # If reconstruction is too accurate, it means the concept is not truly primitive
        irreducibility_loss = F.mse_loss(primitive_embeddings, reconstructed_embeddings)
        
        # We want reconstruction to be poor, so we invert this (maximize reconstruction error)
        # Or we can use a different approach - penalize low reconstruction error
        irreducibility_loss = -torch.mean(torch.norm(primitive_embeddings - reconstructed_embeddings, dim=-1))
        
        total_loss = self.stability_weight * stability_loss + self.irreducibility_weight * irreducibility_loss
        
        return total_loss


class ContrastiveDiscriminationLoss(nn.Module):
    """
    Ensure cones for unrelated concepts are disjoint or minimally overlapping
    """
    def __init__(self, temperature=0.07, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, mu, theta, labels=None):
        """
        Args:
            mu: Centroid directions, shape (batch_size, dim)
            theta: Angular radii, shape (batch_size, 1)
            labels: Optional labels to know which pairs should be similar/dissimilar
        Returns:
            loss: Scalar loss value
        """
        batch_size = mu.size(0)
        
        # Compute similarity matrix based on cone centroids
        similarity_matrix = torch.matmul(mu, mu.t()) / self.temperature  # (batch_size, batch_size)
        
        # Create mask for negative pairs (different concepts if labels provided)
        if labels is not None:
            labels = labels.unsqueeze(1)
            neg_mask = 1 - torch.eq(labels, labels.t()).float()  # (batch_size, batch_size)
        else:
            # If no labels provided, treat all pairs as negative
            neg_mask = torch.ones(batch_size, batch_size, device=mu.device) - torch.eye(batch_size, device=mu.device)
        
        # Apply negative mask to similarity matrix
        neg_similarities = similarity_matrix * neg_mask
        
        # We want negative pairs to have low similarity
        # Use the log-sum-exp trick to compute loss
        exp_neg_sim = torch.exp(neg_similarities)
        sum_exp_neg = torch.sum(exp_neg_sim, dim=1)  # Sum over all negatives for each sample
        
        # We want the negative similarities to be as small as possible
        # So we minimize log(sum(exp(negative_similarities)))
        loss = torch.mean(torch.log(sum_exp_neg + 1e-8))  # Add epsilon to avoid log(0)
        
        return loss


class CombinedTrainingLoss(nn.Module):
    """
    Combined loss function incorporating all training objectives
    """
    def __init__(self, 
                 cross_modal_weight=1.0, 
                 containment_weight=1.0, 
                 primitive_weight=1.0, 
                 discrimination_weight=1.0,
                 temperature=0.07):
        super().__init__()
        
        self.cross_modal_weight = cross_modal_weight
        self.containment_weight = containment_weight
        self.primitive_weight = primitive_weight
        self.discrimination_weight = discrimination_weight
        
        self.cross_modal_loss = CrossModalAlignmentLoss(temperature=temperature)
        self.containment_loss = ConeContainmentLoss()
        self.primitive_loss = PrimitiveGroundingLoss()
        self.discrimination_loss = ContrastiveDiscriminationLoss(temperature=temperature)
    
    def forward(self, 
                mu_list, theta_list, labels,
                parent_cones=None, child_cones=None,
                primitive_embeddings=None, reconstructed_embeddings=None):
        """
        Args:
            mu_list: List of mu tensors from different modalities
            theta_list: List of theta tensors from different modalities
            labels: Labels for cross-modal alignment
            parent_cones: Tuples for containment loss (optional)
            child_cones: Tuples for containment loss (optional)
            primitive_embeddings: Embeddings for primitive grounding (optional)
            reconstructed_embeddings: Reconstructed embeddings for primitive grounding (optional)
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss values
        """
        # Cross-modal alignment loss
        cross_modal_loss_val = self.cross_modal_loss(mu_list, theta_list, labels)
        
        # Containment loss (if parent-child pairs provided)
        if parent_cones is not None and child_cones is not None:
            containment_loss_val = self.containment_loss(parent_cones, child_cones)
        else:
            containment_loss_val = torch.tensor(0.0, device=cross_modal_loss_val.device)
        
        # Primitive grounding loss (if embeddings provided)
        if primitive_embeddings is not None and reconstructed_embeddings is not None:
            primitive_loss_val = self.primitive_loss(primitive_embeddings, reconstructed_embeddings)
        else:
            primitive_loss_val = torch.tensor(0.0, device=cross_modal_loss_val.device)
        
        # Contrastive discrimination loss
        # For simplicity, using the first modality's embeddings
        discrimination_loss_val = self.discrimination_loss(mu_list[0], theta_list[0], labels)
        
        # Combine all losses
        total_loss = (
            self.cross_modal_weight * cross_modal_loss_val +
            self.containment_weight * containment_loss_val +
            self.primitive_weight * primitive_loss_val +
            self.discrimination_weight * discrimination_loss_val
        )
        
        loss_components = {
            'cross_modal': cross_modal_loss_val,
            'containment': containment_loss_val,
            'primitive': primitive_loss_val,
            'discrimination': discrimination_loss_val,
            'total': total_loss
        }
        
        return total_loss, loss_components