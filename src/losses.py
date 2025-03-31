# losses.py
# Custom loss functions for ASL recognition models

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidencePenaltyLoss(nn.Module):
    """
    Loss function that adds a confidence penalty to standard cross-entropy loss.
    
    The confidence penalty encourages the model to make less confident predictions,
    helping to prevent overfitting and improve generalization.
    
    Args:
        weight (torch.Tensor, optional): Class weights for weighted cross-entropy.
        penalty_weight (float): Weight for the confidence penalty term.
        reduction (str): Specifies the reduction to apply to the output.
    """
    def __init__(self, weight=None, penalty_weight=0.1, reduction='mean'):
        super(ConfidencePenaltyLoss, self).__init__()
        self.weight = weight
        self.penalty_weight = penalty_weight
        self.reduction = reduction
        
    def forward(self, outputs, targets):
        """
        Forward pass of confidence penalty loss.
        
        Args:
            outputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss with confidence penalty
        """
        # Standard cross entropy loss with optional class weights
        ce_loss = F.cross_entropy(
            outputs, targets, 
            weight=self.weight,
            reduction=self.reduction
        )
        
        # Calculate entropy of predicted distribution
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Apply reduction to entropy term
        if self.reduction == 'mean':
            entropy = entropy.mean()
        elif self.reduction == 'sum':
            entropy = entropy.sum()
        
        # Combined loss: CE - entropy penalty
        # By subtracting entropy (with negative sign), we encourage higher entropy
        # Higher entropy = more uniform distribution = less confident predictions
        combined_loss = ce_loss - self.penalty_weight * entropy
        
        return combined_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard example mining.
    
    Args:
        weight (torch.Tensor, optional): Class weights for weighted cross-entropy.
        gamma (float): Focusing parameter - higher values give more weight to hard examples.
        alpha (float, optional): Class balancing parameter.
        reduction (str): Specifies the reduction to apply to the output.
    """
    def __init__(self, weight=None, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, outputs, targets):
        """
        Forward pass of focal loss.
        
        Args:
            outputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Focal loss
        """
        # Get softmax probabilities
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get probability of true class
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weights: (1-p)^gamma
        focal_weights = (1 - target_probs) ** self.gamma
        
        # Calculate cross entropy loss (per-sample)
        ce_loss = F.nll_loss(
            log_probs, targets, 
            weight=self.weight,
            reduction='none'
        )
        
        # Apply focal weights to CE loss
        focal_loss = focal_weights * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(config, class_weights=None):
    """
    Factory function to create the appropriate loss function based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        class_weights (torch.Tensor, optional): Optional class weights
        
    Returns:
        nn.Module: The configured loss function
    """
    if config.get("use_focal_loss", False):
        if config.get("use_confidence_penalty", False):
            return DynamicFocalConfidenceLoss(
                weight=class_weights,
                gamma=config.get("focal_gamma", 2.0),
                init_penalty=config.get("init_penalty_weight", 0.05),
                final_penalty=config.get("final_penalty_weight", 0.2),
                total_epochs=config.get("num_epochs", 100),
                reduction='mean'
            )
        else:
            return FocalLoss(
                weight=class_weights,
                gamma=config.get("focal_gamma", 2.0),
                reduction='mean'
            )
    elif config.get("use_confidence_penalty", False):
        return ConfidencePenaltyLoss(
            weight=class_weights,
            penalty_weight=config.get("confidence_penalty_weight", 0.1)
        )
    else:
        return nn.CrossEntropyLoss(weight=class_weights)

class DynamicFocalConfidenceLoss(nn.Module):
    """
    Combined focal loss with confidence penalty and dynamic weighting based on training progress.
    
    Args:
        weight (torch.Tensor, optional): Class weights for weighted cross-entropy.
        gamma (float): Focal loss focusing parameter.
        init_penalty (float): Initial confidence penalty weight.
        final_penalty (float): Final confidence penalty weight.
        total_epochs (int): Total number of training epochs.
        reduction (str): Specifies the reduction to apply to the output.
    """
    def __init__(self, weight=None, gamma=2.0, init_penalty=0.05, final_penalty=0.2, 
                 total_epochs=100, reduction='mean', initial_temperature=1.5, final_temperature=1.0):
        super(DynamicFocalConfidenceLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.init_penalty = init_penalty
        self.final_penalty = final_penalty
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.penalty_weight = init_penalty
        self.reduction = reduction
        
    def update_epoch(self, epoch):
        """Update current epoch and adjust penalty weight accordingly."""
        self.current_epoch = epoch
        # Gradually increase penalty weight as training progresses
        progress = min(1.0, self.current_epoch / self.total_epochs)
        self.penalty_weight = self.init_penalty + progress * (self.final_penalty - self.init_penalty)

        self.temperature = self.initial_temperature - progress * (self.initial_temperature - self.final_temperature)
        
    def forward(self, outputs, targets):
        """
        Forward pass of dynamic focal confidence loss.
        
        Args:
            outputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss with focal weighting and confidence penalty
        """
        # Get softmax probabilities
        scaled_outputs = outputs / self.temperature

        log_probs = F.log_softmax(scaled_outputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get probability of true class
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weights: (1-p)^gamma
        focal_weights = (1 - target_probs) ** self.gamma
        
        # Calculate cross entropy loss (per-sample)
        ce_loss = F.nll_loss(
            log_probs, targets, 
            weight=self.weight,
            reduction='none'
        )
        
        # Apply focal weights to CE loss
        focal_loss = focal_weights * ce_loss
        
        # Apply reduction to focal loss
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        # Calculate entropy for confidence penalty
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Apply reduction to entropy
        if self.reduction == 'mean':
            entropy = entropy.mean()
        elif self.reduction == 'sum':
            entropy = entropy.sum()
        
        # Combined loss: focal loss - entropy penalty
        combined_loss = focal_loss - self.penalty_weight * entropy
        
        return combined_loss