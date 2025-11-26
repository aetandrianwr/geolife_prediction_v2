"""
Advanced loss functions for next-location prediction.

Includes:
1. Focal Loss for class imbalance
2. Multi-task loss (main + auxiliary)
3. Class-balanced loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss: Addressing class imbalance by down-weighting easy examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Balancing factor for class weights (default: 1)
        gamma: Focusing parameter (default: 2). Higher gamma reduces loss for well-classified examples.
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class indices
        """
        # Get probabilities
        p = F.softmax(inputs, dim=-1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
    
    The loss re-weights samples based on the effective number of samples per class.
    """
    
    def __init__(self, num_classes, samples_per_class, beta=0.9999, gamma=2.0, loss_type='focal'):
        """
        Args:
            num_classes: Total number of classes
            samples_per_class: Array of sample counts for each class
            beta: Hyperparameter for re-weighting (default: 0.9999)
            gamma: Focal loss gamma parameter
            loss_type: 'focal' or 'ce'
        """
        super().__init__()
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.loss_type = loss_type
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class indices
        """
        self.weights = self.weights.to(inputs.device)
        
        if self.loss_type == 'focal':
            # Focal loss with class balancing
            p = F.softmax(inputs, dim=-1)
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * ce_loss
            return loss.mean()
        else:
            # Standard cross-entropy with class balancing
            return F.cross_entropy(inputs, targets, weight=self.weights)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining main prediction and auxiliary reconstruction.
    
    Args:
        main_weight: Weight for main prediction loss
        aux_weight: Weight for auxiliary reconstruction loss
        use_focal: Whether to use focal loss
    """
    
    def __init__(self, main_weight=1.0, aux_weight=0.3, use_focal=True, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        
        if use_focal:
            self.main_criterion = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
            self.aux_criterion = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
        else:
            self.main_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.aux_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, main_logits, targets, aux_logits=None, aux_targets=None):
        """
        Args:
            main_logits: (batch_size, num_classes) - main predictions
            targets: (batch_size,) - main targets
            aux_logits: (batch_size, seq_len, num_classes) - auxiliary predictions
            aux_targets: (batch_size, seq_len) - auxiliary targets (locations in sequence)
        """
        # Main loss
        main_loss = self.main_criterion(main_logits, targets)
        
        # Auxiliary loss (if provided)
        if aux_logits is not None and aux_targets is not None:
            # Flatten for loss calculation
            batch_size, seq_len, num_classes = aux_logits.shape
            aux_logits_flat = aux_logits.reshape(-1, num_classes)
            aux_targets_flat = aux_targets.reshape(-1)
            
            # Only calculate loss for non-padding positions
            mask = aux_targets_flat > 0
            if mask.sum() > 0:
                aux_loss = self.aux_criterion(
                    aux_logits_flat[mask],
                    aux_targets_flat[mask]
                )
            else:
                aux_loss = 0.0
            
            total_loss = self.main_weight * main_loss + self.aux_weight * aux_loss
            return total_loss, main_loss, aux_loss
        
        return main_loss, main_loss, 0.0


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing with KL divergence.
    
    Helps prevent overconfidence and improves generalization.
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth target distribution
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # KL divergence
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.mean()


def get_loss_function(loss_type='focal', num_classes=1187, samples_per_class=None, **kwargs):
    """
    Factory function to create appropriate loss function.
    
    Args:
        loss_type: 'focal', 'class_balanced', 'multitask', 'ce', or 'label_smooth'
        num_classes: Number of location classes
        samples_per_class: Array of sample counts per class (for class-balanced loss)
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function module
    """
    if loss_type == 'focal':
        return FocalLoss(
            gamma=kwargs.get('gamma', 2.0),
            label_smoothing=kwargs.get('label_smoothing', 0.1)
        )
    
    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class_balanced loss")
        return ClassBalancedLoss(
            num_classes=num_classes,
            samples_per_class=samples_per_class,
            beta=kwargs.get('beta', 0.9999),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'multitask':
        return MultiTaskLoss(
            main_weight=kwargs.get('main_weight', 1.0),
            aux_weight=kwargs.get('aux_weight', 0.3),
            use_focal=kwargs.get('use_focal', True),
            gamma=kwargs.get('gamma', 2.0),
            label_smoothing=kwargs.get('label_smoothing', 0.1)
        )
    
    elif loss_type == 'label_smooth':
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=kwargs.get('label_smoothing', 0.1)
        )
    
    else:  # 'ce' or default
        return nn.CrossEntropyLoss(
            label_smoothing=kwargs.get('label_smoothing', 0.1)
        )
