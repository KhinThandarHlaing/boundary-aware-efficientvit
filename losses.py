# ==========================================
# losses.py — Loss Functions with Boundary Supervision
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Auxiliary boundary loss with morphological edge extraction.
    """
    
    def __init__(self, edge_weight=0.4):
        super().__init__()
        self.edge_weight = edge_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def extract_boundary(self, mask, radius=1):
        """
        Extract boundary using morphological operations.
        Boundary = mask - eroded(mask)
        """
        # Pad for erosion
        padded = F.pad(mask, (radius, radius, radius, radius), mode='replicate')
        
        # Erosion using max pooling on inverted mask
        eroded = -F.max_pool2d(-padded, kernel_size=2*radius+1, stride=1, padding=0)
        
        # Boundary
        boundary = mask - eroded
        return torch.clamp(boundary, 0.0, 1.0)
    
    def forward(self, seg_logits, edge_logits, gt_mask):
        # Segmentation loss
        seg_bce = F.binary_cross_entropy_with_logits(seg_logits, gt_mask)
        seg_dice = self.dice(seg_logits, gt_mask)
        seg_loss = seg_bce + seg_dice
        
        # Edge loss
        if edge_logits is not None:
            gt_edge = self.extract_boundary(gt_mask)
            edge_bce = self.bce(edge_logits, gt_edge)
            edge_dice = self.dice(edge_logits, gt_edge)
            edge_loss = edge_bce + edge_dice
            
            return seg_loss + self.edge_weight * edge_loss
        else:
            return seg_loss


class MultiClassBoundaryLoss(nn.Module):
    """
    Multi-class segmentation loss with boundary supervision.
    For Cityscapes and ADE20K.
    """
    
    def __init__(self, num_classes=19, edge_weight=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.ce = nn.CrossEntropyLoss()
    
    def extract_multiclass_boundary(self, mask):
        """
        Extract boundaries for multi-class masks.
        """
        # One-hot encode
        one_hot = F.one_hot(mask.long(), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        
        # Erode each class
        padded = F.pad(one_hot, (1, 1, 1, 1), mode='replicate')
        eroded = F.max_pool2d(padded, kernel_size=3, stride=1, padding=0)
        
        # Boundary for each class
        boundaries = one_hot - eroded[:, :, 1:-1, 1:-1]
        
        # Combine all boundaries
        boundary_map = boundaries.sum(dim=1, keepdim=True)
        boundary_map = torch.clamp(boundary_map, 0.0, 1.0)
        
        return boundary_map
    
    def forward(self, seg_logits, edge_logits, gt_mask):
        # Segmentation loss
        seg_loss = self.ce(seg_logits, gt_mask.long())
        
        # Edge loss
        if edge_logits is not None:
            gt_edge = self.extract_multiclass_boundary(gt_mask)
            edge_bce = nn.BCEWithLogitsLoss()(edge_logits, gt_edge)
            
            return seg_loss + self.edge_weight * edge_bce
        else:
            return seg_loss