# ==========================================
# metrics.py — Evaluation Metrics including Boundary F-Score
# ==========================================

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation.
    """
    
    def __init__(self, num_classes=1, thresholds=(0.5,)):
        self.num_classes = num_classes
        self.thresholds = thresholds
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_pixels = 0
    
    def update(self, pred_logits, target):
        """
        Update metrics with batch predictions.
        pred_logits: (B, C, H, W) or (B, 1, H, W) for binary
        target: (B, C, H, W) or (B, 1, H, W) for binary, or (B, H, W) for class indices
        """
        with torch.no_grad():
            if self.num_classes == 1:
                # Binary segmentation
                pred = (torch.sigmoid(pred_logits) > 0.5).float()
                target = target.float()
                
                self.tp += ((pred == 1) & (target == 1)).sum().item()
                self.fp += ((pred == 1) & (target == 0)).sum().item()
                self.fn += ((pred == 0) & (target == 1)).sum().item()
                self.tn += ((pred == 0) & (target == 0)).sum().item()
                self.total_pixels += target.numel()
            else:
                # Multi-class segmentation
                pred = pred_logits.argmax(dim=1)
                target = target.long()
                
                for c in range(self.num_classes):
                    pred_c = (pred == c).float()
                    target_c = (target == c).float()
                    
                    self.tp += ((pred_c == 1) & (target_c == 1)).sum().item()
                    self.fp += ((pred_c == 1) & (target_c == 0)).sum().item()
                    self.fn += ((pred_c == 0) & (target_c == 1)).sum().item()
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        if self.num_classes == 1:
            # Binary metrics
            iou = self.tp / (self.tp + self.fp + self.fn + 1e-7)
            dice = 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + 1e-7)
            accuracy = (self.tp + self.tn) / (self.total_pixels + 1e-7)
            
            return {
                'iou': iou,
                'dice': dice,
                'accuracy': accuracy,
                'precision': self.tp / (self.tp + self.fp + 1e-7),
                'recall': self.tp / (self.tp + self.fn + 1e-7)
            }
        else:
            # Multi-class mIoU
            iou = self.tp / (self.tp + self.fp + self.fn + 1e-7)
            miou = iou.mean()
            
            return {
                'miou': miou,
                'iou_per_class': iou
            }


class BoundaryMetrics:
    """
    Boundary-specific metrics (F-score, Boundary IoU).
    Based on standard boundary evaluation in segmentation papers.
    """
    
    def __init__(self, thresholds=(1, 2, 3)):
        """
        thresholds: Distance thresholds for boundary F-score in PIXELS.
        Common values: 1px, 2px, 3px
        """
        self.thresholds = thresholds
        self.reset()
    
    def reset(self):
        self.boundary_tp = {t: 0 for t in self.thresholds}
        self.boundary_fp = {t: 0 for t in self.thresholds}
        self.boundary_fn = {t: 0 for t in self.thresholds}
        self.boundary_iou_sum = {t: 0.0 for t in self.thresholds}
        self.boundary_iou_count = {t: 0 for t in self.thresholds}
    
    def extract_boundary(self, mask, radius=1):
        """
        Extract boundary from binary mask using morphological erosion.
        
        Args:
            mask: Binary mask tensor (B, 1, H, W) or (B, H, W)
            radius: Erosion radius in pixels
        
        Returns:
            boundary: Boundary map (B, 1, H, W)
        """
        # Ensure 4D tensor
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Pad for erosion
        padded = F.pad(mask, (radius, radius, radius, radius), mode='replicate')
        
        # Erosion using max pooling on inverted mask
        eroded = -F.max_pool2d(-padded, kernel_size=2*radius+1, stride=1, padding=0)
        
        # Boundary = original - eroded
        boundary = mask - eroded
        
        return torch.clamp(boundary, 0.0, 1.0)
    
    def compute_boundary_fscore(self, pred, target, threshold=1):
        """
        Compute Boundary F-Score at given threshold distance.
        
        Args:
            pred: Predicted binary mask (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
            threshold: Distance threshold in pixels
        
        Returns:
            fscore, precision, recall
        """
        # Extract boundaries
        pred_boundary = self.extract_boundary(pred, radius=1)
        target_boundary = self.extract_boundary(target, radius=1)
        
        # Dilate predicted boundary by threshold
        if threshold > 1:
            pred_dilated = F.max_pool2d(
                pred_boundary, 
                kernel_size=2*threshold+1, 
                stride=1, 
                padding=threshold
            )
        else:
            pred_dilated = pred_boundary
        
        # True positives: predicted boundary within threshold of target
        tp = ((pred_dilated > 0) & (target_boundary > 0)).sum().float()
        fp = ((pred_dilated > 0) & (target_boundary == 0)).sum().float()
        fn = ((pred_dilated == 0) & (target_boundary > 0)).sum().float()
        
        # Precision, Recall, F-Score
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        fscore = 2 * precision * recall / (precision + recall + 1e-7)
        
        return fscore.item(), precision.item(), recall.item()
    
    def compute_boundary_iou(self, pred, target, threshold=3):
        """
        Compute Boundary IoU (IoU in boundary regions only).
        
        Args:
            pred: Predicted binary mask (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
            threshold: Boundary region width in pixels
        
        Returns:
            boundary_iou
        """
        # Extract boundary regions from target
        target_boundary = self.extract_boundary(target, radius=threshold)
        
        # Create boundary mask (within threshold pixels of boundary)
        boundary_mask = (target_boundary > 0).float()
        
        # Compute IoU only in boundary regions
        pred_in_boundary = pred * boundary_mask
        target_in_boundary = target * boundary_mask
        
        intersection = (pred_in_boundary * target_in_boundary).sum()
        union = (pred_in_boundary + target_in_boundary).sum() - intersection
        
        boundary_iou = intersection / (union + 1e-7)
        
        return boundary_iou.item()
    
    def update(self, pred_logits, target):
        """
        Update boundary metrics.
        
        Args:
            pred_logits: Model predictions (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
        """
        with torch.no_grad():
            # Convert logits to binary predictions
            pred = (torch.sigmoid(pred_logits) > 0.5).float()
            target = target.float()
            
            # Ensure 4D tensors
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            if target.dim() == 3:
                target = target.unsqueeze(1)
            
            # Compute metrics for each threshold
            for t in self.thresholds:
                # Update Boundary F-Score
                fscore, precision, recall = self.compute_boundary_fscore(
                    pred, target, threshold=t
                )
                
                # Accumulate (will average in compute())
                self.boundary_tp[t] += fscore  # Store F-score directly
                self.boundary_fp[t] += 1  # Count batches
                # Note: This is simplified. For precise metrics, accumulate TP/FP/FN
                
                # Update Boundary IoU
                if t == 3:  # Only compute at 3px threshold
                    boundary_iou = self.compute_boundary_iou(pred, target, threshold=t)
                    self.boundary_iou_sum[t] += boundary_iou
                    self.boundary_iou_count[t] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute boundary F-scores for all thresholds"""
        results = {}
        
        for t in self.thresholds:
            # Boundary F-Score
            if self.boundary_fp[t] > 0:
                avg_fscore = self.boundary_tp[t] / self.boundary_fp[t]
                results[f'boundary_fscore_{t}px'] = avg_fscore
            
            # Boundary IoU (at 3px)
            if t == 3 and self.boundary_iou_count[t] > 0:
                avg_boundary_iou = self.boundary_iou_sum[t] / self.boundary_iou_count[t]
                results[f'boundary_iou_{t}px'] = avg_boundary_iou
        
        return results


def evaluate_model(model, dataloader, device, num_classes=1):
    """
    Comprehensive evaluation of a segmentation model.
    
    Args:
        model: Segmentation model
        dataloader: Data loader
        device: torch.device
        num_classes: 1 for binary, >1 for multi-class
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    seg_metrics = SegmentationMetrics(num_classes=num_classes)
    boundary_metrics = BoundaryMetrics(thresholds=(1, 2, 3))
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            seg_logits, edge_logits = model(images)
            
            # Update segmentation metrics
            seg_metrics.update(seg_logits, targets)
            
            # Update boundary metrics (only for binary)
            if num_classes == 1:
                boundary_metrics.update(seg_logits, targets)
    
    # Compute final metrics
    results = seg_metrics.compute()
    results.update(boundary_metrics.compute())
    
    return results
