# ==========================================
# metrics.py — Evaluation Metrics including Boundary F-Score
# ==========================================

import torch
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
    
    def __init__(self, thresholds=(1,2,3)):
        """
        thresholds: Distance thresholds for boundary F-score.
        Common values: 1px, 2px, 3px at normalized image scale.
        """
        self.thresholds = thresholds
        self.reset()
    
    def reset(self):
        self.boundary_tp = {t: 0 for t in self.thresholds}
        self.boundary_fp = {t: 0 for t in self.thresholds}
        self.boundary_fn = {t: 0 for t in self.thresholds}
    
    def compute_boundary_map(self, mask, radius=1):
        """Extract boundary from binary mask"""
        padded = torch.nn.functional.pad(mask, (radius, radius, radius, radius), mode='replicate')
        eroded = -torch.nn.functional.max_pool2d(-padded, kernel_size=2*radius+1, stride=1, padding=0)
        boundary = mask - eroded
        return torch.clamp(boundary, 0.0, 1.0)
    
    def update(self, pred_logits, target):
        """
        Update boundary metrics.
        """
        with torch.no_grad():
            pred = (torch.sigmoid(pred_logits) > 0.5).float()
            target = target.float()
            
            # Extract boundaries
            pred_boundary = self.compute_boundary_map(pred)
            target_boundary = self.compute_boundary_map(target)
            
            # Compute metrics for each threshold
            for t in self.thresholds:
                # Dilate target boundary by threshold distance
                # Simplified: use fixed radius based on threshold
                radius = max(1, int(t * 1000))
                
                pred_dilated = torch.nn.functional.max_pool2d(
                    pred_boundary, kernel_size=2*radius+1, stride=1, padding=radius
                )
                target_dilated = torch.nn.functional.max_pool2d(
                    target_boundary, kernel_size=2*radius+1, stride=1, padding=radius
                )
                
                # True positives: predicted boundary within threshold of target
                tp = ((pred_dilated > 0) & (target_boundary > 0)).sum().item()
                fp = ((pred_dilated > 0) & (target_boundary == 0)).sum().item()
                fn = ((pred_dilated == 0) & (target_boundary > 0)).sum().item()
                
                self.boundary_tp[t] += tp
                self.boundary_fp[t] += fp
                self.boundary_fn[t] += fn
    
    def compute(self) -> Dict[str, float]:
        """Compute boundary F-scores for all thresholds"""
        results = {}
        
        for t in self.thresholds:
            tp = self.boundary_tp[t]
            fp = self.boundary_fp[t]
            fn = self.boundary_fn[t]
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f_score = 2 * precision * recall / (precision + recall + 1e-7)
            
            results[f'boundary_f_{t:.4f}'] = f_score
            results[f'boundary_precision_{t:.4f}'] = precision
            results[f'boundary_recall_{t:.4f}'] = recall
        
        return results


def evaluate_model(model, dataloader, device, num_classes=1):
    """
    Comprehensive evaluation of a segmentation model.
    """
    model.eval()
    
    seg_metrics = SegmentationMetrics(num_classes=num_classes)
    boundary_metrics = BoundaryMetrics()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            seg_logits, edge_logits = model(images)
            
            # Update metrics
            seg_metrics.update(seg_logits, targets)
            boundary_metrics.update(seg_logits, targets)
    
    # Compute final metrics
    results = seg_metrics.compute()
    results.update(boundary_metrics.compute())
    
    return results
