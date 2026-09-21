# ==========================================
# evaluate.py — Final Evaluation and Results
# ==========================================

import torch
import time
from pathlib import Path
import json

from config import cfg
from datasets import get_dataloaders
from models.barm_efficientvit import BoundaryAwareEfficientViT
from metrics import evaluate_model, SegmentationMetrics, BoundaryMetrics


def measure_latency(model, device, input_size=(1, 3, 224, 224), warmup=10, iterations=50):
    """
    Measure inference latency with proper CUDA synchronization.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / iterations) * 1000
    
    return latency_ms


def evaluate_single_model(model, dataloader, device, num_classes=1, model_name="Model"):
    """
    Evaluate a single model and return results.
    """
    model.eval()
    
    seg_metrics = SegmentationMetrics(num_classes=num_classes)
    boundary_metrics = BoundaryMetrics(thresholds=(1, 2, 3))
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
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


def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(cfg.checkpoint_dir) / "best.pth"  # Fixed: best.pth not best_model.pth
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    model = BoundaryAwareEfficientViT(
        variant=cfg.efficientvit_variant,
        num_classes=cfg.num_classes,
        use_barm=cfg.use_barm
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with mIoU {checkpoint['miou']:.4f}")
    
    # Dataloaders
    _, val_loader, _, val_size = get_dataloaders(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, val_loader, device, num_classes=cfg.num_classes)
    
    # Measure latency
    print("Measuring latency...")
    latency = measure_latency(model, device, input_size=(1, 3, cfg.image_size, cfg.image_size))
    
    # Count parameters
    num_params = model.count_parameters()
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Model: Boundary-Aware EfficientViT-{cfg.efficientvit_variant}")
    print(f"Parameters: {num_params:.2f}M")
    print(f"Latency: {latency:.2f} ms")
    print(f"Validation samples: {val_size}")
    print("-"*60)
    
    # Print metrics
    print("\nSegmentation Metrics:")
    if cfg.num_classes == 1:
        print(f"  IoU: {results.get('iou', 0):.4f} ({results.get('iou',
