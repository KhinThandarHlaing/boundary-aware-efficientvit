# ablation_study.py

import torch
from pathlib import Path
import json
from config import cfg
from datasets import get_dataloaders
from models.barm_efficientvit import BoundaryAwareEfficientViT
from losses import BoundaryLoss
from metrics import SegmentationMetrics, BoundaryMetrics

def train_and_evaluate(use_barm, use_sobel, use_edge_loss, train_loader, val_loader, device, epochs=10):
    """Train model with specific BARM components enabled/disabled"""
    
    model = BoundaryAwareEfficientViT(
        variant="b0",
        num_classes=1,
        use_barm=use_barm
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = BoundaryLoss(edge_weight=0.4 if use_edge_loss else 0.0)
    
    # Training loop (simplified)
    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits, edges = model(images)
            loss = criterion(logits, edges, masks)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    seg_metrics = SegmentationMetrics(num_classes=1)
    boundary_metrics = BoundaryMetrics()
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits, _ = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            seg_metrics.update(preds, masks)
            boundary_results = boundary_metrics.compute_all(preds, masks)
    
    results = seg_metrics.compute()
    results.update(boundary_results)
    results['params_m'] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, val_loader, _, _ = get_dataloaders(
        dataset_name="oxford_pet",
        batch_size=8,
        image_size=224
    )
    
    # Ablation configurations
    ablation_configs = [
        {"name": "Baseline (no BARM)", "use_barm": False, "use_sobel": False, "use_edge_loss": False},
        {"name": "+ Feature Fusion", "use_barm": True, "use_sobel": False, "use_edge_loss": False},
        {"name": "+ Sobel Edge", "use_barm": True, "use_sobel": True, "use_edge_loss": False},
        {"name": "+ Edge Loss (Full BARM)", "use_barm": True, "use_sobel": True, "use_edge_loss": True},
    ]
    
    results = {}
    for config in ablation_configs:
        print(f"\nTraining: {config['name']}")
        eval_results = train_and_evaluate(
            use_barm=config['use_barm'],
            use_sobel=config['use_sobel'],
            use_edge_loss=config['use_edge_loss'],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=10  # Short training for ablation
        )
        results[config['name']] = eval_results
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_study.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print("\n" + "="*80)
    print(f"{'Configuration':<35} | {'IoU':<10} | {'Dice':<10} | {'Params':<10}")
    print("="*80)
    for name, res in results.items():
        print(f"{name:<35} | {res['iou']:<10.4f} | {res['dice']:<10.4f} | {res['params_m']:<10.2f}M")
    print("="*80)

if __name__ == "__main__":
    main()
