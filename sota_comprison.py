# sota_comparison.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json

from config import cfg
from datasets import get_dataloaders
from sota_models import SOTAModelWrapper, count_parameters
from losses import BoundaryLoss, MultiClassBoundaryLoss
from metrics import SegmentationMetrics

def evaluate_model(model, dataloader, criterion, device, num_classes=1):
    """Evaluate a single model"""
    model.eval()
    seg_metrics = SegmentationMetrics(num_classes=num_classes)
    total_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            loss = criterion(logits, None, targets)
            total_loss += loss.item()
            
            if num_classes == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
            else:
                preds = logits.argmax(dim=1)
            
            seg_metrics.update(preds, targets)
    
    results = seg_metrics.compute()
    results['loss'] = total_loss / len(dataloader)
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    _, val_loader, _, val_size = get_dataloaders(
        dataset_name="oxford_pet",  # or "cityscapes"
        batch_size=8,
        image_size=224
    )
    
    # Define models to compare
    models_dict = {
        "MobileNetV3+DeepLabV3": SOTAModelWrapper.create_mobilenet_deeplabv3(num_classes=1),
        "EfficientNet-B0+DeepLabV3+": SOTAModelWrapper.create_efficientnet_deeplabv3plus(num_classes=1),
        "EfficientViT-B0 (Baseline)": SOTAModelWrapper.create_efficientvit(num_classes=1),
        "Ours (BARM)": None,  # Will load from checkpoint
    }
    
    # Load BARM model from checkpoint
    from models.barm_efficientvit import BoundaryAwareEfficientViT
    barm_model = BoundaryAwareEfficientViT(
        variant="b0",
        num_classes=1,
        use_barm=True
    )
    barm_model.load_state_dict(
        torch.load("checkpoints/best.pth", map_location=device, weights_only=True)
    )
    models_dict["Ours (BARM)"] = barm_model
    
    # Move all models to device
    for name, model in models_dict.items():
        models_dict[name] = model.to(device)
    
    # Loss function
    criterion = BoundaryLoss(edge_weight=0.4)
    
    # Evaluate all models
    results = {}
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        eval_results = evaluate_model(model, val_loader, criterion, device, num_classes=1)
        
        results[name] = {
            "params_m": count_parameters(model),
            "miou": eval_results.get('iou', eval_results.get('miou', 0)),
            "dice": eval_results.get('dice', 0),
            "loss": eval_results['loss']
        }
        
        print(f"  Parameters: {results[name]['params_m']:.2f}M")
        print(f"  mIoU: {results[name]['miou']:.4f}")
        print(f"  Dice: {results[name]['dice']:.4f}")
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/sota_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Model':<35} | {'Params (M)':<10} | {'mIoU':<10} | {'Dice':<10}")
    print("="*80)
    for name, res in results.items():
        print(f"{name:<35} | {res['params_m']:<10.2f} | {res['miou']:<10.4f} | {res['dice']:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    main()