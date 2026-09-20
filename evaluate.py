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
from metrics import evaluate_model


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


def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pth"
    
    model = BoundaryAwareEfficientViT(
        variant=cfg.efficientvit_variant,
        num_classes=cfg.num_classes,
        use_barm=cfg.use_barm
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with mIoU {checkpoint['miou']:.4f}")
    
    # Dataloaders
    _, val_loader, _, _ = get_dataloaders(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size
    )
    
    # Evaluate
    print("Evaluating model...")
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
    print("-"*60)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    print("="*60)
    
    # Save results
    output_path = Path(cfg.log_dir) / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': cfg.dataset_name,
            'model': f"BA-EfficientViT-{cfg.efficientvit_variant}",
            'parameters_m': num_params,
            'latency_ms': latency,
            'metrics': results
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()