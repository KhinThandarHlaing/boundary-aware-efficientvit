# ==========================================
# train.py — Main Training Loop
# ==========================================

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time

from config import cfg
from datasets import get_dataloaders
from models.barm_efficientvit import BoundaryAwareEfficientViT
from losses import BoundaryLoss, MultiClassBoundaryLoss
from metrics import SegmentationMetrics, BoundaryMetrics


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        seg_logits, edge_logits = model(images)
        
        if cfg.num_classes == 19:
            loss = criterion(seg_logits, edge_logits, targets)
        else:
            loss = criterion(seg_logits, edge_logits, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device, epoch):
    """Validate for one epoch"""
    model.eval()
    
    seg_metrics = SegmentationMetrics(num_classes=cfg.num_classes)
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            seg_logits, _ = model(images)
            seg_metrics.update(seg_logits, targets)
    
    results = seg_metrics.compute()
    return results


def main():
    # Setup
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Dataloaders
    train_loader, val_loader, train_size, val_size = get_dataloaders(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size
    )
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Model
    model = BoundaryAwareEfficientViT(
        variant=cfg.efficientvit_variant,
        num_classes=cfg.num_classes,
        use_barm=cfg.use_barm
    )
    model = model.to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:.2f}M")
    
    # Loss
    if cfg.num_classes == 1:
        criterion = BoundaryLoss(edge_weight=cfg.edge_loss_weight)
    else:
        criterion = MultiClassBoundaryLoss(num_classes=cfg.num_classes, edge_weight=cfg.edge_loss_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs
    )
    
    # Tensorboard
    writer = SummaryWriter(cfg.log_dir)
    
    # Training loop
    best_miou = 0.0
    
    for epoch in range(1, cfg.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_results = validate(model, val_loader, device, epoch)
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        if cfg.num_classes == 1:
            writer.add_scalar('Metrics/val_iou', val_results['iou'], epoch)
            writer.add_scalar('Metrics/val_dice', val_results['dice'], epoch)
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, IoU={val_results['iou']:.4f}, Dice={val_results['dice']:.4f}")
            
            if val_results['iou'] > best_miou:
                best_miou = val_results['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': best_miou,
                }, Path(cfg.checkpoint_dir) / f"best_model.pth")
        else:
            writer.add_scalar('Metrics/val_miou', val_results['miou'], epoch)
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, mIoU={val_results['miou']:.4f}")
            
            if val_results['miou'] > best_miou:
                best_miou = val_results['miou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': best_miou,
                }, Path(cfg.checkpoint_dir) / f"best_model.pth")
        
        # Scheduler step
        scheduler.step()
    
    writer.close()
    print(f"Training complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
