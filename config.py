# ==========================================
# config.py — Experiment Configuration
# ==========================================

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ExperimentConfig:
    # Dataset
    dataset_name: str = "oxford_pet"  # "oxford_pet" or "cityscapes"
    image_size: int = 224  # 224 for Oxford, 512 for Cityscapes
    num_classes: int = 1  # 1 for Oxford, 19 for Cityscapes
    
    # Model
    use_barm: bool = True
    efficientvit_variant: str = "b0"  # "b0", "b1", "b2", "b3"
    
    # Training
    batch_size: int = 8  # Reduced for memory
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    edge_loss_weight: float = 0.4
    
    # Device
    device: str = "cuda"
    
    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Evaluation
    boundary_thresholds: Tuple[float, ...] = (1, 2, 3)

# Create config instance
cfg = ExperimentConfig()
