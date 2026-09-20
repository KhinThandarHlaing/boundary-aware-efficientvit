# ==========================================
# datasets.py — Dataset Loading and Preprocessing
# ==========================================

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from config import cfg

class OxfordIITPetSegmentation(Dataset):
    """
    Oxford-IIIT Pet Dataset with correct trimap handling.
    Labels: 1=foreground, 2=background, 3=boundary
    We treat 1 and 3 as foreground for binary segmentation.
    """
    
    def __init__(self, root: str = "./data", split: str = "trainval", image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.split = split
        
        # Load official dataset
        self.dataset = torchvision.datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True
        )
        
        # Transforms
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST)
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # Convert trimap to binary mask
        # 1=pet, 2=background, 3=boundary/outline
        target_np = np.array(target, dtype=np.uint8)
        
        # Treat pet (1) and boundary (3) as foreground
        mask = (target_np != 2).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(Image.fromarray((mask.squeeze() * 255).astype(np.uint8)))
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = mask.unsqueeze(0)
        
        return image, mask


class CityscapesSegmentation(Dataset):
    """
    Cityscapes Dataset for multi-class semantic segmentation.
    Uses fine annotations and 19 classes.
    """
    
    def __init__(self, root: str = "./data/cityscapes", split: str = "train", image_size: int = 512):
        super().__init__()
        self.image_size = image_size
        self.split = split
        self.root = Path(root)
        
        # Cityscapes class mapping (19 classes)
        self.class_mapping = {
            6: 0,   # road
            7: 1,   # sidewalk
            8: 2,   # building
            11: 3,  # wall
            12: 4,  # fence
            13: 5,  # pole
            17: 6,  # traffic light
            19: 7,  # traffic sign
            20: 8,  # vegetation
            21: 9,  # terrain
            22: 10, # sky
            23: 11, # person
            24: 12, # rider
            25: 13, # car
            26: 14, # truck
            27: 15, # bus
            28: 16, # train
            31: 17, # motorcycle
            32: 18  # bicycle
        }
        
        self.images = []
        self.masks = []
        
        split_dir = "leftImg8bit/" + ("train" if split == "train" else "val")
        mask_dir = "gtFine/" + ("train" if split == "train" else "val")
        
        for img_path in (self.root / split_dir).glob("*/*_leftImg8bit.png"):
            self.images.append(img_path)
            # Corresponding mask
            mask_path = self.root / mask_dir / img_path.parent.name / img_path.name.replace("_leftImg8bit", "_gtFine_labelIds")
            self.masks.append(mask_path)
        
        # Transforms
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST)
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Apply transforms
        image = self.img_transform(image)
        mask_np = np.array(self.mask_transform(mask), dtype=np.int64)
        
        # Remap to 19 classes
        remapped = np.zeros_like(mask_np, dtype=np.int64)
        for cityscapes_id, class_id in self.class_mapping.items():
            remapped[mask_np == cityscapes_id] = class_id
        
        mask_tensor = torch.from_numpy(remapped).long()
        
        return image, mask_tensor


def get_dataloaders(dataset_name: str = "oxford_pet", batch_size: int = 16, image_size: int = 224):
    """
    Create train and validation dataloaders.
    """
    if dataset_name == "oxford_pet":
        train_dataset = OxfordIITPetSegmentation(split="trainval", image_size=image_size)
        val_dataset = OxfordIITPetSegmentation(split="test", image_size=image_size)
    elif dataset_name == "cityscapes":
        train_dataset = CityscapesSegmentation(split="train", image_size=image_size)
        val_dataset = CityscapesSegmentation(split="val", image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)