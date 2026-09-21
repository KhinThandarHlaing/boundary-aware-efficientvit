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
        
        self.mask_transform = T.Resize((image_size, image_size), interpolation=Image.NEAREST)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # Convert trimap to binary mask
        # 1=pet, 2=background, 3=boundary/outline
        target_np = np.array(target, dtype=np.uint8)
        
        # Treat pet (1) and boundary (3) as foreground
        mask = (target_np != 2).astype(np.float32)
        
        # Convert to PIL Image for transform
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        
        # Apply transforms
        image = self.img_transform(image)
        mask_pil = self.mask_transform(mask_pil)
        
        # Convert back to tensor
        mask = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)
        
        return image, mask


# datasets.py - Cityscapes class ကို အစားထိုးပါ

class CityscapesSegmentation(Dataset):
    """
    Cityscapes-style segmentation dataset.

    Expected structure:
        root/
        ├── train/
        │   ├── img/
        │   └── label/
        └── val/
            ├── img/
            └── label/
    """

    def __init__(
        self,
        root: str = "./data/cityscapes",
        split: str = "train",
        image_size: int = 512
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        # Correct folder structure
        self.img_dir = self.root / split / "img"
        self.label_dir = self.root / split / "label"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}"
            )

        # Find images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.images = sorted([
            p for p in self.img_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in valid_extensions
        ])

        # Find labels
        self.labels = sorted([
            p for p in self.label_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in valid_extensions
        ])

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.img_dir}"
            )

        if len(self.labels) == 0:
            raise RuntimeError(
                f"No labels found in {self.label_dir}"
            )

        if len(self.images) != len(self.labels):
            raise RuntimeError(
                f"Image/label count mismatch: "
                f"{len(self.images)} images, "
                f"{len(self.labels)} labels"
            )

        print(
            f"{split}: "
            f"{len(self.images)} images, "
            f"{len(self.labels)} labels"
        )

        # Image transformation
        self.img_transform = T.Compose([
            T.Resize(
                (image_size, image_size),
                interpolation=Image.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mask transformation
        self.mask_transform = T.Resize(
            (image_size, image_size),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label_path = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)

        image = self.img_transform(image)

        mask = self.mask_transform(mask)

        mask_np = np.array(mask, dtype=np.int64)

        mask_tensor = torch.from_numpy(mask_np).long()

        return image, mask_tensor


def get_dataloaders(
    dataset_name: str = "oxford_pet",
    batch_size: int = 16,
    image_size: int = 224
):
    """
    Create train and validation dataloaders.
    """

    if dataset_name == "oxford_pet":

        train_dataset = OxfordIITPetSegmentation(
            split="trainval",
            image_size=image_size
        )

        val_dataset = OxfordIITPetSegmentation(
            split="test",
            image_size=image_size
        )

    elif dataset_name == "cityscapes":

        train_dataset = CityscapesSegmentation(
            root="./data/cityscapes",
            split="train",
            image_size=image_size
        )

        val_dataset = CityscapesSegmentation(
            root="./data/cityscapes",
            split="val",
            image_size=image_size
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

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

    return (
        train_loader,
        val_loader,
        len(train_dataset),
        len(val_dataset)
    )
