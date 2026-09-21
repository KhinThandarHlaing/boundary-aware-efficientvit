%%writefile /content/boundary-aware-efficientvit/datasets.py

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Cityscapes RGB -> class ID
# ============================================================

CITYSCAPES_COLORS = {
    0:  (128, 64, 128),    # road
    1:  (244, 35, 232),    # sidewalk
    2:  (70, 70, 70),      # building
    3:  (102, 102, 156),   # wall
    4:  (190, 153, 153),   # fence
    5:  (153, 153, 153),   # pole
    6:  (250, 170, 30),    # traffic light
    7:  (220, 220, 0),     # traffic sign
    8:  (107, 142, 35),    # vegetation
    9:  (152, 251, 152),   # terrain
    10: (70, 130, 180),    # sky
    11: (220, 20, 60),     # person
    12: (255, 0, 0),       # rider
    13: (0, 0, 142),       # car
    14: (0, 0, 70),        # truck
    15: (0, 60, 100),      # bus
    16: (0, 80, 100),      # train
    17: (0, 0, 230),       # motorcycle
    18: (119, 11, 32),     # bicycle
}

CITYSCAPES_PALETTE = np.array(
    list(CITYSCAPES_COLORS.values()),
    dtype=np.float32
)


# ============================================================
# Cityscapes Dataset
# ============================================================

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

    RGB labels are converted to Cityscapes class IDs 0-18.
    """

    # Standard Cityscapes RGB colors for the 19 semantic classes
    CITYSCAPES_COLORS = np.array([
        [128, 64, 128],   # 0 road
        [244, 35, 232],   # 1 sidewalk
        [70, 70, 70],     # 2 building
        [102, 102, 156],  # 3 wall
        [190, 153, 153],  # 4 fence
        [153, 153, 153],  # 5 pole
        [250, 170, 30],   # 6 traffic light
        [220, 220, 0],    # 7 traffic sign
        [107, 142, 35],   # 8 vegetation
        [152, 251, 152],  # 9 terrain
        [70, 130, 180],   # 10 sky
        [220, 20, 60],    # 11 person
        [255, 0, 0],      # 12 rider
        [0, 0, 142],      # 13 car
        [0, 0, 70],       # 14 truck
        [0, 60, 100],     # 15 bus
        [0, 80, 100],     # 16 train
        [0, 0, 230],      # 17 motorcycle
        [119, 11, 32],    # 18 bicycle
    ], dtype=np.float32)

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

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.images = sorted([
            p for p in self.img_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in valid_extensions
        ])

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

        self.mask_transform = T.Resize(
            (image_size, image_size),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def rgb_to_class_mask(self, mask_rgb):
        """
        Convert RGB semantic mask into class IDs 0-18.

        Each RGB pixel is assigned to the nearest
        standard Cityscapes class color.
        """

        # [H, W, 3]
        pixels = mask_rgb.reshape(-1, 3).astype(np.float32)

        # Calculate squared Euclidean distance:
        # [N, 1, 3] - [1, 19, 3]
        distances = np.sum(
            (pixels[:, None, :] -
             self.CITYSCAPES_COLORS[None, :, :]) ** 2,
            axis=2
        )

        # Nearest Cityscapes color
        class_ids = np.argmin(distances, axis=1)

        # Restore image shape
        class_ids = class_ids.reshape(
            mask_rgb.shape[0],
            mask_rgb.shape[1]
        )

        return class_ids.astype(np.int64)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label_path = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load RGB label
        mask = Image.open(label_path).convert("RGB")

        # Resize image
        image = self.img_transform(image)

        # Resize RGB mask using NEAREST
        mask = self.mask_transform(mask)

        # RGB -> class IDs
        mask_np = np.array(mask, dtype=np.uint8)

        mask_np = self.rgb_to_class_mask(mask_np)

        # [H,W] int64 tensor
        mask_tensor = torch.from_numpy(mask_np).long()

        return image, mask_tensor


# ============================================================
# DataLoaders
# ============================================================

def get_dataloaders(
    dataset_name="oxford_pet",
    batch_size=16,
    image_size=224
):

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
