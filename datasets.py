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

    def __init__(
        self,
        root="./data/cityscapes",
        split="train",
        image_size=512
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.image_dir = self.root / split / "img"
        self.label_dir = self.root / split / "label"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}"
            )

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        extensions = {".png", ".jpg", ".jpeg"}

        self.images = sorted(
            [
                p for p in self.image_dir.rglob("*")
                if p.suffix.lower() in extensions
            ]
        )

        self.labels = sorted(
            [
                p for p in self.label_dir.rglob("*")
                if p.suffix.lower() in extensions
            ]
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        if len(self.labels) == 0:
            raise RuntimeError(
                f"No labels found in {self.label_dir}"
            )

        if len(self.images) != len(self.labels):
            raise RuntimeError(
                f"Image/label mismatch: "
                f"{len(self.images)} images vs "
                f"{len(self.labels)} labels"
            )

        print(
            f"{split}: "
            f"{len(self.images)} images, "
            f"{len(self.labels)} labels"
        )

        # ----------------------------------------------------
        # Image transform
        # ----------------------------------------------------

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    # ========================================================
    # Convert RGB segmentation image -> class-index mask
    # ========================================================

    def rgb_to_class_mask(self, rgb_mask):

        rgb_mask = rgb_mask.astype(np.float32)

        h, w, _ = rgb_mask.shape

        pixels = rgb_mask.reshape(-1, 3)

        # Calculate squared RGB distance to each
        # standard Cityscapes semantic color.
        distances = (
            (
                pixels[:, None, :]
                - CITYSCAPES_PALETTE[None, :, :]
            ) ** 2
        ).sum(axis=2)

        class_ids = np.argmin(
            distances,
            axis=1
        )

        class_mask = class_ids.reshape(h, w)

        return class_mask.astype(np.int64)


    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(self, index):

        image_path = self.images[index]
        label_path = self.labels[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Load RGB label
        # ----------------------------------------------------

        rgb_label = np.array(
            Image.open(label_path).convert("RGB")
        )

        # Convert RGB colors -> integer classes
        mask = self.rgb_to_class_mask(rgb_label)

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        image = self.image_transform(image)

        # ----------------------------------------------------
        # Resize mask using NEAREST NEIGHBOR
        # ----------------------------------------------------

        mask = Image.fromarray(
            mask.astype(np.uint8)
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.NEAREST
        )

        mask = torch.from_numpy(
            np.array(mask, dtype=np.int64)
        )

        return image, mask


    def __len__(self):
        return len(self.images)


# ============================================================
# Keep Oxford dataset if your project uses it
# ============================================================

class OxfordIITPetSegmentation(Dataset):

    def __init__(
        self,
        split="trainval",
        image_size=224
    ):
        raise NotImplementedError(
            "OxfordIITPetSegmentation is not configured "
            "in this Cityscapes experiment."
        )


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
