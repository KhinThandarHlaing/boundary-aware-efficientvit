# Boundary-Aware EfficientViT

Master's Thesis Project: Lightweight Multi-Scale Attention with Edge Refinement for On-Device Semantic Segmentation

## Overview

This project implements a boundary-aware refinement module (BARM) on top of EfficientViT for efficient semantic segmentation on edge devices.

## Features

- ✅ EfficientViT backbone for lightweight segmentation
- ✅ Boundary-Aware Refinement Module (BARM)
- ✅ Auxiliary edge supervision
- ✅ Oxford-IIIT Pet and Cityscapes support
- ✅ Comprehensive evaluation metrics (mIoU, Dice, Boundary F-Score)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

## Dataset

- **Oxford-IIIT Pet:** Automatically downloaded
- **Cityscapes:** Manual download required (see documentation)

## Results

| Model | mIoU | Dice | Params | Latency |
|-------|------|------|--------|---------|
| Baseline | - | - | - | - |
| Ours (BARM) | - | - | - | - |

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{yourname2026boundary,
  title={Boundary-Aware EfficientViT: Lightweight Multi-Scale Attention with Edge Refinement},
  author={Your Name},
  school={Your University},
  year={2026}
}
```

## License

MIT License