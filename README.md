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
| Baseline | 0.6500 | 0.7800 | 0.20 | - |
| Ours (BARM) | 0.7107 | 0.8286 | 0.27 | - |

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{khinthandarhlaing2026boundary,
  title={Boundary-Aware EfficientViT: Lightweight Multi-Scale Attention with Edge Refinement},
  author={Khin Thandar Hlaing},
  school={University of WolverHampton},
  year={2026}
}
```

## License

MIT License
