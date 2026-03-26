# Slot Autoencoder

**⚠️ This code must be placed in the `slotae/` folder within the [MetaSlot repository](https://github.com/lhj-lhj/MetaSlot/tree/main).**

This code was written within the MetaSlot repository.

## Overview

Train and evaluate an autoencoder that merges two slots into one and splits one slot into two.

- **Encoder**: Two slots → One slot
- **Decoder**: One slot → Two slots

## Model Types

1. **Linear**: Simple linear transformation
2. **Nonlinear**: MLP-based nonlinear transformation

## Usage

### Training

```bash
python trainae.py
```

Set `USE_NONLINEAR` variable inside the code to choose linear/nonlinear.  
Trained model is saved as `slotae_linear.pth` or `slotae_nonlinear.pth`.

### Evaluation

```bash
python evalae.py
```

Set `MODEL_TYPE` variable inside the code to select the model.  
Results are saved as PNG files in the `eval/` folder.

## Visualization Format

Results are visualized in a 3x2 grid:

**Row 1 (Encoder)**
- Original image | 7 slots (★marked: merge targets) | 6 slots (merged)

**Row 2 (Decoder)**
- Original image | 7 slots (★marked: split target) | 8 slots (split)

## Configuration

- Pretrained model: `save/dinosaur_r-coco256/`
- CUDA device: 5
- Slot count: 7
- Slot dimension: 256 