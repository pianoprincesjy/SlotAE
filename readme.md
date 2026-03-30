# Slot Autoencoder

**⚠️ This code must be placed in the `slotae/` folder within the [MetaSlot repository](https://github.com/lhj-lhj/MetaSlot/tree/main).**

This code was written within the MetaSlot repository.

## Overview

Train and evaluate an autoencoder that merges two slots into one and splits one slot into two, using slots from pretrained MetaSlot models.

- **Encoder**: Two slots → One slot
- **Decoder**: One slot → Two slots

## Model Architectures (`models.py`)

Five different model configurations are available:

1. **`linear`**: Simple linear transformation
2. **`nonlinear_simple`**: 2-layer MLP with ReLU
3. **`nonlinear_medium`**: 3-layer MLP with dropout (0.1)
4. **`nonlinear_deep`**: 4-layer MLP with BatchNorm and dropout
5. **`nonlinear_gelu`**: 3-layer MLP with GELU activation

## Training (`trainae2.py`)

### Command-Line Arguments

```bash
python trainae2.py \
    --model-config linear \
    --epochs 30 \
    --batch-size 512 \
    --lr 1e-3 \
    --metaslot-config /path/to/dinosaur_r-coco.py \
    --metaslot-checkpoint /path/to/checkpoint.pth \
    --save-dir ./pth \
    --gpu 5
```

**Key Arguments:**
- `--model-config`: Model architecture (`linear`, `nonlinear_simple`, `nonlinear_medium`, `nonlinear_deep`, `nonlinear_gelu`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 512)
- `--lr`: Learning rate (default: 1e-3)
- `--slot-dim`: Slot dimension (default: 256)
- `--metaslot-config`: Path to MetaSlot config file
- `--metaslot-checkpoint`: Path to pretrained MetaSlot checkpoint
- `--save-dir`: Directory to save checkpoints
- `--gpu`: GPU device ID (default: "5")

### Training Strategy

- Trains on all possible slot pairs: C(7,2) = 21 combinations
- Each batch processes all 21 pairs for comprehensive learning
- Saves checkpoint after every epoch
- Generates loss curve visualization and JSON loss history

### Output Files

Each training run creates a timestamped directory containing:
- `{model}_batch{size}_epoch_{num:04d}.pth`: Per-epoch checkpoints
- `{model}_batch{size}_final.pth`: Final checkpoint with full metadata
- `{model}_batch{size}_loss_history.json`: Training loss history
- `{model}_batch{size}_loss_curve.png`: Loss curve visualization
- `config.json`: Training configuration

### Batch Training Script

Run multiple experiments with `weekend.sh`:

```bash
bash weekend.sh
```

This script runs 30 experiments:
- 5 model types × 3 batch sizes (64, 256, 512) × 2 pretrained models (COCO, ClevrTex)

## Evaluation (`evalae4.py`)

### Configuration

Edit the hyperparameters at the top of the file:

```python
MODEL_CONFIG = 'linear'
MODEL_PATH = "/path/to/checkpoint.pth"
METASLOT_CONFIG = "/path/to/dinosaur_r-coco.py"
METASLOT_CHECKPOINT = "/path/to/checkpoint.pth"
NUM_SAMPLES = 1
ENCODER_PAIR = (0, 6)  # Slots to merge
DECODER_IDX = 3        # Slot to split
TEST_IMAGES = ["/path/to/image.png"]
OUTPUT_DIR = "./eval4"
```

### Run Evaluation

```bash
python evalae4.py
```

### Key Features

- **Direct Decoder Attention**: Uses MetaSlot decoder's `attent2` output (high-resolution mask logits) for visualization
- **Automatic Fallback**: If `model_config` is missing in checkpoint, uses the global `MODEL_CONFIG` variable
- **COCO Fallback**: Automatically uses COCO validation images if test images are not found

### Visualization Output

Results are saved as 2×3 grid visualizations:

**Row 1: Encoder (Merge)**
- Column 1: Original image with merge legend
- Column 2: 7 slots (before merge)
- Column 3: 6 slots (after merge, via decoder attent2)

**Row 2: Decoder (Split)**
- Column 1: Original image with split legend
- Column 2: 7 slots (before split)
- Column 3: 8 slots (after split, via decoder attent2)

### Winner-Take-All Visualization

- Each pixel is assigned to the slot with highest attention value
- Color-coded segmentation masks overlay the original image
- Legend shows which slots are being merged/split

## File Structure

```
slotae/
├── models.py           # Autoencoder architectures
├── trainae2.py        # Training script with argparse
├── evalae4.py         # Evaluation script with decoder attent2
├── trainae2.sh        # Single experiment script
├── weekend.sh         # Batch experiment script (30 runs)
├── readme.md          # This file
├── pth_coco/          # COCO training checkpoints
├── pth_clevrtex/      # ClevrTex training checkpoints
└── eval4/             # Evaluation results
```

## Requirements

- MetaSlot pretrained models (COCO or ClevrTex)
- PyTorch with CUDA support
- Python 3.10+
- Additional dependencies: numpy, matplotlib, PIL, opencv-python, einops, tqdm

## Notes

- GPU device 5 is used by default
- MetaSlot model is frozen during training (no gradient updates)
- Checkpoints include full metadata for reproducibility
- Loss is averaged across all 21 slot pairs per batch 