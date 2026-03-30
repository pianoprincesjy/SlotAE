#!/bin/bash
# Weekend Experiments Evaluation
# Visualize results from all 30 experiments (epoch 1, 5, 15, final)

# Set GPU device
export CUDA_VISIBLE_DEVICES=5

BASE_DIR="/home/jaey00ns/MetaSlot-main/slotae"
OUTPUT_DIR="${BASE_DIR}/evalweek"
VIS_DIR="${OUTPUT_DIR}/visualizations"

MODELS=(linear nonlinear_simple nonlinear_medium nonlinear_deep nonlinear_gelu)
BATCH_SIZES=(64 256 512)
EPOCHS=(1 5 15 30)  # 30 = final

# Test image
TEST_IMAGE="/home/jaey00ns/MetaSlot-main/imgs/slottest.png"

# MetaSlot configs
COCO_CONFIG="/home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-coco.py"
COCO_CHECKPOINT="/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth"
CLEVRTEX_CONFIG="/home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py"
CLEVRTEX_CHECKPOINT="/home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${VIS_DIR}/coco"
mkdir -p "${VIS_DIR}/clevrtex"

echo "========================================================"
echo "Weekend Experiments Evaluation"
echo "========================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Function to find latest run directory for specific batch size
find_latest_run_for_batch() {
    local base_dir=$1
    local model=$2
    local batch=$3
    
    local model_dir="${base_dir}/${model}"
    if [ ! -d "$model_dir" ]; then
        echo ""
        return
    fi
    
    # Find all run directories that contain this batch size
    for run_dir in $(ls -1dt "${model_dir}"/202* 2>/dev/null); do
        # Check if this run has files for this batch size
        if ls "${run_dir}/${model}_batch${batch}_"* >/dev/null 2>&1; then
            echo "$run_dir"
            return
        fi
    done
    
    echo ""
}

# Function to evaluate one checkpoint
evaluate_checkpoint() {
    local pretrain=$1
    local model=$2
    local batch=$3
    local epoch=$4
    local checkpoint_path=$5
    local metaslot_config=$6
    local metaslot_checkpoint=$7
    
    local epoch_label
    if [ "$epoch" -eq 30 ]; then
        epoch_label="final"
    else
        epoch_label="epoch$(printf '%02d' $epoch)"
    fi
    
    local output_dir="${VIS_DIR}/${pretrain}/${model}"
    mkdir -p "${output_dir}"
    local output_file="${output_dir}/batch${batch}_${epoch_label}.png"
    
    echo "  → Evaluating: ${model} / batch ${batch} / ${epoch_label}"
    
    # Create temporary eval script
    cat > /tmp/eval_temp.py << 'EOFPYTHON'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
from einops import rearrange

sys.path.append('/home/jaey00ns/MetaSlot-main')
sys.path.append('/home/jaey00ns/MetaSlot-main/slotae')
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config
from models import create_autoencoder

IMAGENET_MEAN = np.array([[[123.675]], [[116.28]], [[103.53]]], dtype=np.float32)
IMAGENET_STD = np.array([[[58.395]], [[57.12]], [[57.375]]], dtype=np.float32)

def preprocess_image(image_path, target_size=(256, 256)):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    original = np.array(image)
    width, height = image.size
    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    
    image = image.crop((left, top, right, bottom))
    image = image.resize(target_size, Image.BILINEAR)
    
    image_np = np.array(image, dtype=np.float32)
    image_np = image_np.transpose(2, 0, 1)
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    
    image_tensor = pt.from_numpy(image_np).unsqueeze(0).float()
    return image_tensor, original

def generate_attent2_from_slots(new_slots, decoder):
    clue = [16, 16]
    recon, attent2 = decoder(clue, new_slots)
    B, N, HW = attent2.shape
    h = w = 16
    attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)
    attent2 = F.interpolate(attent2, size=(256, 256), mode='bilinear', align_corners=False)
    return attent2, recon

def generate_slot_colors(num_slots):
    colors = []
    for i in range(num_slots):
        hue = i / num_slots
        saturation = 1.0
        value = 1.0
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(rgb)
    return np.array(colors)

def visualize_slots_with_mask(original_image, attention_maps, slot_indices, all_colors, alpha=0.5):
    H, W = attention_maps.shape[1:]
    num_slots = attention_maps.shape[0]
    
    orig_resized = cv2.resize(original_image, (W, H))
    max_slot_indices = np.argmax(attention_maps, axis=0)
    
    segmentation_mask = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(num_slots):
        slot_mask = (max_slot_indices == i)
        color = all_colors[slot_indices[i]]
        for c in range(3):
            segmentation_mask[slot_mask, c] = color[c]
    
    overlay_image = orig_resized.astype(np.float32) / 255.0
    blended = (1 - alpha) * overlay_image + alpha * segmentation_mask
    vis_image = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return vis_image

def add_legend_to_image(image, slot_colors, merge_indices=None, split_index=None, output_indices=None):
    img = image.copy()
    H, W = img.shape[:2]
    
    box_size = 25
    padding = 15
    spacing = 8
    
    y_offset = padding
    x_start = W - padding - 200
    
    if merge_indices is not None:
        cv2.putText(img, "Merge:", (x_start, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, "Merge:", (x_start, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_pos = x_start + 60
        for idx in merge_indices:
            color = (slot_colors[idx] * 255).astype(np.uint8).tolist()
            cv2.rectangle(img, (x_pos, y_offset), 
                         (x_pos + box_size, y_offset + box_size), color, -1)
            cv2.rectangle(img, (x_pos, y_offset), 
                         (x_pos + box_size, y_offset + box_size), (0, 0, 0), 2)
            x_pos += box_size + spacing
        
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x_pos += 30
        
        if output_indices and len(output_indices) > 0:
            out_idx = output_indices[0]
            out_color = (slot_colors[out_idx] * 255).astype(np.uint8).tolist()
            cv2.rectangle(img, (x_pos, y_offset), 
                         (x_pos + box_size, y_offset + box_size), out_color, -1)
            cv2.rectangle(img, (x_pos, y_offset), 
                         (x_pos + box_size, y_offset + box_size), (0, 0, 0), 2)
    
    elif split_index is not None:
        cv2.putText(img, "Split:", (x_start, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, "Split:", (x_start, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        x_pos = x_start + 50
        color = (slot_colors[split_index] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x_pos, y_offset), 
                     (x_pos + box_size, y_offset + box_size), color, -1)
        cv2.rectangle(img, (x_pos, y_offset), 
                     (x_pos + box_size, y_offset + box_size), (0, 0, 0), 2)
        x_pos += box_size + spacing
        
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x_pos += 30
        
        if output_indices:
            for out_idx in output_indices[:2]:
                out_color = (slot_colors[out_idx] * 255).astype(np.uint8).tolist()
                cv2.rectangle(img, (x_pos, y_offset), 
                             (x_pos + box_size, y_offset + box_size), out_color, -1)
                cv2.rectangle(img, (x_pos, y_offset), 
                             (x_pos + box_size, y_offset + box_size), (0, 0, 0), 2)
                x_pos += box_size + spacing
    
    return img

def visualize_autoencoder_results(original_image, slots, decoder, autoencoder,
                                  original_attention_output, encoder_pair=(0, 1), decoder_idx=3, save_path=None):
    device = slots.device
    num_slots = slots.shape[0]
    
    max_possible_slots = num_slots + 2
    all_colors = generate_slot_colors(max_possible_slots)
    
    idx1, idx2 = encoder_pair
    remaining_indices = [i for i in range(num_slots) if i not in encoder_pair]
    
    original_slot_indices = list(range(num_slots))
    img_01 = visualize_slots_with_mask(original_image, original_attention_output, 
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_00 = add_legend_to_image(original_image, all_colors, 
                                  merge_indices=encoder_pair, 
                                  output_indices=[num_slots])
    
    slot1 = slots[idx1:idx1+1]
    slot2 = slots[idx2:idx2+1]
    
    with pt.no_grad():
        encoded_slot = autoencoder.encode(slot1, slot2)
        merged_slots = pt.cat([slots[remaining_indices], encoded_slot], dim=0).unsqueeze(0)
        attent2_merged, recon_merged = generate_attent2_from_slots(merged_slots, decoder)
        attent2_merged = attent2_merged.squeeze(0).cpu().numpy()
    
    merged_slot_indices = remaining_indices + [num_slots]
    img_02 = visualize_slots_with_mask(original_image, attent2_merged,
                                        merged_slot_indices, all_colors, alpha=0.5)
    
    original_slot_indices = list(range(num_slots))
    img_11 = visualize_slots_with_mask(original_image, original_attention_output,
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_10 = add_legend_to_image(original_image, all_colors,
                                  split_index=decoder_idx,
                                  output_indices=[num_slots, num_slots+1])
    
    slot_to_split = slots[decoder_idx:decoder_idx+1]
    
    with pt.no_grad():
        slot_recon1, slot_recon2 = autoencoder.decode(slot_to_split)
        remaining_indices_split = [i for i in range(num_slots) if i != decoder_idx]
        split_slots = pt.cat([slots[remaining_indices_split], slot_recon1, slot_recon2], dim=0).unsqueeze(0)
        attent2_split, recon_split = generate_attent2_from_slots(split_slots, decoder)
        attent2_split = attent2_split.squeeze(0).cpu().numpy()
    
    split_slot_indices = remaining_indices_split + [num_slots, num_slots+1]
    img_12 = visualize_slots_with_mask(original_image, attent2_split,
                                        split_slot_indices, all_colors, alpha=0.5)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img_00)
    axes[0, 0].set_title('Original Image\n(with Merge legend)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_01)
    axes[0, 1].set_title(f'7 Slots (Before Merge)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_02)
    axes[0, 2].set_title(f'After Merge: 6 slots\n(Decoder attent2)', 
                        fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_10)
    axes[1, 0].set_title('Original Image\n(with Split legend)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_11)
    axes[1, 1].set_title(f'7 Slots (Before Split)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_12)
    axes[1, 2].set_title(f'After Split: 8 slots\n(Decoder attent2)', 
                        fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

# Main evaluation
device = 'cuda' if pt.cuda.is_available() else 'cpu'

cfg = Config.fromfile("METASLOT_CONFIG_PLACEHOLDER")
metaslot_model = build_from_config(cfg.model)
metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)

state = pt.load("METASLOT_CHECKPOINT_PLACEHOLDER", map_location="cpu", weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]
metaslot_model.load_state_dict(state, strict=False)
metaslot_model = metaslot_model.to(device).eval()

decoder = metaslot_model.m.decode

checkpoint = pt.load("MODEL_PATH_PLACEHOLDER", map_location=device, weights_only=False)

if 'model_config' not in checkpoint:
    model_config = "MODEL_CONFIG_PLACEHOLDER"
else:
    model_config = checkpoint['model_config']

slot_dim = checkpoint.get('slot_dim', 256)

autoencoder = create_autoencoder(model_config, slot_dim=slot_dim)
autoencoder.load_state_dict(checkpoint['model_state_dict'])
autoencoder = autoencoder.to(device).eval()

image_tensor, original_image = preprocess_image("TEST_IMAGE_PLACEHOLDER")
image_tensor = image_tensor.to(device)

with pt.no_grad():
    batch = {'image': image_tensor}
    output = metaslot_model(batch)
    slots = output['slotz'].squeeze(0)
    
    if 'attent2' in output:
        original_attention = output['attent2']
    elif 'attent' in output:
        original_attention = output['attent']
    else:
        raise ValueError("No attention output found")
    
    if original_attention.shape[-1] != 256:
        original_attention = F.interpolate(original_attention, size=(256, 256), mode='bilinear', align_corners=False)
    
    original_attention_256 = original_attention.squeeze(0).cpu().numpy()

original_resized = cv2.resize(original_image, (256, 256))

save_path = Path("OUTPUT_FILE_PLACEHOLDER")
save_path.parent.mkdir(parents=True, exist_ok=True)

visualize_autoencoder_results(
    original_resized,
    slots,
    decoder,
    autoencoder,
    original_attention_256,
    encoder_pair=(0, 6),
    decoder_idx=3,
    save_path=save_path
)

print(f"✓ Saved: {save_path}")
EOFPYTHON
    
    # Replace placeholders
    sed -i "s|METASLOT_CONFIG_PLACEHOLDER|${metaslot_config}|g" /tmp/eval_temp.py
    sed -i "s|METASLOT_CHECKPOINT_PLACEHOLDER|${metaslot_checkpoint}|g" /tmp/eval_temp.py
    sed -i "s|MODEL_PATH_PLACEHOLDER|${checkpoint_path}|g" /tmp/eval_temp.py
    sed -i "s|MODEL_CONFIG_PLACEHOLDER|${model}|g" /tmp/eval_temp.py
    sed -i "s|TEST_IMAGE_PLACEHOLDER|${TEST_IMAGE}|g" /tmp/eval_temp.py
    sed -i "s|OUTPUT_FILE_PLACEHOLDER|${output_file}|g" /tmp/eval_temp.py
    
    # Run evaluation (suppress most output)
    python /tmp/eval_temp.py 2>&1 | grep -E "(✓|Saved|Error|Warning)" || echo "    ${pretrain}/${model}/batch${batch}/${epoch_label} completed"
    
    # Cleanup
    rm /tmp/eval_temp.py
}

# Counter for progress
total_experiments=0
completed_experiments=0

# Count total experiments to process
for pretrain in coco clevrtex; do
    if [ "$pretrain" == "coco" ]; then
        base_dir="${BASE_DIR}/pth_coco"
    else
        base_dir="${BASE_DIR}/pth_clevrtex"
    fi
    
    for model in "${MODELS[@]}"; do
        for batch in "${BATCH_SIZES[@]}"; do
            run_dir=$(find_latest_run_for_batch "$base_dir" "$model" "$batch")
            
            if [ -n "$run_dir" ]; then
                for epoch in "${EPOCHS[@]}"; do
                    ((total_experiments++))
                done
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "Total experiments to evaluate: ${total_experiments}"
echo "========================================================"
echo ""

# COCO experiments
echo "========== COCO Experiments =========="
for model in "${MODELS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        run_dir=$(find_latest_run_for_batch "${BASE_DIR}/pth_coco" "$model" "$batch")
        
        if [ -z "$run_dir" ]; then
            echo ""
            echo "⚠️  Missing: ${model} / batch ${batch}"
            continue
        fi
        
        echo ""
        echo "Processing: ${model} / batch ${batch} (${run_dir##*/})"
        
        for epoch in "${EPOCHS[@]}"; do
            if [ "$epoch" -eq 30 ]; then
                # Final checkpoint
                checkpoint="${run_dir}/${model}_batch${batch}_final.pth"
            else
                # Epoch checkpoint
                checkpoint="${run_dir}/${model}_batch${batch}_epoch_$(printf '%04d' $epoch).pth"
            fi
            
            if [ -f "$checkpoint" ]; then
                evaluate_checkpoint "coco" "$model" "$batch" "$epoch" \
                    "$checkpoint" "$COCO_CONFIG" "$COCO_CHECKPOINT"
                ((completed_experiments++))
                echo "    Progress: ${completed_experiments}/${total_experiments}"
            else
                echo "  ⚠️  Checkpoint not found: epoch ${epoch}"
            fi
        done
    done
done

echo ""
echo "========== ClevrTex Experiments =========="
for model in "${MODELS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        run_dir=$(find_latest_run_for_batch "${BASE_DIR}/pth_clevrtex" "$model" "$batch")
        
        if [ -z "$run_dir" ]; then
            echo ""
            echo "⚠️  Missing: ${model} / batch ${batch}"
            continue
        fi
        
        echo ""
        echo "Processing: ${model} / batch ${batch} (${run_dir##*/})"
        
        for epoch in "${EPOCHS[@]}"; do
            if [ "$epoch" -eq 30 ]; then
                # Final checkpoint
                checkpoint="${run_dir}/${model}_batch${batch}_final.pth"
            else
                # Epoch checkpoint
                checkpoint="${run_dir}/${model}_batch${batch}_epoch_$(printf '%04d' $epoch).pth"
            fi
            
            if [ -f "$checkpoint" ]; then
                evaluate_checkpoint "clevrtex" "$model" "$batch" "$epoch" \
                    "$checkpoint" "$CLEVRTEX_CONFIG" "$CLEVRTEX_CHECKPOINT"
                ((completed_experiments++))
                echo "    Progress: ${completed_experiments}/${total_experiments}"
            else
                echo "  ⚠️  Checkpoint not found: epoch ${epoch}"
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "Evaluation completed!"
echo "========================================================"
echo "Processed: ${completed_experiments}/${total_experiments} experiments"
echo ""
echo "========================================================"
echo "Running loss analysis..."
echo "========================================================"
echo ""

python evalweek.py

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
echo ""
echo "Results saved to:"
echo "  ${OUTPUT_DIR}/"
echo ""
echo "Directory structure:"
echo "  evalweek/"
echo "  ├── visualizations/"
echo "  │   ├── coco/"
echo "  │   │   ├── linear/"
echo "  │   │   │   ├── batch64_epoch01.png"
echo "  │   │   │   ├── batch64_epoch05.png"
echo "  │   │   │   ├── batch64_epoch15.png"
echo "  │   │   │   ├── batch64_final.png"
echo "  │   │   │   ├── batch256_epoch01.png"
echo "  │   │   │   └── ... (12 images per model)"
echo "  │   │   └── ... (all 5 models)"
echo "  │   └── clevrtex/"
echo "  │       └── ... (all 5 models)"
echo "  ├── loss_curves/"
echo "  │   ├── coco_all_models.png"
echo "  │   ├── clevrtex_all_models.png"
echo "  │   ├── batch_comparison.png"
echo "  │   └── model_comparison.png"
echo "  ├── loss_comparison.csv"
echo "  └── loss_history_full.csv"
echo ""
