"""
Slot Autoencoder Evaluation Script (v4 - with Decoder attent2)
새로운 slot을 만든 후, Decoder를 직접 통과시켜 고해상도 attent2 획득
evalae3의 slot attention refinement 대신 decoder의 mask logit 활용
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config

# Import autoencoder models
from models import create_autoencoder, list_available_models, MODEL_CONFIGS


# ==================== Hyperparameters ====================

# Model Selection
MODEL_CONFIG = 'linear'
MODEL_PATH = "/home/jaey00ns/MetaSlot-main/slotae/pth_coco/linear/20260327_072459/linear_batch64_final.pth"

# MetaSlot Config
METASLOT_CONFIG = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py"
METASLOT_CHECKPOINT = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth"

# Evaluation Settings
NUM_SAMPLES = 1
ENCODER_PAIR = (0, 6)  # merge할 slot pair
DECODER_IDX = 4  # split할 slot index

# Test Images
TEST_IMAGES = [
    "/home/jaey00ns/MetaSlot-main/imgs/clevr2.png",
]

# Output
OUTPUT_DIR = "/home/jaey00ns/MetaSlot-main/slotae/eval4"


# ==================== Decoder attent2 Generation ====================

def generate_attent2_from_slots(new_slots, decoder):
    """
    Decoder를 통해 slots로부터 고해상도 attent2 직접 생성
    
    Args:
        new_slots: (B, num_slots, slot_dim) new slots
        decoder: BroadcastMLPDecoder module
    
    Returns:
        attent2: (B, num_slots, 256, 256) high-resolution attention maps
        recon: (B, H*W, C) reconstruction (optional)
    """
    # Decoder forward
    clue = [16, 16]  # MetaSlot feature resolution (h, w)
    recon, attent2 = decoder(clue, new_slots)
    
    # attent2 shape: (B, N, 16*16=256) - feature map resolution
    # Reshape to (B, N, 16, 16) then upsample to (B, N, 256, 256)
    B, N, HW = attent2.shape
    h = w = 16  # Feature map resolution
    
    # Reshape to 2D
    attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)
    
    # Upsample to 256x256 for visualization
    attent2 = F.interpolate(attent2, size=(256, 256), mode='bilinear', align_corners=False)
    
    return attent2, recon


# ==================== Utility Functions ====================

IMAGENET_MEAN = np.array([[[123.675]], [[116.28]], [[103.53]]], dtype=np.float32)
IMAGENET_STD = np.array([[[58.395]], [[57.12]], [[57.375]]], dtype=np.float32)


def preprocess_image(image_path, target_size=(256, 256)):
    """이미지를 MetaSlot 입력 형태로 전처리"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    original = np.array(image)
    
    # Center crop and resize
    width, height = image.size
    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    
    image = image.crop((left, top, right, bottom))
    image = image.resize(target_size, Image.BILINEAR)
    
    # Normalize
    image_np = np.array(image, dtype=np.float32)
    image_np = image_np.transpose(2, 0, 1)
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    
    image_tensor = pt.from_numpy(image_np).unsqueeze(0).float()
    
    return image_tensor, original


def generate_slot_colors(num_slots):
    """슬롯별 고정 색상 생성"""
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
    """Winner-take-all 방식으로 슬롯들을 시각화"""
    H, W = attention_maps.shape[1:]
    num_slots = attention_maps.shape[0]
    
    orig_resized = cv2.resize(original_image, (W, H))
    
    # Winner-take-all
    max_slot_indices = np.argmax(attention_maps, axis=0)
    
    # 세그멘테이션 마스크 생성
    segmentation_mask = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(num_slots):
        slot_mask = (max_slot_indices == i)
        color = all_colors[slot_indices[i]]
        for c in range(3):
            segmentation_mask[slot_mask, c] = color[c]
    
    # 블렌딩
    overlay_image = orig_resized.astype(np.float32) / 255.0
    blended = (1 - alpha) * overlay_image + alpha * segmentation_mask
    vis_image = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return vis_image


def add_legend_to_image(image, slot_colors, merge_indices=None, split_index=None, output_indices=None):
    """원본 이미지 우상단에 legend 추가"""
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


# ==================== Main Visualization Function ====================

def visualize_autoencoder_results(original_image, slots, decoder, autoencoder,
                                  original_attention_output, encoder_pair=(0, 1), decoder_idx=3, save_path=None):
    """
    Autoencoder + Decoder attent2 결과 시각화
    """
    device = slots.device
    num_slots = slots.shape[0]
    
    # Generate consistent colors
    max_possible_slots = num_slots + 2
    all_colors = generate_slot_colors(max_possible_slots)
    
    print(f"\n{'='*60}")
    print(f"Processing visualization...")
    print(f"  Original slots: {num_slots}")
    print(f"  Encoder pair: {encoder_pair}")
    print(f"  Decoder index: {decoder_idx}")
    print(f"{'='*60}\n")
    
    # ==================== Row 1: Encoder (Merge) ====================
    
    idx1, idx2 = encoder_pair
    remaining_indices = [i for i in range(num_slots) if i not in encoder_pair]
    
    # Original slots visualization
    original_slot_indices = list(range(num_slots))
    img_01 = visualize_slots_with_mask(original_image, original_attention_output, 
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_00 = add_legend_to_image(original_image, all_colors, 
                                  merge_indices=encoder_pair, 
                                  output_indices=[num_slots])
    
    # Encode slots
    print(f"[1/2] Merging slots {encoder_pair}...")
    slot1 = slots[idx1:idx1+1]
    slot2 = slots[idx2:idx2+1]
    
    with pt.no_grad():
        encoded_slot = autoencoder.encode(slot1, slot2)
        
        # Create new slot set
        merged_slots = pt.cat([slots[remaining_indices], encoded_slot], dim=0).unsqueeze(0)
        
        # 🎯 핵심: Decoder를 통해 attent2 직접 생성!
        print(f"  → Generating attent2 via decoder for {merged_slots.shape[1]} merged slots...")
        print(f"  → merged_slots shape: {merged_slots.shape}")
        attent2_merged, recon_merged = generate_attent2_from_slots(merged_slots, decoder)
        
        # attent2는 이미 (B, N, 256, 256) 형태
        attent2_merged = attent2_merged.squeeze(0).cpu().numpy()
        
        print(f"  → attent2_merged shape: {attent2_merged.shape}")
        print(f"  → attent2 stats - min: {attent2_merged.min():.6f}, max: {attent2_merged.max():.6f}, mean: {attent2_merged.mean():.6f}")
        print(f"  → attent2 sum per slot: {[attent2_merged[i].sum()/(256*256) for i in range(attent2_merged.shape[0])]}")

    
    merged_slot_indices = remaining_indices + [num_slots]
    img_02 = visualize_slots_with_mask(original_image, attent2_merged,
                                        merged_slot_indices, all_colors, alpha=0.5)
    
    # ==================== Row 2: Decoder (Split) ====================
    
    original_slot_indices = list(range(num_slots))
    img_11 = visualize_slots_with_mask(original_image, original_attention_output,
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_10 = add_legend_to_image(original_image, all_colors,
                                  split_index=decoder_idx,
                                  output_indices=[num_slots, num_slots+1])
    
    # Decode slot
    print(f"\n[2/2] Splitting slot {decoder_idx}...")
    slot_to_split = slots[decoder_idx:decoder_idx+1]
    
    with pt.no_grad():
        slot_recon1, slot_recon2 = autoencoder.decode(slot_to_split)
        
        # Create new slot set
        remaining_indices_split = [i for i in range(num_slots) if i != decoder_idx]
        split_slots = pt.cat([slots[remaining_indices_split], slot_recon1, slot_recon2], dim=0).unsqueeze(0)
        
        # 🎯 핵심: Decoder를 통해 attent2 직접 생성!
        print(f"  → Generating attent2 via decoder for {split_slots.shape[1]} split slots...")
        print(f"  → split_slots shape: {split_slots.shape}")
        attent2_split, recon_split = generate_attent2_from_slots(split_slots, decoder)
        
        attent2_split = attent2_split.squeeze(0).cpu().numpy()
        
        print(f"  → attent2_split shape: {attent2_split.shape}")
        print(f"  → attent2 stats - min: {attent2_split.min():.6f}, max: {attent2_split.max():.6f}, mean: {attent2_split.mean():.6f}")
        print(f"  → attent2 sum per slot: {[attent2_split[i].sum()/(256*256) for i in range(attent2_split.shape[0])]}")

    
    split_slot_indices = remaining_indices_split + [num_slots, num_slots+1]
    img_12 = visualize_slots_with_mask(original_image, attent2_split,
                                        split_slot_indices, all_colors, alpha=0.5)
    
    # ==================== Create 2x3 Grid ====================
    
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
        print(f"\n✓ Saved visualization to {save_path}")
    
    plt.close()


def evaluate_autoencoder(model_path, test_image_paths, num_samples=4):
    """학습된 autoencoder 평가 (decoder attent2 방식)"""
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ==================== Load Models ====================
    print("\n[1/4] Loading models...")
    
    cfg = Config.fromfile(METASLOT_CONFIG)
    metaslot_model = build_from_config(cfg.model)
    metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
    
    state = pt.load(METASLOT_CHECKPOINT, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    metaslot_model.load_state_dict(state, strict=False)
    metaslot_model = metaslot_model.to(device).eval()
    
    # Extract decoder module
    decoder = metaslot_model.m.decode
    
    # Load Autoencoder
    checkpoint = pt.load(model_path, map_location=device, weights_only=False)
    
    # Handle old checkpoints without model_config (fallback to MODEL_CONFIG global)
    if 'model_config' not in checkpoint:
        print(f"⚠️  Warning: 'model_config' not found in checkpoint, using MODEL_CONFIG={MODEL_CONFIG}")
        model_config = MODEL_CONFIG
    else:
        model_config = checkpoint['model_config']
    
    slot_dim = checkpoint.get('slot_dim', 256)
    
    autoencoder = create_autoencoder(model_config, slot_dim=slot_dim)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder = autoencoder.to(device).eval()
    
    print(f"✓ Loaded {model_config} autoencoder")
    print(f"✓ Using Decoder attent2 (high-resolution mask logit)")
    
    # ==================== Setup Output Directory ====================
    model_name = Path(model_path).stem
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # ==================== Process Images ====================
    print(f"\n[2/4] Processing {min(num_samples, len(test_image_paths))} test images...")
    
    for img_idx, image_path in enumerate(test_image_paths[:num_samples]):
        print(f"\n{'='*60}")
        print(f"Image {img_idx + 1}/{num_samples}: {Path(image_path).name}")
        print(f"{'='*60}")
        
        image_tensor, original_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        with pt.no_grad():
            batch = {'image': image_tensor}
            
            # Get slots and original attention
            output = metaslot_model(batch)
            slots = output['slotz'].squeeze(0)
            
            # Get original attent2 (prefer attent2 over attent)
            if 'attent2' in output:
                original_attention = output['attent2']  # (B, N, H, W)
                print(f"  Using attent2: {original_attention.shape}")
            elif 'attent' in output:
                original_attention = output['attent']  # (B, N, 16, 16)
                print(f"  Using attent: {original_attention.shape}")
            else:
                raise ValueError("No attention output found")
            
            # Always upsample to 256x256 for consistent visualization
            if original_attention.shape[-1] != 256:
                print(f"  Upsampling from {original_attention.shape[-2:]} to (256, 256)")
                original_attention = F.interpolate(original_attention, size=(256, 256), mode='bilinear', align_corners=False)
            
            original_attention_256 = original_attention.squeeze(0).cpu().numpy()
            print(f"  Final original_attention shape: {original_attention_256.shape}")
        
        original_resized = cv2.resize(original_image, (256, 256))
        
        save_path = output_dir / f"{model_name}_img{img_idx:02d}.png"
        
        visualize_autoencoder_results(
            original_resized,
            slots,
            decoder,
            autoencoder,
            original_attention_256,
            encoder_pair=ENCODER_PAIR,
            decoder_idx=DECODER_IDX,
            save_path=save_path
        )
    
    print(f"\n{'='*60}")
    print(f"[3/4] ✓ All visualizations saved to {output_dir}")
    print(f"[4/4] Evaluation completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # COCO fallback 체크
    if not os.path.exists(TEST_IMAGES[0]):
        print(f"Warning: Test image not found at {TEST_IMAGES[0]}")
        coco_val_dir = Path("/home/jaey00ns/coco_original/val2017")
        if coco_val_dir.exists():
            coco_images = list(coco_val_dir.glob("*.jpg"))[:NUM_SAMPLES]
            if coco_images:
                TEST_IMAGES = [str(p) for p in coco_images]
                print(f"Using COCO validation images: {len(TEST_IMAGES)} images")
    
    print("="*60)
    print("Slot Autoencoder Evaluation (v4 - Decoder attent2)")
    print("="*60)
    print(f"Model Config: {MODEL_CONFIG}")
    print(f"Model Description: {MODEL_CONFIGS[MODEL_CONFIG]['description']}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Images: {len(TEST_IMAGES)}")
    print(f"Method: Direct decoder attent2 generation")
    print(f"Encoder Pair: {ENCODER_PAIR}")
    print(f"Decoder Index: {DECODER_IDX}")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python trainae2.py")
        print("Then update MODEL_PATH with the correct timestamp")
    else:
        evaluate_autoencoder(
            model_path=MODEL_PATH,
            test_image_paths=TEST_IMAGES,
            num_samples=NUM_SAMPLES
        )
