"""
Slot Autoencoder Evaluation Script (v3 - with Slot Attention Refinement)
새로운 slot을 만든 후, slot attention을 n iterations 반복하여 slot을 refine
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

sys.path.append('/home/jaey00ns/MetaSlot-main')
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config


# ==================== Hyperparameters ====================
NUM_SLOT_ATTENTION_ITERS = 3  # Slot attention refinement iterations

# ==================== Autoencoder Models ====================

class LinearSlotAutoencoder(nn.Module):
    """간단한 선형 변환 autoencoder (slot만)"""
    def __init__(self, slot_dim):
        super().__init__()
        self.encoder = nn.Linear(slot_dim * 2, slot_dim)
        self.decoder = nn.Linear(slot_dim, slot_dim * 2)
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)
        return self.encoder(combined)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.decoder(encoded_slot)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon


class NonlinearSlotAutoencoder(nn.Module):
    """비선형 MLP autoencoder (slot만)"""
    def __init__(self, slot_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim * 2),
        )
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)
        return self.encoder(combined)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.decoder(encoded_slot)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon


# ==================== Slot Attention Refinement ====================

def refine_slots_with_aggregat(features, new_slots, model):
    """
    기존 features로 새로운 slots만 aggregat에 통과
    
    Args:
        features: (B, H*W, C) encoded features (이미 계산된 것 재사용)
        new_slots: (B, num_slots, slot_dim) new slot initialization
        model: MetaSlot model
    
    Returns:
        refined_slots: (B, num_slots, slot_dim)
        attention: (B, num_slots, H, W) attention maps
    """
    # aggregat만 실행 (features는 재사용)
    refined_slots, attention = model.m.aggregat(features, new_slots)
    
    # attention shape: (B, num_slots, H*W) -> (B, num_slots, H, W)
    B = features.shape[0]
    H = W = 16  # MetaSlot feature resolution
    attention = attention.view(B, -1, H, W)
    
    return refined_slots, attention


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
    """Winner-take-all 방식으로 슬롯들을 시각화 (ipynb 방식)"""
    H, W = attention_maps.shape[1:]
    num_slots = attention_maps.shape[0]
    
    orig_resized = cv2.resize(original_image, (W, H))
    
    # Winner-take-all: 각 픽셀에서 가장 높은 attention을 가진 slot
    max_slot_indices = np.argmax(attention_maps, axis=0)  # (H, W)
    
    # 세그멘테이션 마스크 생성
    segmentation_mask = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(num_slots):
        slot_mask = (max_slot_indices == i)
        color = all_colors[slot_indices[i]]
        for c in range(3):
            segmentation_mask[slot_mask, c] = color[c]
    
    # 원본과 블렌딩 (ipynb와 동일한 방식)
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

def visualize_autoencoder_results(original_image, slots, features, metaslot_model, autoencoder,
                                  original_attention_output, encoder_pair=(0, 1), decoder_idx=3, save_path=None):
    """
    Autoencoder + Slot Attention Refinement 결과 시각화
    """
    device = slots.device
    num_slots = slots.shape[0]
    
    # Generate consistent colors
    max_possible_slots = num_slots + 2
    all_colors = generate_slot_colors(max_possible_slots)
    
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
    slot1 = slots[idx1:idx1+1]
    slot2 = slots[idx2:idx2+1]
    
    with pt.no_grad():
        encoded_slot = autoencoder.encode(slot1, slot2)
        
        # Create new slot set
        merged_slots = pt.cat([slots[remaining_indices], encoded_slot], dim=0).unsqueeze(0)
        
        # Refine with slot attention
        print(f"\n[INFO] Refining {merged_slots.shape[1]} merged slots...")
        refined_slots, refined_attention = refine_slots_with_aggregat(
            features, merged_slots, metaslot_model
        )
        
        # Upsample attention to 256x256
        refined_attention_256 = F.interpolate(
            refined_attention, 
            size=(256, 256), 
            mode='bilinear'
        )
        refined_attention_256 = refined_attention_256.squeeze(0).cpu().numpy()
        
        # Debug: 원본 vs refined attention 비교
        print(f"[DEBUG] Original attention shape: {original_attention_output.shape}")
        print(f"[DEBUG] Refined attention shape: {refined_attention_256.shape}")
        print(f"[DEBUG] Original attention stats - min: {original_attention_output.min():.6f}, max: {original_attention_output.max():.6f}")
        print(f"[DEBUG] Refined attention stats - min: {refined_attention_256.min():.6f}, max: {refined_attention_256.max():.6f}")
        
        # 특정 slot의 attention을 출력
        print(f"[DEBUG] Original slot 0 - unique values: {len(np.unique(original_attention_output[0]))}")
        print(f"[DEBUG] Refined merged slot - unique values: {len(np.unique(refined_attention_256[-1]))}")
    
    merged_slot_indices = remaining_indices + [num_slots]
    img_02 = visualize_slots_with_mask(original_image, refined_attention_256,
                                        merged_slot_indices, all_colors, alpha=0.5)
    
    # ==================== Row 2: Decoder (Split) ====================
    
    original_slot_indices = list(range(num_slots))
    img_11 = visualize_slots_with_mask(original_image, original_attention_output,
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_10 = add_legend_to_image(original_image, all_colors,
                                  split_index=decoder_idx,
                                  output_indices=[num_slots, num_slots+1])
    
    # Decode slot
    slot_to_split = slots[decoder_idx:decoder_idx+1]
    
    with pt.no_grad():
        slot_recon1, slot_recon2 = autoencoder.decode(slot_to_split)
        
        # Create new slot set
        remaining_indices_split = [i for i in range(num_slots) if i != decoder_idx]
        split_slots = pt.cat([slots[remaining_indices_split], slot_recon1, slot_recon2], dim=0).unsqueeze(0)
        
        # Refine with slot attention
        print(f"[INFO] Refining {split_slots.shape[1]} split slots...")
        refined_slots_split, refined_attention_split = refine_slots_with_aggregat(
            features, split_slots, metaslot_model
        )
        
        # Upsample attention to 256x256
        refined_attention_split_256 = F.interpolate(
            refined_attention_split, 
            size=(256, 256), 
            mode='bilinear'
        )
        refined_attention_split_256 = refined_attention_split_256.squeeze(0).cpu().numpy()
    
    split_slot_indices = remaining_indices_split + [num_slots, num_slots+1]
    img_12 = visualize_slots_with_mask(original_image, refined_attention_split_256,
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
    axes[0, 2].set_title(f'After Merge: 6 slots\n(Refined with {NUM_SLOT_ATTENTION_ITERS} iters)', 
                        fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_10)
    axes[1, 0].set_title('Original Image\n(with Split legend)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_11)
    axes[1, 1].set_title(f'7 Slots (Before Split)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_12)
    axes[1, 2].set_title(f'After Split: 8 slots\n(Refined with {NUM_SLOT_ATTENTION_ITERS} iters)', 
                        fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    
    plt.close()


def evaluate_autoencoder(model_path, test_image_paths, num_samples=4):
    """학습된 autoencoder 평가 (slot attention refinement 포함)"""
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ==================== Load Models ====================
    print("\n[1/4] Loading models...")
    
    cfg_file = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py"
    ckpt_file = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth"
    
    cfg = Config.fromfile(cfg_file)
    metaslot_model = build_from_config(cfg.model)
    metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
    
    state = pt.load(ckpt_file, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    metaslot_model.load_state_dict(state, strict=False)
    metaslot_model = metaslot_model.to(device).eval()
    
    # Load Autoencoder
    checkpoint = pt.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint['model_type']
    slot_dim = checkpoint['slot_dim']
    
    if model_type == 'linear':
        autoencoder = LinearSlotAutoencoder(slot_dim=slot_dim)
    else:
        hidden_dim = checkpoint.get('hidden_dim', 512)
        autoencoder = NonlinearSlotAutoencoder(slot_dim=slot_dim, hidden_dim=hidden_dim)
    
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder = autoencoder.to(device).eval()
    
    print(f"✓ Loaded {model_type} autoencoder")
    print(f"✓ Slot attention refinement: {NUM_SLOT_ATTENTION_ITERS} iterations")
    
    # ==================== Setup Output Directory ====================
    model_name = Path(model_path).stem
    output_dir = Path("/home/jaey00ns/MetaSlot-main/slotae/eval3")
    output_dir.mkdir(exist_ok=True)
    
    # ==================== Process Images ====================
    print(f"\n[2/4] Processing {min(num_samples, len(test_image_paths))} test images...")
    
    for img_idx, image_path in enumerate(test_image_paths[:num_samples]):
        print(f"\n  Processing image {img_idx + 1}/{num_samples}: {image_path}")
        
        image_tensor, original_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        with pt.no_grad():
            batch = {'image': image_tensor}
            
            # Encode image
            features = metaslot_model.m.encode_backbone(image_tensor)
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
            
            features = metaslot_model.m.encode_posit_embed(features)
            features = metaslot_model.m.encode_project(features)
            
            # Get slots
            output = metaslot_model(batch)
            slots = output['slotz'].squeeze(0)
            
            # Get original attention
            if 'attent2' in output:
                original_attention = output['attent2']
            elif 'attent' in output:
                original_attention = output['attent']
            else:
                raise ValueError("No attention output found")
            
            original_attention_256 = F.interpolate(original_attention, size=(256, 256), mode='bilinear')
            original_attention_256 = original_attention_256.squeeze(0).cpu().numpy()
        
        original_resized = cv2.resize(original_image, (256, 256))
        
        encoder_pair = (0, 1)
        decoder_idx = 3
        
        save_path = output_dir / f"{model_name}_img{img_idx:02d}.png"
        
        visualize_autoencoder_results(
            original_resized,
            slots,
            features,
            metaslot_model,
            autoencoder,
            original_attention_256,
            encoder_pair=encoder_pair,
            decoder_idx=decoder_idx,
            save_path=save_path
        )
    
    print(f"\n[3/4] ✓ All visualizations saved to {output_dir}")
    print(f"\n[4/4] Evaluation completed!")


if __name__ == "__main__":
    MODEL_TYPE = "linear"
    
    model_path = f"/home/jaey00ns/MetaSlot-main/slotae/pth/linear/20260326_055826/final.pth"
    
    test_images = [
        "/home/jaey00ns/MetaSlot-main/imgs/slottest.png",
    ]
    
    if not os.path.exists(test_images[0]):
        print(f"Warning: Test image not found at {test_images[0]}")
        coco_val_dir = Path("/home/jaey00ns/coco_original/val2017")
        if coco_val_dir.exists():
            coco_images = list(coco_val_dir.glob("*.jpg"))[:4]
            if coco_images:
                test_images = [str(p) for p in coco_images]
                print(f"Using COCO validation images: {len(test_images)} images")
    
    print("="*60)
    print("Slot Autoencoder Evaluation (v3 - with Slot Attention)")
    print("="*60)
    print(f"Model: {MODEL_TYPE}")
    print(f"Model path: {model_path}")
    print(f"Test images: {len(test_images)}")
    print(f"Refinement iterations: {NUM_SLOT_ATTENTION_ITERS}")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the model first using: python trainae2.py")
        print("Then update the model_path with the correct timestamp")
    else:
        evaluate_autoencoder(
            model_path=model_path,
            test_image_paths=test_images,
            num_samples=4
        )
