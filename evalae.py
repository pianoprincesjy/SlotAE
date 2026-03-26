"""
Slot Autoencoder Evaluation Script
학습된 autoencoder를 시각화하여 질적 평가
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import cv2

sys.path.append('/home/jaey00ns/MetaSlot-main')
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config


# ==================== Autoencoder Models (same as trainae.py) ====================

class LinearSlotAutoencoder(nn.Module):
    """간단한 선형 변환 autoencoder"""
    def __init__(self, slot_dim=256, attention_size=256*256):
        super().__init__()
        self.slot_encoder = nn.Linear(slot_dim * 2, slot_dim)
        self.slot_decoder = nn.Linear(slot_dim, slot_dim * 2)
        self.attention_decoder = nn.Linear(attention_size, attention_size * 2)
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)
        return self.slot_encoder(combined)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.slot_decoder(encoded_slot)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon
    
    def decode_attention(self, attention):
        decoded = self.attention_decoder(attention)
        mid = decoded.shape[-1] // 2
        attention1 = decoded[..., :mid]
        attention2 = decoded[..., mid:]
        attention1 = pt.softmax(attention1.view(-1, attention1.shape[-1]), dim=-1)
        attention2 = pt.softmax(attention2.view(-1, attention2.shape[-1]), dim=-1)
        return attention1, attention2


class NonlinearSlotAutoencoder(nn.Module):
    """비선형 MLP autoencoder"""
    def __init__(self, slot_dim=256, hidden_dim=512, attention_size=256*256):
        super().__init__()
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
        self.slot_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim * 2),
        )
        
        self.attention_decoder = nn.Sequential(
            nn.Linear(attention_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_size * 2),
        )
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)
        return self.slot_encoder(combined)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.slot_decoder(encoded_slot)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon
    
    def decode_attention(self, attention):
        decoded = self.attention_decoder(attention)
        mid = decoded.shape[-1] // 2
        attention1 = decoded[..., :mid]
        attention2 = decoded[..., mid:]
        attention1 = pt.softmax(attention1.view(-1, attention1.shape[-1]), dim=-1)
        attention2 = pt.softmax(attention2.view(-1, attention2.shape[-1]), dim=-1)
        return attention1, attention2


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
    image_np = image_np.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    
    image_tensor = pt.from_numpy(image_np).unsqueeze(0).float()
    
    return image_tensor, original


def add_legend_to_image(image, slot_colors, merge_indices=None, split_index=None, output_indices=None):
    """
    원본 이미지 우상단에 legend 추가
    
    Args:
        image: (H, W, 3) numpy array
        slot_colors: (num_slots, 3) array of RGB colors
        merge_indices: merge할 슬롯 인덱스들 (tuple)
        split_index: split할 슬롯 인덱스 (int)
        output_indices: 결과 슬롯 인덱스들 (list)
    """
    img = image.copy()
    H, W = img.shape[:2]
    
    # Legend parameters
    box_size = 25
    padding = 15
    spacing = 8
    text_height = 20
    
    y_offset = padding
    x_start = W - padding - 200
    
    if merge_indices is not None:
        # "Merge:" label
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
        
        # Arrow
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x_pos += 30
        
        # Output color (averaged)
        if output_indices and len(output_indices) > 0:
            out_idx = output_indices[0]
            out_color = (slot_colors[out_idx] * 255).astype(np.uint8).tolist()
            cv2.rectangle(img, (x_pos, y_offset), 
                         (x_pos + box_size, y_offset + box_size), out_color, -1)
            cv2.rectangle(img, (x_pos, y_offset), 
                         (x_pos + box_size, y_offset + box_size), (0, 0, 0), 2)
    
    elif split_index is not None:
        # "Split:" label
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
        
        # Arrow
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, "->", (x_pos, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x_pos += 30
        
        # Output colors (2 new slots)
        if output_indices:
            for out_idx in output_indices[:2]:
                out_color = (slot_colors[out_idx] * 255).astype(np.uint8).tolist()
                cv2.rectangle(img, (x_pos, y_offset), 
                             (x_pos + box_size, y_offset + box_size), out_color, -1)
                cv2.rectangle(img, (x_pos, y_offset), 
                             (x_pos + box_size, y_offset + box_size), (0, 0, 0), 2)
                x_pos += box_size + spacing
    
    return img


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
    """
    Winner-take-all 방식으로 슬롯들을 뚜렷하게 시각화
    
    Args:
        original_image: (H, W, 3) numpy array
        attention_maps: (num_slots, H, W) numpy array
        slot_indices: 각 attention map이 어떤 원본 슬롯 인덱스인지 (list)
        all_colors: 전체 슬롯 색상 배열 (max_slots, 3)
        alpha: blending alpha value
    """
    H, W = attention_maps.shape[1:]
    num_slots = attention_maps.shape[0]
    
    # Resize original image to match attention map size
    orig_resized = cv2.resize(original_image, (W, H))
    
    # Winner-take-all segmentation
    max_slot_indices = np.argmax(attention_maps, axis=0)
    
    # 세그멘테이션 마스크 생성 (원본 색상 유지)
    segmentation_mask = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(num_slots):
        slot_mask = (max_slot_indices == i)
        # 원본 슬롯 인덱스의 색상 사용
        color = all_colors[slot_indices[i]]
        for c in range(3):
            segmentation_mask[slot_mask, c] = color[c]
    
    # 원본과 블렌딩
    overlay_image = orig_resized.astype(np.float32) / 255.0
    blended = overlay_image * (1 - alpha) + segmentation_mask * alpha
    vis_image = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return vis_image


def visualize_autoencoder_results(original_image, slots, attention_maps, autoencoder, 
                                  encoder_pair=(0, 1), decoder_idx=3, save_path=None):
    """
    3x2 그리드로 autoencoder 결과 시각화
    
    Row 1:
    - (0,0): 원본 이미지
    - (0,1): 7개 슬롯 시각화 (encoder_pair 별표)
    - (0,2): encoder로 합쳐진 6개 슬롯 시각화
    
    Row 2:
    - (1,0): 원본 이미지
    - (1,1): 7개 슬롯 시각화 (decoder_idx 별표)
    - (1,2): decoder로 나누어진 8개 슬롯 시각화
    
    Args:
        original_image: (H, W, 3) numpy array
        slots: (num_slots, slot_dim) tensor
        attention_maps: (num_slots, H, W) numpy array
        autoencoder: trained autoencoder model
        encoder_pair: tuple of (idx1, idx2) to merge
        decoder_idx: slot index to split
        save_path: path to save the visualization
    """
    device = slots.device
    num_slots = slots.shape[0]
    
    # Generate consistent colors for all slots (including potential new ones)
    max_possible_slots = num_slots + 2  # Original + potential split outputs
    all_colors = generate_slot_colors(max_possible_slots)
    
    # ==================== Row 1: Encoder Test ====================
    
    # Encode: merge two slots
    idx1, idx2 = encoder_pair
    
    # Original slots with all original indices
    original_slot_indices = list(range(num_slots))
    img_01 = visualize_slots_with_mask(original_image, attention_maps, 
                                        original_slot_indices, all_colors, alpha=0.5)
    
    # Add legend to ORIGINAL image
    img_00 = add_legend_to_image(original_image, all_colors, 
                                  merge_indices=encoder_pair, 
                                  output_indices=[num_slots])  # New merged slot gets next index
    
    slot1 = slots[idx1:idx1+1]
    slot2 = slots[idx2:idx2+1]
    attention1 = attention_maps[idx1]  # (H, W)
    attention2 = attention_maps[idx2]
    
    with pt.no_grad():
        encoded_slot = autoencoder.encode(slot1, slot2)
        
        # Merge attentions (sum and normalize)
        H, W = attention1.shape
        merged_attention_np = attention1 + attention2
        merged_attention_np = merged_attention_np / (merged_attention_np.sum() + 1e-8)
    
    # Create new slot set with encoded slot
    remaining_indices = [i for i in range(num_slots) if i not in encoder_pair]
    merged_slots = pt.cat([slots[remaining_indices], encoded_slot], dim=0)
    
    # For visualization: include merged attention
    merged_attention_all = np.concatenate([
        attention_maps[remaining_indices],
        merged_attention_np[np.newaxis, :, :]  # Add merged attention
    ], axis=0)
    merged_slot_indices = remaining_indices + [num_slots]
    img_02 = visualize_slots_with_mask(original_image, merged_attention_all, 
                                        merged_slot_indices, all_colors, alpha=0.5)
    
    # ==================== Row 2: Decoder Test ====================
    
    # Original slots
    img_11 = visualize_slots_with_mask(original_image, attention_maps,
                                        original_slot_indices, all_colors, alpha=0.5)
    
    # Add legend to ORIGINAL image for split
    img_10 = add_legend_to_image(original_image, all_colors,
                                  split_index=decoder_idx,
                                  output_indices=[num_slots, num_slots+1])  # Two new slots
    
    # Decode: split one slot into two
    slot_to_split = slots[decoder_idx:decoder_idx+1]
    attention_to_split = attention_maps[decoder_idx]  # (H, W)
    
    with pt.no_grad():
        slot_recon1, slot_recon2 = autoencoder.decode(slot_to_split)
        
        # Split attention using autoencoder
        H, W = attention_to_split.shape
        attention_flat = pt.from_numpy(attention_to_split).flatten().unsqueeze(0).to(device)  # (1, H*W)
        attention_split1, attention_split2 = autoencoder.decode_attention(attention_flat)
        
        # Reshape back to spatial
        attention_split1_np = attention_split1.squeeze(0).cpu().numpy().reshape(H, W)
        attention_split2_np = attention_split2.squeeze(0).cpu().numpy().reshape(H, W)
    
    # Create new slot set with split slots
    remaining_indices = [i for i in range(num_slots) if i != decoder_idx]
    split_slots = pt.cat([slots[remaining_indices], slot_recon1, slot_recon2], dim=0)
    
    # For visualization: remaining slots + 2 new decoded attentions
    split_attention_all = np.concatenate([
        attention_maps[remaining_indices],
        attention_split1_np[np.newaxis, :, :],
        attention_split2_np[np.newaxis, :, :]
    ], axis=0)
    
    # Slot indices: remaining keep original, new ones get num_slots, num_slots+1
    split_slot_indices = remaining_indices + [num_slots, num_slots+1]
    img_12 = visualize_slots_with_mask(original_image, split_attention_all,
                                        split_slot_indices, all_colors, alpha=0.5)
    
    # ==================== Create 3x2 Grid ====================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(img_00)
    axes[0, 0].set_title('Original Image\n(with Merge legend)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_01)
    axes[0, 1].set_title(f'7 Slots (Before Merge)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_02)
    axes[0, 2].set_title(f'After Merge: 6 slots\n(Slots {idx1},{idx2} merged with summed attention)', 
                        fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(img_10)
    axes[1, 0].set_title('Original Image\n(with Split legend)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_11)
    axes[1, 1].set_title(f'7 Slots (Before Split)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_12)
    axes[1, 2].set_title(f'After Split: 8 slots\n(Slot {decoder_idx} split with decoded attention)', 
                        fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


# ==================== Main Evaluation Function ====================

def evaluate_autoencoder(model_path, test_image_paths, num_samples=4):
    """
    학습된 autoencoder 평가
    
    Args:
        model_path: 학습된 모델 경로 (.pth file)
        test_image_paths: 테스트 이미지 경로 리스트
        num_samples: 평가할 이미지 수
    """
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ==================== Load Models ====================
    print("\n[1/4] Loading models...")
    
    # Load MetaSlot
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
        autoencoder = NonlinearSlotAutoencoder(slot_dim=slot_dim, hidden_dim=512)
    
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder = autoencoder.to(device).eval()
    
    print(f"✓ Loaded {model_type} autoencoder")
    
    # ==================== Setup Output Directory ====================
    model_name = Path(model_path).stem  # e.g., "slotae_linear"
    output_dir = Path("/home/jaey00ns/MetaSlot-main/slotae/eval")
    output_dir.mkdir(exist_ok=True)
    
    # ==================== Process Images ====================
    print(f"\n[2/4] Processing {min(num_samples, len(test_image_paths))} test images...")
    
    for img_idx, image_path in enumerate(test_image_paths[:num_samples]):
        print(f"\n  Processing image {img_idx + 1}/{num_samples}: {image_path}")
        
        # Preprocess
        image_tensor, original_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Get slots from MetaSlot
        with pt.no_grad():
            batch = {'image': image_tensor}
            output = metaslot_model(batch)
            slots = output['slotz'].squeeze(0)  # (num_slots, slot_dim)
            
            # Get attention maps
            if 'attent2' in output:
                attention = output['attent2']
            else:
                attention = output['attent']
            
            attention = F.interpolate(
                attention, 
                size=(256, 256), 
                mode='bilinear'
            ).squeeze(0).cpu().numpy()  # (num_slots, 256, 256)
        
        # Resize original image to 256x256
        original_resized = cv2.resize(original_image, (256, 256))
        
        # Visualize with different encoder/decoder pairs
        encoder_pair = (0, 1)  # merge slot 0 and 1
        decoder_idx = 3  # split slot 3
        
        save_path = output_dir / f"{model_name}_img{img_idx:02d}.png"
        
        visualize_autoencoder_results(
            original_resized,
            slots,
            attention,
            autoencoder,
            encoder_pair=encoder_pair,
            decoder_idx=decoder_idx,
            save_path=save_path
        )
    
    print(f"\n[3/4] ✓ All visualizations saved to {output_dir}")
    print(f"\n[4/4] Evaluation completed!")
    print(f"\n{'='*60}")
    print(f"Results saved in: {output_dir}")
    print(f"Model evaluated: {model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ==================== Configuration ====================
    
    # Choose which model to evaluate
    MODEL_TYPE = "linear"  # "linear" or "nonlinear"
    
    model_path = f"/home/jaey00ns/MetaSlot-main/slotae/slotae_{MODEL_TYPE}.pth"
    
    # Test images (modify these paths)
    test_images = [
        "/home/jaey00ns/MetaSlot-main/imgs/slottest.png",
        # Add more test image paths here
    ]
    
    # If test image doesn't exist, try to use COCO validation images
    if not os.path.exists(test_images[0]):
        print(f"Warning: Test image not found at {test_images[0]}")
        print("Please update test_images list with valid image paths")
        
        # Try to find some images from COCO val
        coco_val_dir = Path("/home/jaey00ns/coco_original/val2017")
        if coco_val_dir.exists():
            coco_images = list(coco_val_dir.glob("*.jpg"))[:4]
            if coco_images:
                test_images = [str(p) for p in coco_images]
                print(f"Using COCO validation images: {len(test_images)} images")
    
    print("="*60)
    print("Slot Autoencoder Evaluation")
    print("="*60)
    print(f"Model: {MODEL_TYPE}")
    print(f"Model path: {model_path}")
    print(f"Test images: {len(test_images)}")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the model first using: python trainae.py")
    else:
        # Evaluate
        evaluate_autoencoder(
            model_path=model_path,
            test_image_paths=test_images,
            num_samples=4
        )
