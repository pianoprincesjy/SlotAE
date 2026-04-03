"""
Layered Linear Models Analysis
학습된 레이어별 선형 모델들의 loss history를 수집하고 CSV 및 비교 차트 생성 + 시각화
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import torch as pt
import torch.nn.functional as F
from PIL import Image
import cv2
from einops import rearrange

sys.path.append('/home/jaey00ns/MetaSlot-main')
sys.path.append('/home/jaey00ns/MetaSlot-main/slotae')

# Configuration
BASE_DIR = Path("/home/jaey00ns/MetaSlot-main/slotae")
OUTPUT_DIR = BASE_DIR / "eval_layered"
LOSS_DIR = OUTPUT_DIR / "loss_curves"
VIS_DIR = OUTPUT_DIR / "visualizations"

MODELS = [f'linear_layered_{i}' for i in range(2, 11)]  # 2 to 10 layers
BATCH_SIZE = 256  # Fixed batch size

PRETRAINS = {
    'coco': BASE_DIR / 'pth_layered_coco',
    'clevrtex': BASE_DIR / 'pth_layered_clevrtex'
}

# MetaSlot configs
METASLOT_CONFIGS = {
    'coco': {
        'config': '/home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-coco.py',
        'checkpoint': '/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth'
    },
    'clevrtex': {
        'config': '/home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py',
        'checkpoint': '/home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth'
    }
}

TEST_IMAGE = "/home/jaey00ns/MetaSlot-main/imgs/slottest.png"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
LOSS_DIR.mkdir(exist_ok=True)
(VIS_DIR / 'coco').mkdir(parents=True, exist_ok=True)
(VIS_DIR / 'clevrtex').mkdir(parents=True, exist_ok=True)

# Color palette - gradient from light to dark
COLORS = {
    'linear_layered_2': '#fee5d9',
    'linear_layered_3': '#fcbba1',
    'linear_layered_4': '#fc9272',
    'linear_layered_5': '#fb6a4a',
    'linear_layered_6': '#ef3b2c',
    'linear_layered_7': '#cb181d',
    'linear_layered_8': '#a50f15',
    'linear_layered_9': '#67000d',
    'linear_layered_10': '#000000',
}

# ImageNet normalization
IMAGENET_MEAN = np.array([[[123.675]], [[116.28]], [[103.53]]], dtype=np.float32)
IMAGENET_STD = np.array([[[58.395]], [[57.12]], [[57.375]]], dtype=np.float32)


def find_latest_run(model_dir):
    """Find the most recent training run directory"""
    if not model_dir.exists():
        return None
    
    # Get all timestamped directories
    runs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('202')]
    if not runs:
        return None
    
    # Return the most recent one
    return max(runs, key=lambda p: p.name)


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for MetaSlot model"""
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
    """Generate attention maps from slots"""
    clue = [16, 16]
    recon, attent2 = decoder(clue, new_slots)
    B, N, HW = attent2.shape
    h = w = 16
    attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)
    attent2 = F.interpolate(attent2, size=(256, 256), mode='bilinear', align_corners=False)
    return attent2, recon


def generate_slot_colors(num_slots):
    """Generate distinct colors for slots"""
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
    """Visualize slots with colored masks"""
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
    """Add legend to image showing merge/split operations"""
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
                                  original_attention_output, encoder_pair=(0, 1), 
                                  decoder_idx=3, save_path=None, title_suffix=""):
    """Visualize merge and split operations"""
    device = slots.device
    num_slots = slots.shape[0]
    
    # Validate indices
    idx1, idx2 = encoder_pair
    if idx1 >= num_slots or idx2 >= num_slots:
        # Adjust to valid indices
        idx1 = min(idx1, num_slots - 2)
        idx2 = min(idx2, num_slots - 1)
        encoder_pair = (idx1, idx2)
    
    if decoder_idx >= num_slots:
        decoder_idx = min(decoder_idx, num_slots - 1)
    
    max_possible_slots = num_slots + 2
    all_colors = generate_slot_colors(max_possible_slots)
    
    remaining_indices = [i for i in range(num_slots) if i not in encoder_pair]
    
    # Original slots
    original_slot_indices = list(range(num_slots))
    img_01 = visualize_slots_with_mask(original_image, original_attention_output, 
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_00 = add_legend_to_image(original_image, all_colors, 
                                  merge_indices=encoder_pair, 
                                  output_indices=[num_slots])
    
    # Merge operation
    slot1 = slots[idx1:idx1+1]  # (1, 256)
    slot2 = slots[idx2:idx2+1]  # (1, 256)
    
    with pt.no_grad():
        encoded_slot = autoencoder.encode(slot1, slot2)  # Should return (1, 256)
        
        # Ensure encoded_slot has correct shape
        if encoded_slot.dim() == 1:
            encoded_slot = encoded_slot.unsqueeze(0)
        
        # Get remaining slots
        if len(remaining_indices) > 0:
            remaining_slots = slots[remaining_indices]  # (n-2, 256)
            merged_slots = pt.cat([remaining_slots, encoded_slot], dim=0).unsqueeze(0)  # (1, n-1, 256)
        else:
            merged_slots = encoded_slot.unsqueeze(0)  # (1, 1, 256)
        
        attent2_merged, recon_merged = generate_attent2_from_slots(merged_slots, decoder)
        attent2_merged = attent2_merged.squeeze(0).cpu().numpy()
    
    merged_slot_indices = remaining_indices + [num_slots]
    img_02 = visualize_slots_with_mask(original_image, attent2_merged,
                                        merged_slot_indices, all_colors, alpha=0.5)
    
    # Split operation
    img_11 = visualize_slots_with_mask(original_image, original_attention_output,
                                        original_slot_indices, all_colors, alpha=0.5)
    
    img_10 = add_legend_to_image(original_image, all_colors,
                                  split_index=decoder_idx,
                                  output_indices=[num_slots, num_slots+1])
    
    slot_to_split = slots[decoder_idx:decoder_idx+1]  # (1, 256)
    
    with pt.no_grad():
        slot_recon1, slot_recon2 = autoencoder.decode(slot_to_split)  # Each (1, 256)
        
        # Ensure correct shapes
        if slot_recon1.dim() == 1:
            slot_recon1 = slot_recon1.unsqueeze(0)
        if slot_recon2.dim() == 1:
            slot_recon2 = slot_recon2.unsqueeze(0)
        
        remaining_indices_split = [i for i in range(num_slots) if i != decoder_idx]
        
        if len(remaining_indices_split) > 0:
            remaining_slots_split = slots[remaining_indices_split]  # (n-1, 256)
            split_slots = pt.cat([remaining_slots_split, slot_recon1, slot_recon2], dim=0).unsqueeze(0)  # (1, n+1, 256)
        else:
            split_slots = pt.cat([slot_recon1, slot_recon2], dim=0).unsqueeze(0)  # (1, 2, 256)
        
        attent2_split, recon_split = generate_attent2_from_slots(split_slots, decoder)
        attent2_split = attent2_split.squeeze(0).cpu().numpy()
    
    split_slot_indices = remaining_indices_split + [num_slots, num_slots+1]
    img_12 = visualize_slots_with_mask(original_image, attent2_split,
                                        split_slot_indices, all_colors, alpha=0.5)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if title_suffix:
        fig.suptitle(f'Layered Linear Model: {title_suffix}', fontsize=16, fontweight='bold')
    
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


def visualize_layered_models():
    """Visualize all layered models"""
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.utils import Config, build_from_config
    from models import create_autoencoder
    
    # Use available GPU or CPU
    if pt.cuda.is_available():
        device = 'cuda:0'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        device = 'cpu'
    
    print("\n" + "="*80)
    print("Visualizing Layered Models")
    print("="*80)
    
    for pretrain_name in ['coco', 'clevrtex']:
        print(f"\n{'='*60}")
        print(f"Processing {pretrain_name.upper()} models...")
        print(f"{'='*60}")
        
        # Load MetaSlot model
        cfg = Config.fromfile(METASLOT_CONFIGS[pretrain_name]['config'])
        metaslot_model = build_from_config(cfg.model)
        metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
        
        state = pt.load(METASLOT_CONFIGS[pretrain_name]['checkpoint'], 
                       map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        metaslot_model.load_state_dict(state, strict=False)
        metaslot_model = metaslot_model.to(device).eval()
        
        decoder = metaslot_model.m.decode
        
        # Preprocess test image
        image_tensor, original_image = preprocess_image(TEST_IMAGE)
        image_tensor = image_tensor.to(device)
        
        # Get MetaSlot slots
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
                original_attention = F.interpolate(original_attention, size=(256, 256), 
                                                   mode='bilinear', align_corners=False)
            
            original_attention_256 = original_attention.squeeze(0).cpu().numpy()
        
        original_resized = cv2.resize(original_image, (256, 256))
        
        # Process each layered model
        pretrain_dir = PRETRAINS[pretrain_name]
        
        for model in MODELS:
            model_dir = pretrain_dir / model
            run_dir = find_latest_run(model_dir)
            
            if run_dir is None:
                print(f"  ⚠️  Missing: {model}")
                continue
            
            num_layers = int(model.split('_')[-1])
            
            # Find all available checkpoints in the run directory
            epoch_checkpoints = sorted(run_dir.glob(f"{model}_batch{BATCH_SIZE}_epoch_*.pth"))
            final_checkpoint = run_dir / f"{model}_batch{BATCH_SIZE}_final.pth"
            
            # Create list of all checkpoints to process
            checkpoints_to_process = []
            
            # Add epoch checkpoints
            for ckpt_path in epoch_checkpoints:
                # Extract epoch number from filename
                filename = ckpt_path.stem
                epoch_str = filename.split('epoch_')[-1]
                try:
                    epoch_num = int(epoch_str)
                    checkpoints_to_process.append((ckpt_path, f"epoch{epoch_num:02d}"))
                except ValueError:
                    continue
            
            # Add final checkpoint
            if final_checkpoint.exists():
                checkpoints_to_process.append((final_checkpoint, 'final'))
            
            if not checkpoints_to_process:
                print(f"  ⚠️  No checkpoints: {model}")
                continue
            
            for checkpoint_path, epoch_label in checkpoints_to_process:
                try:
                    # Load autoencoder (load to CPU first, then move to device)
                    checkpoint = pt.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    if 'model_config' not in checkpoint:
                        model_config = model
                    else:
                        model_config = checkpoint['model_config']
                    
                    slot_dim = checkpoint.get('slot_dim', 256)
                    
                    autoencoder = create_autoencoder(model_config, slot_dim=slot_dim)
                    autoencoder.load_state_dict(checkpoint['model_state_dict'])
                    autoencoder = autoencoder.to(device).eval()
                    
                    # Generate visualization
                    output_dir = VIS_DIR / pretrain_name / model
                    output_dir.mkdir(parents=True, exist_ok=True)
                    save_path = output_dir / f"{epoch_label}.png"
                    
                    visualize_autoencoder_results(
                        original_resized,
                        slots,
                        decoder,
                        autoencoder,
                        original_attention_256,
                        encoder_pair=(0, 7),  # Valid indices for 7 slots (0-6)
                        decoder_idx=3,
                        save_path=save_path,
                        title_suffix=f"{num_layers} layers - {epoch_label}"
                    )
                    
                    print(f"  ✓ {model:20s} / {epoch_label}")
                    
                    # Clean up
                    del autoencoder
                    del checkpoint
                    pt.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  ✗ {model:20s} / {epoch_label} - Error: {str(e)}")
                    continue
        
        # Clean up
        del metaslot_model
        pt.cuda.empty_cache()
    
    print("\n✓ Visualization complete!")


def find_latest_run(model_dir):
    """Find the most recent training run directory"""
    if not model_dir.exists():
        return None
    
    # Get all timestamped directories
    runs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('202')]
    if not runs:
        return None
    
    # Return the most recent one
    return max(runs, key=lambda p: p.name)


def collect_loss_data():
    """Collect all loss history data into a structured format"""
    data = []
    
    for pretrain_name, pretrain_dir in PRETRAINS.items():
        print(f"\n{'='*60}")
        print(f"Collecting {pretrain_name.upper()} experiments...")
        print(f"{'='*60}")
        
        for model in MODELS:
            model_dir = pretrain_dir / model
            run_dir = find_latest_run(model_dir)
            
            if run_dir is None:
                print(f"  ⚠️  Missing: {model}")
                continue
            
            # Find loss history file
            loss_file = list(run_dir.glob(f"{model}_batch{BATCH_SIZE}_loss_history.json"))
            if not loss_file:
                print(f"  ⚠️  No loss history: {model}")
                continue
            
            # Load loss history
            with open(loss_file[0], 'r') as f:
                loss_history = json.load(f)
            
            # Add to dataframe
            for epoch, total_loss, recon_loss, feature_loss in zip(
                loss_history['epoch'], 
                loss_history['train_loss'],
                loss_history['recon_loss'],
                loss_history['feature_loss']
            ):
                data.append({
                    'pretrain': pretrain_name,
                    'model': model,
                    'num_layers': int(model.split('_')[-1]),
                    'batch_size': BATCH_SIZE,
                    'epoch': epoch,
                    'total_loss': total_loss,
                    'recon_loss': recon_loss,
                    'feature_loss': feature_loss,
                    'run_dir': str(run_dir)
                })
            
            final_total = loss_history['train_loss'][-1]
            final_recon = loss_history['recon_loss'][-1]
            final_feature = loss_history['feature_loss'][-1]
            print(f"  ✓ {model:20s} | Total: {final_total:.6f} | Recon: {final_recon:.6f} | Feature: {final_feature:.6f}")
    
    df = pd.DataFrame(data)
    return df


def save_full_csv(df):
    """Save complete loss history to CSV"""
    csv_path = OUTPUT_DIR / "loss_history_layered_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Full loss history saved: {csv_path}")
    print(f"  Total records: {len(df)}")


def create_comparison_csv(df):
    """Create summary comparison CSV (final losses only)"""
    # Get final epoch for each experiment
    final_df = df.loc[df.groupby(['pretrain', 'model'])['epoch'].idxmax()]
    
    # Select relevant columns
    comparison = final_df[['pretrain', 'model', 'num_layers', 'total_loss', 'recon_loss', 'feature_loss']].copy()
    comparison = comparison.sort_values(['pretrain', 'num_layers'])
    
    csv_path = OUTPUT_DIR / "loss_comparison_layered.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Final Loss Comparison (by number of layers)")
    print("="*80)
    for pretrain in ['coco', 'clevrtex']:
        print(f"\n{pretrain.upper()}:")
        pretrain_data = comparison[comparison['pretrain'] == pretrain]
        print(f"{'Layers':>8} | {'Total Loss':>12} | {'Recon Loss':>12} | {'Feature Loss':>12}")
        print("-" * 60)
        for _, row in pretrain_data.iterrows():
            print(f"{row['num_layers']:>8} | {row['total_loss']:>12.6f} | {row['recon_loss']:>12.6f} | {row['feature_loss']:>12.6f}")


def plot_loss_curves_by_pretrain(df):
    """Plot loss curves for each pretrain dataset (all models together)"""
    loss_types = ['total_loss', 'recon_loss', 'feature_loss']
    loss_names = ['Total Loss', 'Reconstruction Loss', 'Feature Matching Loss']
    
    for pretrain in ['coco', 'clevrtex']:
        pretrain_df = df[df['pretrain'] == pretrain]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{pretrain.upper()} - Layered Linear Models', fontsize=16, fontweight='bold')
        
        for ax, loss_type, loss_name in zip(axes, loss_types, loss_names):
            for model in MODELS:
                model_df = pretrain_df[pretrain_df['model'] == model]
                if len(model_df) > 0:
                    num_layers = int(model.split('_')[-1])
                    ax.plot(model_df['epoch'], model_df[loss_type], 
                           label=f'{num_layers} layers', 
                           color=COLORS[model], linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(loss_name, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        save_path = LOSS_DIR / f"{pretrain}_all_models.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Loss curves saved: {save_path}")


def plot_final_loss_comparison(df):
    """Plot bar chart comparing final losses"""
    # Get final losses
    final_df = df.loc[df.groupby(['pretrain', 'model'])['epoch'].idxmax()]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Final Loss Comparison (All Models)', fontsize=16, fontweight='bold')
    
    loss_types = ['total_loss', 'recon_loss', 'feature_loss']
    loss_names = ['Total Loss', 'Reconstruction Loss', 'Feature Matching Loss']
    
    for ax, loss_type, loss_name in zip(axes, loss_types, loss_names):
        # Prepare data
        coco_data = final_df[final_df['pretrain'] == 'coco'].sort_values('num_layers')
        clevrtex_data = final_df[final_df['pretrain'] == 'clevrtex'].sort_values('num_layers')
        
        x = np.arange(len(MODELS))
        width = 0.35
        
        ax.bar(x - width/2, coco_data[loss_type], width, label='COCO', alpha=0.8, color='#3498db')
        ax.bar(x + width/2, clevrtex_data[loss_type], width, label='ClevrTex', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Number of Layers', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(loss_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(range(2, 11))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "final_loss_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Final loss comparison saved: {save_path}")


def plot_loss_vs_layers(df):
    """Plot how loss changes with number of layers"""
    final_df = df.loc[df.groupby(['pretrain', 'model'])['epoch'].idxmax()]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Loss vs Number of Layers', fontsize=16, fontweight='bold')
    
    loss_types = ['total_loss', 'recon_loss', 'feature_loss']
    loss_names = ['Total Loss', 'Reconstruction Loss', 'Feature Matching Loss']
    
    for ax, loss_type, loss_name in zip(axes, loss_types, loss_names):
        for pretrain, color, marker in [('coco', '#3498db', 'o'), ('clevrtex', '#e74c3c', 's')]:
            pretrain_df = final_df[final_df['pretrain'] == pretrain].sort_values('num_layers')
            ax.plot(pretrain_df['num_layers'], pretrain_df[loss_type], 
                   marker=marker, markersize=8, linewidth=2, 
                   label=pretrain.upper(), color=color)
        
        ax.set_xlabel('Number of Layers', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(loss_name, fontsize=14, fontweight='bold')
        ax.set_xticks(range(2, 11))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "loss_vs_layers.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss vs layers plot saved: {save_path}")


def main():
    print("="*80)
    print("Layered Linear Models Analysis")
    print("="*80)
    
    # Visualize models first
    print("\n[1/6] Visualizing layered models...")
    try:
        visualize_layered_models()
    except Exception as e:
        print(f"\n⚠️  Visualization failed: {str(e)}")
        print("Continuing with loss analysis...")
    
    # Collect data
    print("\n[2/6] Collecting loss data...")
    df = collect_loss_data()
    
    if df.empty:
        print("\n⚠️  No data collected. Please check if training has been completed.")
        return
    
    # Save full CSV
    print("\n[3/6] Saving full loss history...")
    save_full_csv(df)
    
    # Create comparison CSV
    print("\n[4/6] Creating comparison summary...")
    create_comparison_csv(df)
    
    # Plot loss curves
    print("\n[5/6] Plotting loss curves...")
    plot_loss_curves_by_pretrain(df)
    
    # Plot comparisons
    print("\n[6/6] Creating comparison plots...")
    plot_final_loss_comparison(df)
    plot_loss_vs_layers(df)
    
    # Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Visualizations: {VIS_DIR}")
    print(f"Loss curves: {LOSS_DIR}")
    print("\nGenerated files:")
    print(f"  Visualizations:")
    print(f"    - visualizations/coco/<model>/epoch01.png, epoch02.png, ..., final.png")
    print(f"    - visualizations/clevrtex/<model>/ (same structure)")
    print(f"    - (All available epochs for each model)")
    print(f"  CSV files:")
    print(f"    - loss_history_layered_full.csv")
    print(f"    - loss_comparison_layered.csv")
    print(f"  Loss curves:")
    print(f"    - coco_all_models.png")
    print(f"    - clevrtex_all_models.png")
    print(f"    - final_loss_comparison.png")
    print(f"    - loss_vs_layers.png")
    print("="*80)


if __name__ == "__main__":
    main()
