"""
Slot Autoencoder Training Script - Layered Linear Models with Feature Matching Loss
레이어별 선형 모델 학습 (feature-matching loss 포함)
"""
import os
import sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tqdm
from itertools import combinations
from datetime import datetime
import argparse
import json
import matplotlib.pyplot as plt

sys.path.append('/home/jaey00ns/MetaSlot-main')
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config
from object_centric_bench.datum import DataLoader

# Import autoencoder models
from models import create_autoencoder, list_available_models, MODEL_CONFIGS


# ==================== Argument Parsing ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Layered Linear Slot Autoencoder')
    
    # Model
    parser.add_argument('--model-config', type=str, default='linear_layered_4',
                        help='Model configuration (e.g., linear_layered_2, linear_layered_4, etc.)')
    parser.add_argument('--slot-dim', type=int, default=256,
                        help='Slot dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--feature-match-weight', type=float, default=0.5,
                        help='Weight for feature matching loss')
    
    # Paths
    parser.add_argument('--metaslot-config', type=str,
                        default='/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py',
                        help='MetaSlot config path')
    parser.add_argument('--metaslot-checkpoint', type=str,
                        default='/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth',
                        help='MetaSlot checkpoint path')
    parser.add_argument('--data-dir', type=str,
                        default='/home/jaey00ns/MetaSlot-main/data',
                        help='Data directory')
    parser.add_argument('--save-dir', type=str,
                        default='/home/jaey00ns/MetaSlot-main/slotae/pth_layered',
                        help='Save directory')
    
    # Device
    parser.add_argument('--gpu', type=str, default='5',
                        help='GPU device ID')
    
    return parser.parse_args()


# ==================== Loss Functions ====================

def compute_feature_matching_loss(enc_intermediates, dec_intermediates):
    """
    Feature matching loss: 인코더와 디코더의 대응되는 레이어 feature 비교
    
    Args:
        enc_intermediates: 인코더 중간 feature 리스트 [input, layer1, layer2, ..., latent]
        dec_intermediates: 디코더 중간 feature 리스트 [latent, layer1, layer2, ..., output]
    
    Returns:
        feature matching loss (scalar)
    """
    # 인코더: [512, dim1, dim2, ..., 256]
    # 디코더: [256, ..., dim2, dim1, 512]
    # 대응: enc[1] <-> dec[-2], enc[2] <-> dec[-3], ...
    
    num_enc_layers = len(enc_intermediates) - 1  # input 제외
    num_dec_layers = len(dec_intermediates) - 1  # output 제외
    
    assert num_enc_layers == num_dec_layers, "Encoder and decoder should have same number of layers"
    
    total_loss = 0.0
    num_pairs = 0
    
    # 대응되는 레이어끼리 비교 (latent는 제외)
    for i in range(1, num_enc_layers):  # enc[1], enc[2], ... (enc[0]은 input)
        enc_feat = enc_intermediates[i]
        dec_feat = dec_intermediates[-(i+1)]  # dec[-2], dec[-3], ...
        
        # 차원이 같은지 확인
        if enc_feat.shape[-1] == dec_feat.shape[-1]:
            loss = F.mse_loss(enc_feat, dec_feat)
            total_loss += loss
            num_pairs += 1
    
    if num_pairs > 0:
        return total_loss / num_pairs
    else:
        return pt.tensor(0.0, device=enc_intermediates[0].device)


# ==================== Training Functions ====================

def train_autoencoder(autoencoder, dataloader, metaslot_model, num_epochs, device, save_dir, 
                     model_type, batch_size, learning_rate, slot_dim=256, feature_match_weight=0.5):
    """
    Layered Autoencoder 학습 (feature-matching loss 포함)
    
    Loss = reconstruction_loss + feature_match_weight * feature_matching_loss
    """
    autoencoder.train()
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    # Loss history 기록
    loss_history = {
        'epoch': [],
        'train_loss': [],
        'recon_loss': [],
        'feature_loss': [],
        'timestamp': []
    }
    
    # 모든 slot 쌍 생성
    num_slots = 7
    slot_pairs = list(combinations(range(num_slots), 2))
    print(f"Training on {len(slot_pairs)} slot pairs with feature matching")
    print(f"Feature matching weight: {feature_match_weight}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_feature_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            image = batch['image'].to(device)
            
            # MetaSlot으로 slots 추출 (frozen)
            with pt.no_grad():
                output = metaslot_model({'image': image})
                slots = output['slotz']  # (B, num_slots, slot_dim)
            
            # 모든 slot 쌍에 대해 학습
            total_loss = 0.0
            total_recon = 0.0
            total_feature = 0.0
            
            for idx1, idx2 in slot_pairs:
                slot1 = slots[:, idx1, :]  # (B, 256)
                slot2 = slots[:, idx2, :]
                
                # Forward with intermediates
                slot1_recon, slot2_recon, encoded, enc_intermediates, dec_intermediates = \
                    autoencoder(slot1, slot2, return_intermediates=True)
                
                # Reconstruction loss
                loss_slot1 = F.mse_loss(slot1_recon, slot1)
                loss_slot2 = F.mse_loss(slot2_recon, slot2)
                recon_loss = loss_slot1 + loss_slot2
                
                # Feature matching loss
                feature_loss = compute_feature_matching_loss(enc_intermediates, dec_intermediates)
                
                # Total loss
                pair_loss = recon_loss + feature_match_weight * feature_loss
                
                total_loss += pair_loss
                total_recon += recon_loss
                total_feature += feature_loss
            
            # Average loss across all pairs
            total_loss = total_loss / len(slot_pairs)
            total_recon = total_recon / len(slot_pairs)
            total_feature = total_feature / len(slot_pairs)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += total_recon.item()
            epoch_feature_loss += total_feature.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.6f}',
                'recon': f'{total_recon.item():.6f}',
                'feat': f'{total_feature.item():.6f}'
            })
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_feature = epoch_feature_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss: {avg_loss:.6f}")
        print(f"  Recon Loss: {avg_recon:.6f}")
        print(f"  Feature Loss: {avg_feature:.6f}")
        
        # Record loss history
        loss_history['epoch'].append(epoch + 1)
        loss_history['train_loss'].append(avg_loss)
        loss_history['recon_loss'].append(avg_recon)
        loss_history['feature_loss'].append(avg_feature)
        loss_history['timestamp'].append(datetime.now().isoformat())
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'feature_loss': avg_feature,
            'model_config': model_type,
            'slot_dim': slot_dim,
            'feature_match_weight': feature_match_weight,
        }
        checkpoint_path = save_dir / f"{model_type}_batch{batch_size}_epoch_{epoch+1:04d}.pth"
        pt.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
    
    return autoencoder, loss_history


def save_loss_history(loss_history, save_path):
    """Loss history를 JSON으로 저장"""
    with open(save_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"✓ Loss history saved: {save_path}")


def plot_loss_curves(loss_history, save_path):
    """Loss curve 시각화 (total, recon, feature 모두)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = loss_history['epoch']
    
    # Total Loss
    axes[0].plot(epochs, loss_history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[1].plot(epochs, loss_history['recon_loss'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Feature Matching Loss
    axes[2].plot(epochs, loss_history['feature_loss'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Feature Matching Loss', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Loss curves saved: {save_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Validate model config
    if not args.model_config.startswith('linear_layered_'):
        print(f"Warning: This script is designed for linear_layered models")
        print(f"You specified: {args.model_config}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (GPU: {args.gpu})")
    
    print("\n" + "="*60)
    print("Layered Linear Slot Autoencoder Training")
    print("with Feature Matching Loss")
    print("="*60)
    print(f"Model Config: {args.model_config}")
    print(f"Model Description: {MODEL_CONFIGS[args.model_config]['description']}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Feature Match Weight: {args.feature_match_weight}")
    print(f"Slot Dim: {args.slot_dim}")
    print(f"MetaSlot Config: {Path(args.metaslot_config).name}")
    print(f"MetaSlot Checkpoint: {Path(args.metaslot_checkpoint).name}")
    print("="*60 + "\n")
    
    # ==================== Load MetaSlot (Frozen) ====================
    print("[1/6] Loading MetaSlot model...")
    cfg = Config.fromfile(args.metaslot_config)
    metaslot_model = build_from_config(cfg.model)
    metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
    
    state = pt.load(args.metaslot_checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    metaslot_model.load_state_dict(state, strict=False)
    metaslot_model = metaslot_model.to(device).eval()
    
    # Freeze MetaSlot
    for param in metaslot_model.parameters():
        param.requires_grad = False
    
    print("✓ MetaSlot loaded and frozen")
    
    # ==================== Setup Dataloader ====================
    print("\n[2/6] Setting up dataloader...")
    
    dataset_config = cfg.dataset_t
    dataset_config['base_dir'] = Path(args.data_dir)
    dataset = build_from_config(dataset_config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=None,
        drop_last=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"✓ Dataloader ready: {len(dataloader)} batches per epoch")
    
    # ==================== Initialize Autoencoder ====================
    print("\n[3/6] Initializing autoencoder...")
    
    # Create autoencoder using factory function
    autoencoder = create_autoencoder(args.model_config, slot_dim=args.slot_dim)
    autoencoder = autoencoder.to(device)
    
    num_params = sum(p.numel() for p in autoencoder.parameters())
    num_layers = MODEL_CONFIGS[args.model_config].get('num_layers', 'N/A')
    print(f"✓ Autoencoder initialized: {args.model_config}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total parameters: {num_params:,}")
    
    # ==================== Create Save Directory ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / args.model_config / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Save directory created: {save_dir}")
    
    # Save training config
    config_dict = vars(args)
    config_dict['timestamp'] = timestamp
    config_dict['num_parameters'] = num_params
    config_dict['num_layers'] = num_layers
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Config saved: {save_dir / 'config.json'}")
    
    # ==================== Train ====================
    print(f"\n[4/6] Training for {args.epochs} epochs...")
    
    autoencoder, loss_history = train_autoencoder(
        autoencoder=autoencoder,
        dataloader=dataloader,
        metaslot_model=metaslot_model,
        num_epochs=args.epochs,
        device=device,
        save_dir=save_dir,
        model_type=args.model_config,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        slot_dim=args.slot_dim,
        feature_match_weight=args.feature_match_weight
    )
    
    # ==================== Save Results ====================
    print("\n[5/6] Saving results...")
    
    # Save final checkpoint
    checkpoint = {
        'model_state_dict': autoencoder.state_dict(),
        'model_config': args.model_config,
        'model_description': MODEL_CONFIGS[args.model_config]['description'],
        'slot_dim': args.slot_dim,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'feature_match_weight': args.feature_match_weight,
        'loss_history': loss_history,
    }
    
    save_path = save_dir / f"{args.model_config}_batch{args.batch_size}_final.pth"
    pt.save(checkpoint, save_path)
    print(f"✓ Model saved to: {save_path}")
    
    # Save loss history
    loss_path = save_dir / f"{args.model_config}_batch{args.batch_size}_loss_history.json"
    save_loss_history(loss_history, loss_path)
    
    # Plot loss curves
    plot_path = save_dir / f"{args.model_config}_batch{args.batch_size}_loss_curves.png"
    plot_loss_curves(loss_history, plot_path)
    
    # ==================== Summary ====================
    print("\n[6/6] Training summary...")
    print("\n" + "="*60)
    print("Training Completed Successfully!")
    print("="*60)
    print(f"Model: {args.model_config}")
    print(f"Final Total Loss: {loss_history['train_loss'][-1]:.6f}")
    print(f"Final Recon Loss: {loss_history['recon_loss'][-1]:.6f}")
    print(f"Final Feature Loss: {loss_history['feature_loss'][-1]:.6f}")
    print(f"Save Path: {save_path}")
    print(f"Loss History: {loss_path}")
    print(f"Loss Curves: {plot_path}")
    print("="*60)


if __name__ == "__main__":
    main()
