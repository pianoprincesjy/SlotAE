"""
Slot Autoencoder Training Script (v2 - Slot Only)
두 개의 slot을 합치고 다시 분리하는 autoencoder 학습 (attention 학습 제거)
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
    parser = argparse.ArgumentParser(description='Train Slot Autoencoder')
    
    # Model
    parser.add_argument('--model-config', type=str, default='nonlinear_deep',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model configuration')
    parser.add_argument('--slot-dim', type=int, default=256,
                        help='Slot dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of data loading workers')
    
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
                        default='/home/jaey00ns/MetaSlot-main/slotae/pth',
                        help='Save directory')
    
    # Device
    parser.add_argument('--gpu', type=str, default='5',
                        help='GPU device ID')
    
    return parser.parse_args()

# ==================== Training Functions ====================

def train_autoencoder(autoencoder, dataloader, metaslot_model, num_epochs, device, save_dir, model_type, batch_size, learning_rate, slot_dim=256):
    """
    Autoencoder 학습 (slot만)
    
    전략: 모든 가능한 slot 쌍 (21개)에 대해 학습
    - 7개 슬롯에서 2개를 선택하는 조합: C(7,2) = 21
    - 각 배치에서 모든 21개 쌍에 대해 loss 계산
    """
    autoencoder.train()
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    # Loss history 기록
    loss_history = {
        'epoch': [],
        'train_loss': [],
        'timestamp': []
    }
    
    # 모든 slot 쌍 생성
    num_slots = 7
    slot_pairs = list(combinations(range(num_slots), 2))
    print(f"Training on {len(slot_pairs)} slot pairs: {slot_pairs}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
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
            
            for idx1, idx2 in slot_pairs:
                slot1 = slots[:, idx1, :]  # (B, 256)
                slot2 = slots[:, idx2, :]
                
                # Encode
                encoded = autoencoder.encode(slot1, slot2)
                
                # Decode
                slot1_recon, slot2_recon = autoencoder.decode(encoded)
                
                # Loss: MSE between original and reconstructed slots
                loss_slot1 = F.mse_loss(slot1_recon, slot1)
                loss_slot2 = F.mse_loss(slot2_recon, slot2)
                
                pair_loss = loss_slot1 + loss_slot2
                total_loss += pair_loss
            
            # Average loss across all pairs
            total_loss = total_loss / len(slot_pairs)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{total_loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Record loss history
        loss_history['epoch'].append(epoch + 1)
        loss_history['train_loss'].append(avg_loss)
        loss_history['timestamp'].append(datetime.now().isoformat())
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'model_config': model_type,
            'slot_dim': slot_dim,
        }
        checkpoint_path = save_dir / f"{model_type}_batch{batch_size}_epoch_{epoch+1:04d}.pth"
        pt.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    return autoencoder, loss_history


def save_loss_history(loss_history, save_path):
    """Loss history를 JSON으로 저장"""
    with open(save_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"✓ Loss history saved: {save_path}")


def plot_loss_curve(loss_history, save_path):
    """Loss curve 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['epoch'], loss_history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Loss curve saved: {save_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (GPU: {args.gpu})")
    
    print("\n" + "="*60)
    print("Slot Autoencoder Training (v2 - Slot Only)")
    print("="*60)
    print(f"Model Config: {args.model_config}")
    print(f"Model Description: {MODEL_CONFIGS[args.model_config]['description']}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
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
    print(f"✓ Autoencoder initialized: {args.model_config}")
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
        slot_dim=args.slot_dim
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
        'loss_history': loss_history,
    }
    
    save_path = save_dir / f"{args.model_config}_batch{args.batch_size}_final.pth"
    pt.save(checkpoint, save_path)
    print(f"✓ Model saved to: {save_path}")
    
    # Save loss history
    loss_path = save_dir / f"{args.model_config}_batch{args.batch_size}_loss_history.json"
    save_loss_history(loss_history, loss_path)
    
    # Plot loss curve
    plot_path = save_dir / f"{args.model_config}_batch{args.batch_size}_loss_curve.png"
    plot_loss_curve(loss_history, plot_path)
    
    # ==================== Summary ====================
    print("\n[6/6] Training summary...")
    print("\n" + "="*60)
    print("Training Completed Successfully!")
    print("="*60)
    print(f"Model: {args.model_config}")
    print(f"Final Loss: {loss_history['train_loss'][-1]:.6f}")
    print(f"Save Path: {save_path}")
    print(f"Loss History: {loss_path}")
    print(f"Loss Curve: {plot_path}")
    print("="*60)


if __name__ == "__main__":
    main()
