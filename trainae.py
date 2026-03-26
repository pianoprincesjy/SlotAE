"""
Slot Autoencoder Training Script
두 개의 slot을 합치고 다시 분리하는 autoencoder 학습
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tqdm
from itertools import combinations
from datetime import datetime

sys.path.append('/home/jaey00ns/MetaSlot-main')
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config
from object_centric_bench.datum import DataLoader


# ==================== Hyperparameters ====================
# 여기서 모든 하이퍼파라미터를 수정하세요

# Model Type
USE_NONLINEAR = False  # True: nonlinear MLP, False: linear

# Training Settings
NUM_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# Model Architecture
SLOT_DIM = 256
HIDDEN_DIM = 512  # nonlinear 모델의 hidden layer 크기
ATTENTION_SIZE = 16 * 16  # attention map 크기 (H*W) - MetaSlot uses 16x16

# Data Loading
NUM_WORKERS = 16  # 데이터 로딩 병렬 처리 워커 수

# Paths
METASLOT_CONFIG = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py"
METASLOT_CHECKPOINT = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth"
DATA_BASE_DIR = "/home/jaey00ns/MetaSlot-main/data"
SAVE_DIR = "/home/jaey00ns/MetaSlot-main/slotae/pth"

# ==================== Autoencoder Models ====================

class LinearSlotAutoencoder(nn.Module):
    """간단한 선형 변환 autoencoder"""
    def __init__(self, slot_dim, attention_size):
        super().__init__()
        self.slot_encoder = nn.Linear(slot_dim * 2, slot_dim)
        self.slot_decoder = nn.Linear(slot_dim, slot_dim * 2)
        self.attention_decoder = nn.Linear(attention_size, attention_size * 2)
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)  # (B, 512)
        return self.slot_encoder(combined)  # (B, 256)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.slot_decoder(encoded_slot)  # (B, 512)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon
    
    def decode_attention(self, attention):
        """하나의 attention map을 두 개로 분리
        Args:
            attention: (B, H*W) flattened attention map
        Returns:
            attention1, attention2: (B, H*W) each
        """
        decoded = self.attention_decoder(attention)  # (B, 2*H*W)
        mid = decoded.shape[-1] // 2
        attention1 = decoded[..., :mid]
        attention2 = decoded[..., mid:]
        # Normalize
        attention1 = pt.softmax(attention1.view(-1, attention1.shape[-1]), dim=-1)
        attention2 = pt.softmax(attention2.view(-1, attention2.shape[-1]), dim=-1)
        return attention1, attention2


class NonlinearSlotAutoencoder(nn.Module):
    """비선형 MLP autoencoder"""
    def __init__(self, slot_dim, hidden_dim, attention_size):
        super().__init__()
        # Slot Encoder: 2*slot_dim -> hidden -> slot_dim
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
        # Slot Decoder: slot_dim -> hidden -> 2*slot_dim
        self.slot_decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim * 2),
        )
        
        # Attention Decoder: attention_size -> hidden -> 2*attention_size
        # Split 시 하나의 attention을 두 개로 분리
        self.attention_decoder = nn.Sequential(
            nn.Linear(attention_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_size * 2),
        )
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)  # (B, 512)
        return self.slot_encoder(combined)  # (B, 256)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.slot_decoder(encoded_slot)  # (B, 512)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon
    
    def decode_attention(self, attention):
        """하나의 attention map을 두 개로 분리
        Args:
            attention: (B, H*W) flattened attention map
        Returns:
            attention1, attention2: (B, H*W) each
        """
        decoded = self.attention_decoder(attention)  # (B, 2*H*W)
        mid = decoded.shape[-1] // 2
        attention1 = decoded[..., :mid]
        attention2 = decoded[..., mid:]
        # Normalize
        attention1 = pt.softmax(attention1.view(-1, attention1.shape[-1]), dim=-1)
        attention2 = pt.softmax(attention2.view(-1, attention2.shape[-1]), dim=-1)
        return attention1, attention2


# ==================== Training Function ====================

def train_slot_autoencoder():
    """Slot Autoencoder 학습"""
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ==================== Load Pretrained MetaSlot ====================
    print("\n[1/5] Loading pretrained MetaSlot model...")
    
    cfg = Config.fromfile(METASLOT_CONFIG)
    
    # Build model
    metaslot_model = build_from_config(cfg.model)
    metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
    
    # Load checkpoint
    state = pt.load(METASLOT_CHECKPOINT, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    metaslot_model.load_state_dict(state, strict=False)
    
    metaslot_model = metaslot_model.to(device).eval()
    
    # Freeze MetaSlot model (no gradient)
    for param in metaslot_model.parameters():
        param.requires_grad = False
    
    print("✓ MetaSlot model loaded and frozen")
    
    # ==================== Create Autoencoder ====================
    print(f"\n[2/5] Creating {'Nonlinear' if USE_NONLINEAR else 'Linear'} Slot Autoencoder...")
    
    if USE_NONLINEAR:
        autoencoder = NonlinearSlotAutoencoder(
            slot_dim=SLOT_DIM,
            hidden_dim=HIDDEN_DIM,
            attention_size=ATTENTION_SIZE
        )
        model_name = "nonlinear"
    else:
        autoencoder = LinearSlotAutoencoder(
            slot_dim=SLOT_DIM,
            attention_size=ATTENTION_SIZE
        )
        model_name = "linear"
    
    autoencoder = autoencoder.to(device)
    print(f"✓ {model_name.capitalize()} autoencoder created")
    
    # ==================== Setup Dataset ====================
    print("\n[3/5] Setting up dataset...")
    
    # Update dataset config
    cfg.dataset_t.base_dir = Path(DATA_BASE_DIR)
    cfg.batch_size_t = BATCH_SIZE
    cfg.num_work = 8
    
    dataset = build_from_config(cfg.dataset_t)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✓ Dataset ready with {len(dataset)} samples")
    
    # ==================== Setup Optimizer ====================
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    
    # ==================== Setup Save Directory ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(SAVE_DIR) / model_name / timestamp
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Save directory: {save_dir}")
    
    # ==================== Training Loop ====================
    print(f"\n[4/5] Training {model_name} autoencoder for {NUM_EPOCHS} epochs...")
    
    autoencoder.train()
    
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get slots from MetaSlot (frozen, no gradient)
            with pt.no_grad():
                output = metaslot_model(batch)
                slots = output['slotz']  # (B, num_slots, slot_dim)
                
                # Get attention maps
                if 'attent2' in output:
                    attention = output['attent2']
                else:
                    attention = output['attent']
                # attention shape: (B, num_slots, H, W)
            
            B, num_slots, slot_dim = slots.shape
            _, _, H, W = attention.shape
            attention_flat = attention.view(B, num_slots, H*W)  # (B, num_slots, H*W)
            
            # Train on all possible pairs of slots
            batch_loss = 0
            num_pairs = 0
            
            # Generate all pairs (0,1), (0,2), ..., (5,6)
            for i, j in combinations(range(num_slots), 2):
                slot_i = slots[:, i, :]  # (B, slot_dim)
                slot_j = slots[:, j, :]  # (B, slot_dim)
                attention_i = attention_flat[:, i, :]  # (B, H*W)
                attention_j = attention_flat[:, j, :]  # (B, H*W)
                
                # Encode two slots into one
                encoded = autoencoder.encode(slot_i, slot_j)  # (B, slot_dim)
                
                # Merge attentions (sum and normalize)
                merged_attention = attention_i + attention_j  # (B, H*W)
                merged_attention = merged_attention / (merged_attention.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Decode back to two slots
                slot_i_recon, slot_j_recon = autoencoder.decode(encoded)
                
                # Decode attention back to two
                attention_i_recon, attention_j_recon = autoencoder.decode_attention(merged_attention)
                
                # Slot reconstruction loss (MSE)
                loss_slot_i = F.mse_loss(slot_i_recon, slot_i)
                loss_slot_j = F.mse_loss(slot_j_recon, slot_j)
                
                # Attention reconstruction loss (MSE)
                loss_attention_i = F.mse_loss(attention_i_recon, attention_i)
                loss_attention_j = F.mse_loss(attention_j_recon, attention_j)
                
                pair_loss = loss_slot_i + loss_slot_j + loss_attention_i + loss_attention_j
                
                batch_loss += pair_loss
                num_pairs += 1
            
            # Average loss over all pairs
            batch_loss = batch_loss / num_pairs
            
            # Backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += batch_loss.item()
            num_batches += 1
            total_steps += 1
            
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'avg_loss': f'{epoch_loss/num_batches:.4f}'
            })
            
            # Full epoch training (remove limit for complete training)
            # if batch_idx >= 100:  # 100 batches per epoch
            #     break
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        save_path = save_dir / f"epoch_{epoch+1:03d}.pth"
        pt.save({
            'model_state_dict': autoencoder.state_dict(),
            'model_type': model_name,
            'slot_dim': SLOT_DIM,
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
        }, save_path)
        print(f"  Saved checkpoint: {save_path.name}")
    
    # ==================== Save Final Model ====================
    print("\n[5/5] Saving final model...")
    
    final_path = save_dir / "final.pth"
    pt.save({
        'model_state_dict': autoencoder.state_dict(),
        'model_type': model_name,
        'slot_dim': SLOT_DIM,
        'epoch': NUM_EPOCHS,
    }, final_path)
    
    # Also save to root pth folder for easy access
    root_save_path = Path(SAVE_DIR) / f"slotae_{model_name}.pth"
    pt.save({
        'model_state_dict': autoencoder.state_dict(),
        'model_type': model_name,
        'slot_dim': SLOT_DIM,
        'epoch': NUM_EPOCHS,
    }, root_save_path)
    
    print(f"✓ Final model saved to {final_path}")
    print(f"✓ Latest model saved to {root_save_path}")
    print(f"\n{'='*60}")
    print(f"Training completed for {model_name} autoencoder!")
    print(f"All checkpoints saved in: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("Slot Autoencoder Training")
    print("="*60)
    print(f"Model Type: {'Nonlinear MLP' if USE_NONLINEAR else 'Linear'}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Slot Dim: {SLOT_DIM}")
    if USE_NONLINEAR:
        print(f"Hidden Dim: {HIDDEN_DIM}")
    print(f"Attention Size: {ATTENTION_SIZE}")
    print(f"Num Workers: {NUM_WORKERS}")
    print("="*60)
    
    # Train
    train_slot_autoencoder()
