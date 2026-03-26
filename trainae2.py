"""
Slot Autoencoder Training Script (v2 - Slot Only)
두 개의 slot을 합치고 다시 분리하는 autoencoder 학습 (attention 학습 제거)
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
NUM_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# Model Architecture
SLOT_DIM = 256
HIDDEN_DIM = 512  # nonlinear 모델의 hidden layer 크기

# Data Loading
NUM_WORKERS = 16  # 데이터 로딩 병렬 처리 워커 수

# Paths
METASLOT_CONFIG = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py"
METASLOT_CHECKPOINT = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth"
DATA_BASE_DIR = "/home/jaey00ns/MetaSlot-main/data"
SAVE_DIR = "/home/jaey00ns/MetaSlot-main/slotae/pth"

# ==================== Autoencoder Models (Slot Only) ====================

class LinearSlotAutoencoder(nn.Module):
    """간단한 선형 변환 autoencoder (slot만)"""
    def __init__(self, slot_dim):
        super().__init__()
        self.encoder = nn.Linear(slot_dim * 2, slot_dim)
        self.decoder = nn.Linear(slot_dim, slot_dim * 2)
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)  # (B, 512)
        return self.encoder(combined)  # (B, 256)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.decoder(encoded_slot)  # (B, 512)
        slot1_recon = decoded[..., :256]
        slot2_recon = decoded[..., 256:]
        return slot1_recon, slot2_recon


class NonlinearSlotAutoencoder(nn.Module):
    """비선형 MLP autoencoder (slot만)"""
    def __init__(self, slot_dim, hidden_dim):
        super().__init__()
        # Slot Encoder: 2*slot_dim -> hidden -> slot_dim
        self.encoder = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
        # Slot Decoder: slot_dim -> hidden -> 2*slot_dim
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


# ==================== Training Functions ====================

def train_autoencoder(autoencoder, dataloader, metaslot_model, num_epochs, device):
    """
    Autoencoder 학습 (slot만)
    
    전략: 모든 가능한 slot 쌍 (21개)에 대해 학습
    - 7개 슬롯에서 2개를 선택하는 조합: C(7,2) = 21
    - 각 배치에서 모든 21개 쌍에 대해 loss 계산
    """
    autoencoder.train()
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    
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
    
    return autoencoder


def main():
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Slot Autoencoder Training (v2 - Slot Only)")
    print("="*60)
    print(f"Model Type: {'Nonlinear MLP' if USE_NONLINEAR else 'Linear'}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Slot Dim: {SLOT_DIM}")
    if USE_NONLINEAR:
        print(f"Hidden Dim: {HIDDEN_DIM}")
    print("="*60 + "\n")
    
    # ==================== Load MetaSlot (Frozen) ====================
    print("[1/5] Loading MetaSlot model...")
    cfg = Config.fromfile(METASLOT_CONFIG)
    metaslot_model = build_from_config(cfg.model)
    metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
    
    state = pt.load(METASLOT_CHECKPOINT, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    metaslot_model.load_state_dict(state, strict=False)
    metaslot_model = metaslot_model.to(device).eval()
    
    # Freeze MetaSlot
    for param in metaslot_model.parameters():
        param.requires_grad = False
    
    print("✓ MetaSlot loaded and frozen")
    
    # ==================== Setup Dataloader ====================
    print("\n[2/5] Setting up dataloader...")
    
    dataset_config = cfg.dataset_t
    dataset_config['base_dir'] = Path(DATA_BASE_DIR)
    dataset = build_from_config(dataset_config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=None,
        drop_last=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"✓ Dataloader ready: {len(dataloader)} batches per epoch")
    
    # ==================== Initialize Autoencoder ====================
    print("\n[3/5] Initializing autoencoder...")
    
    if USE_NONLINEAR:
        autoencoder = NonlinearSlotAutoencoder(
            slot_dim=SLOT_DIM,
            hidden_dim=HIDDEN_DIM
        )
        model_type = 'nonlinear'
    else:
        autoencoder = LinearSlotAutoencoder(slot_dim=SLOT_DIM)
        model_type = 'linear'
    
    autoencoder = autoencoder.to(device)
    
    num_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"✓ {model_type.capitalize()} autoencoder initialized")
    print(f"  Total parameters: {num_params:,}")
    
    # ==================== Train ====================
    print(f"\n[4/5] Training for {NUM_EPOCHS} epochs...")
    
    autoencoder = train_autoencoder(
        autoencoder=autoencoder,
        dataloader=dataloader,
        metaslot_model=metaslot_model,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    # ==================== Save Model ====================
    print("\n[5/5] Saving model...")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(SAVE_DIR) / model_type / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': autoencoder.state_dict(),
        'model_type': model_type,
        'slot_dim': SLOT_DIM,
        'hidden_dim': HIDDEN_DIM if USE_NONLINEAR else None,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    }
    
    save_path = save_dir / "final.pth"
    pt.save(checkpoint, save_path)
    
    print(f"✓ Model saved to: {save_path}")
    
    print("\n" + "="*60)
    print("Training Completed Successfully!")
    print("="*60)
    print(f"Model: {model_type}")
    print(f"Save Path: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
