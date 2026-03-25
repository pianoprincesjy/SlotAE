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

sys.path.append('/home/jaey00ns/MetaSlot-main')
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config
from object_centric_bench.datum import DataLoader


# ==================== Autoencoder Models ====================

class LinearSlotAutoencoder(nn.Module):
    """간단한 선형 변환 autoencoder"""
    def __init__(self, slot_dim=256):
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
    """비선형 MLP autoencoder"""
    def __init__(self, slot_dim=256, hidden_dim=512):
        super().__init__()
        # Encoder: 2*slot_dim -> hidden -> slot_dim
        self.encoder = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
        # Decoder: slot_dim -> hidden -> 2*slot_dim
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim * 2),
        )
        
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


# ==================== Training Function ====================

def train_slot_autoencoder(use_nonlinear=False, num_epochs=50, batch_size=64, lr=1e-3):
    """
    Slot Autoencoder 학습
    
    Args:
        use_nonlinear: True면 nonlinear MLP, False면 linear autoencoder
        num_epochs: 학습 epoch 수
        batch_size: 배치 크기
        lr: learning rate
    """
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ==================== Load Pretrained MetaSlot ====================
    print("\n[1/5] Loading pretrained MetaSlot model...")
    
    cfg_file = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py"
    ckpt_file = "/home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth"
    
    cfg = Config.fromfile(cfg_file)
    
    # Build model
    metaslot_model = build_from_config(cfg.model)
    metaslot_model = ModelWrap(metaslot_model, cfg.model_imap, cfg.model_omap)
    
    # Load checkpoint
    state = pt.load(ckpt_file, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    metaslot_model.load_state_dict(state, strict=False)
    
    metaslot_model = metaslot_model.to(device).eval()
    
    # Freeze MetaSlot model (no gradient)
    for param in metaslot_model.parameters():
        param.requires_grad = False
    
    print("✓ MetaSlot model loaded and frozen")
    
    # ==================== Create Autoencoder ====================
    print(f"\n[2/5] Creating {'Nonlinear' if use_nonlinear else 'Linear'} Slot Autoencoder...")
    
    slot_dim = 256  # from config
    if use_nonlinear:
        autoencoder = NonlinearSlotAutoencoder(slot_dim=slot_dim, hidden_dim=512)
        model_name = "nonlinear"
    else:
        autoencoder = LinearSlotAutoencoder(slot_dim=slot_dim)
        model_name = "linear"
    
    autoencoder = autoencoder.to(device)
    print(f"✓ {model_name.capitalize()} autoencoder created")
    
    # ==================== Setup Dataset ====================
    print("\n[3/5] Setting up dataset...")
    
    # Update dataset config with correct path
    cfg.dataset_t.base_dir = "/home/jaey00ns"
    cfg.batch_size_t = batch_size
    cfg.num_work = 4
    
    dataset = build_from_config(cfg.dataset_t)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✓ Dataset ready with {len(dataset)} samples")
    
    # ==================== Setup Optimizer ====================
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=lr)
    
    # ==================== Training Loop ====================
    print(f"\n[4/5] Training {model_name} autoencoder for {num_epochs} epochs...")
    
    autoencoder.train()
    
    total_steps = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get slots from MetaSlot (frozen, no gradient)
            with pt.no_grad():
                output = metaslot_model(batch)
                slots = output['slotz']  # (B, num_slots, slot_dim)
            
            B, num_slots, slot_dim = slots.shape
            
            # Train on all possible pairs of slots
            batch_loss = 0
            num_pairs = 0
            
            # Generate all pairs (0,1), (0,2), ..., (5,6)
            for i, j in combinations(range(num_slots), 2):
                slot_i = slots[:, i, :]  # (B, slot_dim)
                slot_j = slots[:, j, :]  # (B, slot_dim)
                
                # Encode two slots into one
                encoded = autoencoder.encode(slot_i, slot_j)  # (B, slot_dim)
                
                # Decode back to two slots
                slot_i_recon, slot_j_recon = autoencoder.decode(encoded)
                
                # Reconstruction loss (MSE)
                loss_i = F.mse_loss(slot_i_recon, slot_i)
                loss_j = F.mse_loss(slot_j_recon, slot_j)
                pair_loss = loss_i + loss_j
                
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
            
            # Limit batches per epoch for faster iteration
            if batch_idx >= 100:  # 100 batches per epoch
                break
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
    
    # ==================== Save Model ====================
    print("\n[5/5] Saving trained autoencoder...")
    
    save_dir = Path("/home/jaey00ns/MetaSlot-main/slotae")
    save_dir.mkdir(exist_ok=True)
    
    save_path = save_dir / f"slotae_{model_name}.pth"
    pt.save({
        'model_state_dict': autoencoder.state_dict(),
        'model_type': model_name,
        'slot_dim': slot_dim,
        'epoch': num_epochs,
    }, save_path)
    
    print(f"✓ Model saved to {save_path}")
    print(f"\n{'='*60}")
    print(f"Training completed for {model_name} autoencoder!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ==================== Configuration ====================
    # 여기서 linear 또는 nonlinear를 선택
    USE_NONLINEAR = False  # True: nonlinear MLP, False: linear
    
    NUM_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    
    print("="*60)
    print("Slot Autoencoder Training")
    print("="*60)
    print(f"Model Type: {'Nonlinear MLP' if USE_NONLINEAR else 'Linear'}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*60)
    
    # Train
    train_slot_autoencoder(
        use_nonlinear=USE_NONLINEAR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
