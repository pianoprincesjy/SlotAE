"""
Slot Autoencoder Models
다양한 복잡도의 slot autoencoder 모델들
"""
import torch as pt
import torch.nn as nn
import torch.nn.functional as F


# ==================== Configuration ====================
# 여기서 모델 구조를 쉽게 수정할 수 있습니다

MODEL_CONFIGS = {
    'linear': {
        'type': 'linear',
        'description': '단순 선형 변환',
    },
    'nonlinear_simple': {
        'type': 'nonlinear',
        'description': '2-layer MLP',
        'encoder_layers': [512, 512],  # hidden dimensions
        'decoder_layers': [512, 512],
        'activation': 'relu',
        'dropout': 0.0,
        'use_batchnorm': False,
    },
    'nonlinear_medium': {
        'type': 'nonlinear',
        'description': '3-layer MLP with dropout',
        'encoder_layers': [768, 512, 512],
        'decoder_layers': [512, 512, 768],
        'activation': 'relu',
        'dropout': 0.1,
        'use_batchnorm': False,
    },
    'nonlinear_deep': {
        'type': 'nonlinear',
        'description': '4-layer MLP with BatchNorm',
        'encoder_layers': [1024, 768, 512, 512],
        'decoder_layers': [512, 512, 768, 1024],
        'activation': 'relu',
        'dropout': 0.1,
        'use_batchnorm': True,
    },
    'nonlinear_gelu': {
        'type': 'nonlinear',
        'description': '3-layer MLP with GELU activation',
        'encoder_layers': [768, 512, 512],
        'decoder_layers': [512, 512, 768],
        'activation': 'gelu',
        'dropout': 0.1,
        'use_batchnorm': False,
    },
}


# ==================== Helper Functions ====================

def get_activation(activation_type):
    """활성화 함수 반환"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'silu': nn.SiLU(),
    }
    return activations.get(activation_type, nn.ReLU())


def build_mlp_layers(input_dim, hidden_dims, output_dim, activation='relu', 
                     dropout=0.0, use_batchnorm=False):
    """MLP 레이어 빌더"""
    layers = []
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


# ==================== Autoencoder Models ====================

class LinearSlotAutoencoder(nn.Module):
    """간단한 선형 변환 autoencoder (slot만)"""
    def __init__(self, slot_dim=256):
        super().__init__()
        self.slot_dim = slot_dim
        self.encoder = nn.Linear(slot_dim * 2, slot_dim)
        self.decoder = nn.Linear(slot_dim, slot_dim * 2)
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)  # (B, 512)
        return self.encoder(combined)  # (B, 256)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.decoder(encoded_slot)  # (B, 512)
        slot1_recon = decoded[..., :self.slot_dim]
        slot2_recon = decoded[..., self.slot_dim:]
        return slot1_recon, slot2_recon
    
    def forward(self, slot1, slot2):
        """Forward pass"""
        encoded = self.encode(slot1, slot2)
        slot1_recon, slot2_recon = self.decode(encoded)
        return slot1_recon, slot2_recon, encoded


class NonlinearSlotAutoencoder(nn.Module):
    """비선형 MLP autoencoder (slot만)"""
    def __init__(self, slot_dim=256, encoder_layers=[512, 512], decoder_layers=[512, 512],
                 activation='relu', dropout=0.0, use_batchnorm=False):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Slot Encoder: 2*slot_dim -> hidden -> slot_dim
        self.encoder = build_mlp_layers(
            input_dim=slot_dim * 2,
            hidden_dims=encoder_layers,
            output_dim=slot_dim,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm
        )
        
        # Slot Decoder: slot_dim -> hidden -> 2*slot_dim
        self.decoder = build_mlp_layers(
            input_dim=slot_dim,
            hidden_dims=decoder_layers,
            output_dim=slot_dim * 2,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm
        )
        
    def encode(self, slot1, slot2):
        """두 slot을 하나로 합침"""
        combined = pt.cat([slot1, slot2], dim=-1)
        return self.encoder(combined)
    
    def decode(self, encoded_slot):
        """하나의 slot을 두 개로 분리"""
        decoded = self.decoder(encoded_slot)
        slot1_recon = decoded[..., :self.slot_dim]
        slot2_recon = decoded[..., self.slot_dim:]
        return slot1_recon, slot2_recon
    
    def forward(self, slot1, slot2):
        """Forward pass"""
        encoded = self.encode(slot1, slot2)
        slot1_recon, slot2_recon = self.decode(encoded)
        return slot1_recon, slot2_recon, encoded


# ==================== Model Factory ====================

def create_autoencoder(config_name='nonlinear_simple', slot_dim=256):
    """
    Autoencoder 모델 생성 팩토리 함수
    
    Args:
        config_name: MODEL_CONFIGS의 키 (예: 'linear', 'nonlinear_simple', 'nonlinear_deep')
        slot_dim: slot의 차원 (기본: 256)
    
    Returns:
        autoencoder 모델
    
    Examples:
        >>> # 선형 모델
        >>> model = create_autoencoder('linear', slot_dim=256)
        
        >>> # 간단한 비선형 모델
        >>> model = create_autoencoder('nonlinear_simple', slot_dim=256)
        
        >>> # 복잡한 비선형 모델
        >>> model = create_autoencoder('nonlinear_deep', slot_dim=256)
    """
    if config_name not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    
    config = MODEL_CONFIGS[config_name]
    
    if config['type'] == 'linear':
        return LinearSlotAutoencoder(slot_dim=slot_dim)
    
    elif config['type'] == 'nonlinear':
        return NonlinearSlotAutoencoder(
            slot_dim=slot_dim,
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            activation=config['activation'],
            dropout=config['dropout'],
            use_batchnorm=config['use_batchnorm']
        )
    
    else:
        raise ValueError(f"Unknown model type: {config['type']}")


def list_available_models():
    """사용 가능한 모델 목록 출력"""
    print("=" * 60)
    print("Available Autoencoder Models")
    print("=" * 60)
    for name, config in MODEL_CONFIGS.items():
        print(f"\n[{name}]")
        print(f"  Type: {config['type']}")
        print(f"  Description: {config['description']}")
        if config['type'] == 'nonlinear':
            print(f"  Encoder layers: {config['encoder_layers']}")
            print(f"  Decoder layers: {config['decoder_layers']}")
            print(f"  Activation: {config['activation']}")
            print(f"  Dropout: {config['dropout']}")
            print(f"  BatchNorm: {config['use_batchnorm']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 모델 목록 출력 테스트
    list_available_models()
    
    # 각 모델 생성 테스트
    print("\n\nModel Parameter Counts:")
    print("=" * 60)
    for config_name in MODEL_CONFIGS.keys():
        model = create_autoencoder(config_name, slot_dim=256)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{config_name:20s}: {num_params:,} parameters")
