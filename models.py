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
    'linear_layered_2': {
        'type': 'linear_layered',
        'description': '2-layer gradual linear transformation',
        'num_layers': 2,
    },
    'linear_layered_3': {
        'type': 'linear_layered',
        'description': '3-layer gradual linear transformation',
        'num_layers': 3,
    },
    'linear_layered_4': {
        'type': 'linear_layered',
        'description': '4-layer gradual linear transformation',
        'num_layers': 4,
    },
    'linear_layered_5': {
        'type': 'linear_layered',
        'description': '5-layer gradual linear transformation',
        'num_layers': 5,
    },
    'linear_layered_6': {
        'type': 'linear_layered',
        'description': '6-layer gradual linear transformation',
        'num_layers': 6,
    },
    'linear_layered_7': {
        'type': 'linear_layered',
        'description': '7-layer gradual linear transformation',
        'num_layers': 7,
    },
    'linear_layered_8': {
        'type': 'linear_layered',
        'description': '8-layer gradual linear transformation',
        'num_layers': 8,
    },
    'linear_layered_9': {
        'type': 'linear_layered',
        'description': '9-layer gradual linear transformation',
        'num_layers': 9,
    },
    'linear_layered_10': {
        'type': 'linear_layered',
        'description': '10-layer gradual linear transformation',
        'num_layers': 10,
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


class LinearLayeredSlotAutoencoder(nn.Module):
    """
    Multi-layer linear autoencoder with gradual dimension reduction
    각 레이어의 feature도 저장하여 feature-matching loss 계산 가능
    """
    def __init__(self, slot_dim=256, num_layers=4):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_layers = num_layers
        
        # 차원 계산: 512 -> ... -> 256 (점진적으로 감소)
        input_dim = slot_dim * 2
        output_dim = slot_dim
        
        # 각 레이어의 차원 계산
        dims = self._compute_layer_dims(input_dim, output_dim, num_layers)
        self.encoder_dims = dims
        self.decoder_dims = dims[::-1]  # 역순
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
        
        # Decoder layers (symmetric)
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]))
        
    def _compute_layer_dims(self, start_dim, end_dim, num_layers):
        """레이어별 차원 계산 (등간격으로)"""
        dims = [start_dim]
        step = (start_dim - end_dim) / num_layers
        for i in range(1, num_layers):
            dim = int(start_dim - step * i)
            dims.append(dim)
        dims.append(end_dim)
        return dims
    
    def encode(self, slot1, slot2, return_intermediates=False):
        """
        두 slot을 하나로 합침
        return_intermediates=True이면 중간 feature들도 반환
        """
        combined = pt.cat([slot1, slot2], dim=-1)
        
        intermediates = [combined]
        x = combined
        for layer in self.encoder_layers:
            x = layer(x)
            intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def decode(self, encoded_slot, return_intermediates=False):
        """
        하나의 slot을 두 개로 분리
        return_intermediates=True이면 중간 feature들도 반환
        """
        intermediates = [encoded_slot]
        x = encoded_slot
        for layer in self.decoder_layers:
            x = layer(x)
            intermediates.append(x)
        
        slot1_recon = x[..., :self.slot_dim]
        slot2_recon = x[..., self.slot_dim:]
        
        if return_intermediates:
            return slot1_recon, slot2_recon, intermediates
        return slot1_recon, slot2_recon
    
    def forward(self, slot1, slot2, return_intermediates=False):
        """
        Forward pass
        return_intermediates=True이면 인코더/디코더 중간 feature들도 반환
        """
        if return_intermediates:
            encoded, enc_intermediates = self.encode(slot1, slot2, return_intermediates=True)
            slot1_recon, slot2_recon, dec_intermediates = self.decode(encoded, return_intermediates=True)
            return slot1_recon, slot2_recon, encoded, enc_intermediates, dec_intermediates
        else:
            encoded = self.encode(slot1, slot2)
            slot1_recon, slot2_recon = self.decode(encoded)
            return slot1_recon, slot2_recon, encoded


# ==================== Model Factory ====================

def create_autoencoder(config_name='nonlinear_simple', slot_dim=256):
    """
    Autoencoder 모델 생성 팩토리 함수
    
    Args:
        config_name: MODEL_CONFIGS의 키 (예: 'linear', 'linear_layered_4', 'nonlinear_simple', 'nonlinear_deep')
        slot_dim: slot의 차원 (기본: 256)
    
    Returns:
        autoencoder 모델
    
    Examples:
        >>> # 선형 모델
        >>> model = create_autoencoder('linear', slot_dim=256)
        
        >>> # 4-layer 선형 모델 (feature-matching loss 지원)
        >>> model = create_autoencoder('linear_layered_4', slot_dim=256)
        
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
    
    elif config['type'] == 'linear_layered':
        return LinearLayeredSlotAutoencoder(
            slot_dim=slot_dim,
            num_layers=config['num_layers']
        )
    
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
        if config['type'] == 'linear_layered':
            print(f"  Num layers: {config['num_layers']}")
        elif config['type'] == 'nonlinear':
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
