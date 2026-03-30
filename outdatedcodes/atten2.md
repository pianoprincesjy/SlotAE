# MetaSlot Attention Mechanism 분석 보고서

**작성일**: 2026-03-27  
**분석 대상**: MetaSlot (DINOSAUR) 모델의 Attention 생성 메커니즘

---

## 1. 개요

MetaSlot 모델은 두 가지 서로 다른 attention map을 생성합니다:
- **`attent`**: Slot Attention 모듈에서 생성 (저해상도, 16×16)
- **`attent2`**: Decoder에서 생성 (고해상도, 256×256)

본 보고서는 `attent2`의 생성 메커니즘과 활용 방법을 분석합니다.

---

## 2. `attent2` 생성 과정

### 2.1 모델 구조 (`dinosaur.py`)

```python
# Forward pass (dinosaur.py, line 47-82)
feature = encode_backbone(input)  # DINO feature extraction
encode = encode_posit_embed(encode)
encode = encode_project(encode)

query = initializ(b)  # Initialize slots
slotz, attent = aggregat(encode, query)  # Slot Attention
# attent: (B, N, 16, 16) - 저해상도

clue = [h, w]  # h=16, w=16 (feature map resolution)
recon, attent2 = decode(clue, slotz)  # BroadcastMLPDecoder
# attent2: (B, N, 256, 256) - 고해상도
```

### 2.2 BroadcastMLPDecoder 구조 (`dinosaur.py`, line 151-179)

```python
class BroadcastMLPDecoder(nn.Module):
    def forward(self, input, slotz):
        h, w = input  # [16, 16]
        b, n, c = slotz.shape  # [B, 7, 256]
        
        # Step 1: Slot을 모든 픽셀 위치로 broadcast
        mixture = repeat(slotz, "b n c -> (b n) hw c", hw=h * w)
        
        # Step 2: Positional embedding 추가
        mixture = self.posit_embed(mixture)
        
        # Step 3: MLP backbone 통과
        # MLP config: [256] → [2048, 2048, 2048, 385]
        #             ↑                            ↑
        #          slot dim         384 (feature) + 1 (mask logit)
        mixture = self.backbone(mixture)
        
        # Step 4: Feature와 Mask logit 분리
        recon = mixture[:, :, :-1]   # (B*N, H*W, 384)
        alpha = mixture[:, :, -1:]   # (B*N, H*W, 1) ← 핵심!
        
        # Step 5: Mask logit을 softmax로 정규화
        alpha = alpha.softmax(1)  # over slots dimension
        # → attent2: (B, N, H*W)
        
        # Step 6: Attention-weighted reconstruction
        recon = (recon * alpha).sum(1)
        
        return recon, alpha  # alpha = attent2
```

### 2.3 핵심 포인트

1. **MLP 출력 차원**: `[2048, 2048, 2048, 385]`
   - 384차원: 재구성용 feature
   - **1차원: Mask logit** (각 slot이 해당 픽셀을 차지할 확률)

2. **Mask logit → attent2 변환**:
   ```python
   alpha = mixture[:, :, -1:]  # 마지막 1차원 추출
   alpha = alpha.softmax(1)    # Slot 차원에서 softmax
   ```
   - Softmax를 통해 각 픽셀에서 모든 slot의 확률 합이 1이 됨
   - Winner-take-all 방식의 세그멘테이션에 적합

3. **해상도 차이**:
   - `attent`: Slot Attention의 부산물 (16×16)
   - `attent2`: Decoder가 명시적으로 학습한 마스크 (256×256)
   - **attent2가 더 정확한 세그멘테이션 제공**

---

## 3. Config 설정 (`dinosaur_r-coco.py`)

```python
decode=dict(
    type="BroadcastMLPDecoder",
    posit_embed=dict(
        type="LearntPositionalEmbedding",
        resolut=[16 * 16],  # 256 pixels
        embed_dim=256,
    ),
    backbone=dict(
        type="MLP",
        in_dim=256,
        dims=[2048, 2048, 2048, 384 + 1],  # ← +1이 mask logit!
        ln=None,
        dropout=0,
    ),
),
```

---

## 4. 활용 방법

### 4.1 기존 방식 (evalae3.py)
```python
# 새로운 slots 생성
merged_slots = autoencoder.encode(slot1, slot2)

# Slot Attention으로 refinement
refined_slots, attention = aggregat(features, merged_slots)

# 저해상도 attention을 업샘플링
attention_256 = F.interpolate(attention, size=(256, 256), mode='bilinear')
```

**문제점**:
- Slot Attention은 feature와의 상호작용을 재계산 (시간 소요)
- 16×16 → 256×256 업샘플링으로 인한 정밀도 손실

### 4.2 제안 방식 (evalae4.py)
```python
# 새로운 slots 생성
merged_slots = autoencoder.encode(slot1, slot2)

# Decoder를 직접 호출하여 고해상도 attention 획득
clue = [16, 16]  # feature map resolution
recon, attent2 = metaslot_model.m.decode(clue, merged_slots)

# attent2는 이미 256×256 고해상도!
attent2_256 = rearrange(attent2, "b n (h w) -> b n h w", h=256)
```

**장점**:
1. **속도**: Slot Attention 반복 없이 단일 forward pass
2. **정밀도**: Decoder가 학습한 고해상도 마스크 직접 사용
3. **일관성**: 원본 MetaSlot의 attent2와 동일한 방식

---

## 5. 실험적 검증

### 5.1 출력 비교
```python
# Original MetaSlot output
output = metaslot_model(batch)
# output.keys(): ['feature', 'slotz', 'attent', 'attent2', 'recon']

# attent shape:  (1, 7, 16, 16)   - Slot Attention
# attent2 shape: (1, 7, 256, 256) - Decoder mask
```

### 5.2 시각화 우선순위 (노트북 코드)
```python
# metaslot_single_image_experiment.ipynb
if 'attent2' in output:
    attn_maps = output['attent2']  # Decoder 출력 우선!
elif 'attent' in output:
    attn_maps = output['attent']
```

→ 노트북도 attent2를 우선적으로 사용 (더 정확하기 때문)

---

## 6. 결론

### 핵심 발견
1. **Decoder는 mask logit을 명시적으로 생성**합니다 (MLP 마지막 1차원)
2. **attent2 = softmax(mask logit)**으로 고해상도 세그멘테이션 제공
3. **새로운 slots → Decoder 통과**만으로 attent2 획득 가능

### evalae4.py 구현 방향
```python
# Step 1: Autoencoder로 slots 변형
merged_slots = autoencoder.encode(slot1, slot2)

# Step 2: Decoder 직접 호출
recon, attent2_merged = decoder([16, 16], merged_slots)

# Step 3: attent2를 256×256로 reshape
attent2_merged = rearrange(attent2_merged, "b n (h w) -> b n h w", h=256)

# Step 4: 시각화
visualize_with_attent2(original_image, attent2_merged)
```

### 예상 효과
- ✅ **더 빠른 처리**: Slot Attention 반복 생략
- ✅ **더 정확한 마스크**: Decoder가 학습한 고해상도 직접 사용
- ✅ **원본 일관성**: MetaSlot의 attent2와 동일한 메커니즘

---

## 7. 참고 코드 위치

- Decoder 구현: `object_centric_bench/model/dinosaur.py` (line 151-179)
- Forward pass: `object_centric_bench/model/dinosaur.py` (line 47-82)
- Config: `Config/config-metaslot/dinosaur_r-coco.py` (line 96-107)
- 노트북 사용: `metaslot_single_image_experiment.ipynb` (Cell 10)
