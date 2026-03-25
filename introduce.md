slot ae란? slot autoencoder
두 개의 slot을 합치고 다시 분리하는 autoencoder를 학습하고 테스트한다
오토인코더의 인코더는 두 slot을 받아 하나의 slot을 출력하고, 디코더는 하나의 slot을 받아 두 slot을 출력한다
활용하는 코드는 metaslot_single_image_experiment.ipynb를 참고한다
다만 현재 config는 슬롯 수가 7이고 aggregator를 일반 slot attention으로 하는 등 약간의 수정이 가해졌으며 현재 config를 따라야한다. coco pretrained model을 사용한다. save폴더 안의 dinosaur_r-coco256의 가중치와 그 config를 사용할 것


autoencoder는 두가지로 분류한다
1. 간단한 선형 변환 인코더, 디코더
2. 비선형 변환 MLP (모델 구성은 자유)
학습 과정:
1. pretrained slot attention과정을 따라간다. 즉 sa는 추가로 학습하지 않는다
2. 만들어진 slot토큰들 중 두개를 골라 autoencoder를 reconstruction error로 학습한다
   1. 하나의 이미지에서 7개 토큰이 나오면 거기서 2개씩 골라 만들 수 있는 모든 쌍에 대해 전부 loss를 구하는 식으로 진행한다
테스트:
1. 시각화를 통해 질적 평가를 한다.
   1. 이미지는 3x2그리드로 사진 4장을 한 png로 저장한다
   2. 0,0에는원본 이미지가 들어가고 그 오른쪽엔 원본 이미지에 alpha약 0.3으로 각 슬롯들을 시각화한다. 이때 이중 두개 슬롯을 고르고 그 두 슬롯은 별표처리를 한다. 그 오른쪽에는 오토인코더의 인코더를 통해 두 슬롯을 하나의 슬롯으로 합치고 총 6개의 슬롯을 시각화한다
   3. 아랫줄의 가장 왼쪽엔 다시 원본 이미지가 들어가고 그 오른쪽엔 위이미지처럼 슬롯을 시각화한다. 이때 슬롯 하나를 골라 거기에 별표처리를 한다. 그 오른쪽엔 오토인코더의 디코더를 통해 그 슬롯을 두 슬롯으로 나누고 총 8개의 슬롯을 시각화한다

코드 형태
오토인코더를 사용하는 모든 코드와 결과물은 slotae폴더 내에 저장한다. 코드는 trainae.py, evalae.py만 작성하며 python trainae.py, python evalae.py만으로 작동 가능해야한다. train의 경우 내부 하드코딩을 통해 linear와 nonlinear를 설정 가능해야한다. 이 설정은 가중치를 저장할 때 이름에 포함한다. eval의 경우 eval폴더 내에 사용한 학습 가중치를 이름으로 하여 png로 저장한다.

공통적으로 cuda visivle devices = 5를 이용한다