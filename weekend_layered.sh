#!/bin/bash
# Layered Linear Models Training - 2 to 10 layers
# Feature-matching loss 사용
# 9 models x 2 datasets = 18 experiments

MODELS=(linear_layered_2 linear_layered_3 linear_layered_4 linear_layered_5 linear_layered_6 linear_layered_7 linear_layered_8 linear_layered_9 linear_layered_10)
BATCH_SIZE=256
EPOCHS=20

# COCO experiments
# metaslot-config에 데이터셋 정보가 포함되어 있음 (dinosaur_r-coco.py → COCO 데이터셋 사용)
# echo "=========================================="
# echo "Starting COCO dataset experiments..."
# echo "=========================================="
# for MODEL in "${MODELS[@]}"; do
#     echo "========== COCO: ${MODEL} =========="
#     python trainae_layered.py \
#         --model-config ${MODEL} \
#         --epochs ${EPOCHS} \
#         --batch-size ${BATCH_SIZE} \
#         --feature-match-weight 0.1 \
#         --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-coco.py \
#         --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth \
#         --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_layered_coco \
#         --gpu 5
# done

# ClevrTex experiments
# metaslot-config에 데이터셋 정보가 포함되어 있음 (dinosaur_r-clevrtex.py → ClevrTex 데이터셋 사용)
echo ""
echo "=========================================="
echo "Starting ClevrTex dataset experiments..."
echo "=========================================="
for MODEL in "${MODELS[@]}"; do
    echo "========== ClevrTex: ${MODEL} =========="
    python trainae_layered.py \
        --model-config ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --feature-match-weight 0.1 \
        --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py \
        --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth \
        --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_layered_clevrtex \
        --gpu 4
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Total: ${#MODELS[@]} models x 2 datasets = $((${#MODELS[@]} * 2)) experiments"
