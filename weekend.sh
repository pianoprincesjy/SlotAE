#!/bin/bash
# Weekend Experiments - All Combinations
# 5 models x 3 batch sizes x 2 pretrains = 30 experiments

MODELS=(linear nonlinear_simple nonlinear_medium nonlinear_deep nonlinear_gelu)
BATCH_SIZES=(64 256 512)

# COCO experiments
for MODEL in "${MODELS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        echo "========== COCO: ${MODEL} / Batch ${BATCH} =========="
        python trainae2.py \
            --model-config ${MODEL} \
            --epochs 30 \
            --batch-size ${BATCH} \
            --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-coco.py \
            --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth \
            --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_coco \
            --gpu 5
    done
done

# ClevrTex experiments
for MODEL in "${MODELS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        echo "========== ClevrTex: ${MODEL} / Batch ${BATCH} =========="
        python trainae2.py \
            --model-config ${MODEL} \
            --epochs 30 \
            --batch-size ${BATCH} \
            --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py \
            --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth \
            --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_clevrtex \
            --gpu 5
    done
done

echo "All experiments completed!"
