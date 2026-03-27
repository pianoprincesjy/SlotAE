#!/bin/bash
# Slot Autoencoder Training - COCO Base

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocl

python trainae2.py \
    --model-config nonlinear_deep \
    --epochs 30 \
    --batch-size 512 \
    --metaslot-config /home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/dinosaur_r-coco.py \
    --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-coco256/42/0054.pth \
    --gpu 5
