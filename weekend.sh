#!/bin/bash
# Weekend Experiments - ClevrTex Model (Batch 64, 256, 512)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocl

# Batch 64
python trainae2.py \
    --model-config nonlinear_deep \
    --epochs 30 \
    --batch-size 64 \
    --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py \
    --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth \
    --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_clevrtex \
    --gpu 5

# Batch 256
python trainae2.py \
    --model-config nonlinear_deep \
    --epochs 30 \
    --batch-size 256 \
    --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py \
    --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth \
    --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_clevrtex \
    --gpu 5

# Batch 512
python trainae2.py \
    --model-config nonlinear_deep \
    --epochs 30 \
    --batch-size 512 \
    --metaslot-config /home/jaey00ns/MetaSlot-main/Config/config-metaslot/dinosaur_r-clevrtex.py \
    --metaslot-checkpoint /home/jaey00ns/MetaSlot-main/save/dinosaur_r-clevrtex/42/0031.pth \
    --save-dir /home/jaey00ns/MetaSlot-main/slotae/pth_clevrtex \
    --gpu 5
