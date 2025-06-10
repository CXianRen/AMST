#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

cd ../code/

SEED=0
LR=0.001
BS=64
EPOCH=100

FUSION=lsum


AUDIO_LR=0.001
VISUAL_LR=0.001
TEXT_LR=0.001

DATASET="AVE"

python -m baseline.mslr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --audio_lr ${AUDIO_LR} \
          --visual_lr ${VISUAL_LR} \
          --text_lr ${TEXT_LR} \
          --epochs ${EPOCH} \
          --no_tf32 \
          # --no_using_ploader \