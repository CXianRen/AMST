#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

cd ../Mart/

SEED=6
LR=0.001
BS=64
EPOCH=100

FUSION=lsum



AUDIO_LR=0.001
VISUAL_LR=0.01
TEXT_LR=8e-5

DATASET="URFUNNY"

python -m baseline.mslr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --audio_lr ${AUDIO_LR} \
          --visual_lr ${VISUAL_LR} \
          --text_lr ${TEXT_LR} \
          --epochs ${EPOCH} \

SEED=7
python -m baseline.mslr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --audio_lr ${AUDIO_LR} \
          --visual_lr ${VISUAL_LR} \
          --text_lr ${TEXT_LR} \
          --epochs ${EPOCH} \

SEED=8
python -m baseline.mslr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --audio_lr ${AUDIO_LR} \
          --visual_lr ${VISUAL_LR} \
          --text_lr ${TEXT_LR} \
          --epochs ${EPOCH} \