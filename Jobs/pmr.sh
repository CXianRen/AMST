#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

cd ../Mart/

SEED=6
LR=0.001
BS=64
EPOCH=100

FUSION=concat
DATASET="IEMOCAP"


python -m baseline.pmr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --learning_rate ${LR} \
          --epochs ${EPOCH} 

SEED=7
python -m baseline.pmr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --learning_rate ${LR} \
          --epochs ${EPOCH} 

SEED=8
python -m baseline.pmr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --learning_rate ${LR} \
          --epochs ${EPOCH} 