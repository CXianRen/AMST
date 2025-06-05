#!/bin/env bash
source ../env.sh
source my_venv/bin/activate


cd ../Mart/


SEED=6
LR=0.001
BS=64
EPOCH=100

FUSION=concat
DATASET="IEMOCAP3"

# DATASET="AVE"
# python ../AMST/ogm3_new.py --save_path ../ckpt \
#           --dataset ${DATASET} \
#           --random_seed ${SEED} \
#           --fusion_method ${FUSION} \
#           --epochs ${EPOCH}

python -m baseline.ogm3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \

SEED=7
python -m baseline.ogm3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \

SEED=8
python -m baseline.ogm3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \