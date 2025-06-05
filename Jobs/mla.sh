#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

SEED=0
LR=0.001
BS=64


DATASET="CREMAD"
EPOCH=100

cd ../Mart/

# DATASET="URFUNNY"

python -m baseline.mla_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --epochs ${EPOCH} \


SEED=1
python -m baseline.mla_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --epochs ${EPOCH} \

SEED=2
python -m baseline.mla_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --epochs ${EPOCH} \