#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

SEED=0
LR=0.001
BS=64

DATASET="AVE"
EPOCH=100

cd ../code/

# DATASET="URFUNNY"

python -m baseline.mla_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --epochs ${EPOCH} \
          # --no_tf32 \
          # --no_using_ploader \