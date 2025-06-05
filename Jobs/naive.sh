#!/bin/env bash
# set -x

source ../env.sh
source my_venv/bin/activate


LR=0.001
BS=64
FUSION=lsum
DATASET="MVSA"
EPOCH=100
SEED=0

cd ../code

python -m baseline.naive_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
          --no_test \
          --no_using_ploader
