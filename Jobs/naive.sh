#!/bin/env bash
# set -x

source ../env.sh
source my_venv/bin/activate


FUSION=lsum
DATASET="CREMAD"
EPOCH=100
SEED=0

cd ../code


python -m baseline.naive_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
          --no_tf32 \
          --no_using_ploader \

# python temp.py --random_seed 2