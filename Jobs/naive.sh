#!/bin/env bash
# set -x

source ../env.sh
source my_venv/bin/activate


FUSION=lsum
DATASET="AVE"
EPOCH=100
SEED=1

cd ../code

# FOR CREMAD & AVE, disable tf32, 
# this will effect the perfromance(improve a bit actually)

python -m baseline.naive_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
        #   --no_tf32 \
        #   --no_using_ploader \
