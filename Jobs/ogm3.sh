#!/bin/env bash
source ../env.sh
source my_venv/bin/activate





SEED=10
EPOCH=100

FUSION=lsum
DATASET="AVE"

cd ../code/
# FOR CREMAD & AVE, disable tf32, 
# this will effect the perfromance(improve a bit actually)
python -m baseline.ogm3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
          --no_tf32 \
          --no_using_ploader \
          