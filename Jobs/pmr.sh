#!/bin/env bash
source ../env.sh
source my_venv/bin/activate



SEED=0
LR=0.001
EPOCH=100

FUSION=lsum
DATASET="MVSA"

# FOR CREMAD & AVE, disable tf32, 
# this will effect the perfromance(improve a bit actually)
# Factor, CREMAD and others using the default(0.1), AVE using 1.0 as original paper
cd ../code/
python -m baseline.pmr3_new \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --learning_rate ${LR} \
          --epochs ${EPOCH} \
          --factor 0.1 \
          --no_using_ploader \
          # --no_tf32 \
          # --no_using_ploader \