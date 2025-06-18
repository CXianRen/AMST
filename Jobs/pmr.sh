#!/bin/env bash
source ../env.sh
source my_venv/bin/activate



SEED=0
LR=0.001
EPOCH=100

FUSION=lsum
DATASET="CREMAD"
FACTOR=0.1 # Factor for AVE is 1.0, for others is 0.1

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
          --factor ${FACTOR} \
          --no_using_ploader \
          --no_tf32 \