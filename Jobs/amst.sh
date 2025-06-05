#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

cd ../Mart/


LR=0.001
BS=64
EPOCH=100

FUSION=film


DATASET="AVE"
# CREMAD 5, AVE 2, IEMOCAP2 4 IEMOCAP3 4, MVSA, URFUNNY 6
A_F=2
# A_F=5
# A_F=4
# A_F=6

# DEFAULT 1,  
# V_F = 

# T_F = 10


SEED=6
python -m baseline.amst \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
          --a_skip_factor ${A_F} \
          --v_skip_factor 1 \
          # --t_skip_factor 1 \

SEED=7
python -m baseline.amst \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
          --a_skip_factor ${A_F} \
          --v_skip_factor 1 \

SEED=8
python -m baseline.amst \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method ${FUSION} \
          --epochs ${EPOCH} \
          --a_skip_factor ${A_F} \
          --v_skip_factor 1 \