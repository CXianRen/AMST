#!/bin/env bash
source ../env.sh
source my_venv/bin/activate

cd ../code/

LR=0.001
BS=64
EPOCH=100
SEED=10


# CREMAD 5, AVE 2, IEMOCAP2 4 IEMOCAP3 4, MVSA, URFUNNY 6
# A_F=2  # AVE
# A_F=5
# A_F=4
# A_F=6

# V_F = 1 DEFAULT FOR ALL DATASETS

# T_F = 10
# T_F = 10

# AVE
DATASET="AVE"
A_F=2
V_F=1
T_F=0  # T_F is not used in this dataset


# CREMAD
# DATASET="CREMAD"
# A_F=5
# V_F=1
# T_F=0  # T_F is not used in this dataset


python amst_joint.py \
          --save_path ../ckpt \
          --dataset ${DATASET} \
          --random_seed ${SEED} \
          --fusion_method concat \
          --epochs ${EPOCH} \
          --a_skip_factor ${A_F} \
          --v_skip_factor ${V_F} \
          --t_skip_factor ${T_F} \
          --no_tf32 \
          --no_using_ploader \