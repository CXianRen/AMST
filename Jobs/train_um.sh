#!/bin/env bash
# set -x

source ../env.sh
source my_venv/bin/activate


CKPT="../ckpt/"
SEED=16
LR=0.01
BS=64
EPOCHS=10


run_naive() {
    echo "-------------- ${DATASET} ${MODALITY} --------------"
 
    cd ../Mart
    python  -m baseline.single_modality \
                  --dataset ${DATASET} \
                    --modality ${MODALITY} \
                    --random_seed ${SEED} \
                    --learning_rate ${LR} \
                    --save_path ${CKPT}\
                    --batch_size ${BS} \
                    --epochs ${EPOCHS} \

    cd -
}

# CREMAD
DATASET="CREMAD"
MODALITY="visual"
run_naive

MODALITY="audio"
run_naive

# AVE
DATASET="AVE"
MODALITY="visual"  # audio, visual
run_naive

MODALITY="audio"
run_naive
# MVSA
DATASET="MVSA"
MODALITY="visual"
run_naive
MODALITY="text"
run_naive
# IEMOCAP3
DATASET="IEMOCAP3"
MODALITY="audio"
run_naive
MODALITY="visual"
run_naive
MODALITY="text"
run_naive
# URFUNNY
DATASET="URFUNNY"
MODALITY="visual"
run_naive
MODALITY="audio"
run_naive
MODALITY="text"
run_naive

