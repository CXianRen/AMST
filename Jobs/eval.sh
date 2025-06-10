#!/bin/env bash
# set -x

source ../env.sh
source my_venv/bin/activate


# MODEL_PATH="../ckpt/naive/cremad.pth"
MODEL_PATH="../ckpt/OGM_lsum_AVE_0/best_model.pth"

cd ../code


python3 -m evaluate.eval_mm \
  --model_path ${MODEL_PATH}