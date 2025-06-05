#!/bin/bash

set -e 
set -x 


ml purge
ml load virtualenv/20.23.1-GCCcore-12.3.0
ml load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
ml load librosa/0.10.1-foss-2023a
ml load tensorboard/2.15.1-gfbf-2023a
ml load tqdm/4.66.1-GCCcore-12.3.0
ml load tensorboardX/2.6.2.2-foss-2023a
ml load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
ml load Transformers/4.39.3-gfbf-2023a
ml load matplotlib/3.7.2-gfbf-2023a

rm -rf Jobs/my_venv
virtualenv Jobs/my_venv

source Jobs/my_venv/bin/activate

pip list

pip install -r requirement.txt


