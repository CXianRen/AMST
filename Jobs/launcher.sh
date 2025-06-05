#!/bin/bash
set -e 

PROJECT_ID=NAISS2025-22-117
PARTITION=alvis
TIME=0-05:00:00
GPU_TYPE=A100
GPU_NUM=1

export GPU_TYPE=$GPU_TYPE
export GPU_NUM=$GPU_NUM

# get the name of the script
JOB_NAME=$(basename $1)
JOB_NAME=${JOB_NAME%.*}

DATE=$(date +'%Y-%m-%d')
# if $2 is not empty, use it as the job name
if [ -n "$2" ]; then
  JOB_NAME=${JOB_NAME}_$2
fi

mkdir -p logs/$DATE
OUTPUT_PATH=logs/$DATE/${JOB_NAME}_$(date +'%H-%M-%S').out
touch $OUTPUT_PATH
echo "Waiting for job to start..." > $OUTPUT_PATH

echo "Submitting job $JOB_NAME to partition $PARTITION with $GPU_NUM $GPU_TYPE GPUs"
echo "Output will be written to $OUTPUT_PATH"


sbatch \
  -A $PROJECT_ID \
  -p $PARTITION \
  -t $TIME \
  --gpus-per-node=$GPU_TYPE:$GPU_NUM \
  -J $JOB_NAME \
  --output=$OUTPUT_PATH \
  $1 # your script path

# tail the output file
tail -f $OUTPUT_PATH