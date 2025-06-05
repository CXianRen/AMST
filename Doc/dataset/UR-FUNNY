# IEMOCAP Dataset Usage Guide

## 1. Download Data

Please download the dataset from [Link](https://github.com/ROC-HCI/UR-FUNNY/blob/master/README.md) and unzip it.

---

## 2. Preprocessing

Most steps are same as other datasets
```sh
export DATASET_PATH=`[YOU_DATASET_PATH]`
#1. use gen_urfunny_txt.py to generate ur_funny_x.txt

#2. use mp4_to_wav.py to generate wav files
data/scripts/mp4_to_wav.py ${DATASET_PATH}/urfunny2_videos/ ${DATASET_PATH}/WAV
#2.1 convert to fbank
mkdir ${DATASET_PATH}/fbank/
python3 data/scripts/extract_fbank.py ${DATASET_PATH}/WAV/ ${DATASET_PATH}/fbank/

#3. use video_processing.py to generate imgs
python3 data/scripts/video_preprocessing.py ${DATASET_PATH}/urfunny2_videos/ ${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/ 1 ".mp4"

#4. extract text feature using bert pretrain model
cd data/UR-FUNNY
python extract_text_tokens.py
```