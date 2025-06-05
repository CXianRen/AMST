# AVE Dataset Usage Guide

## 1. Download Data

Please download the AVE dataset from [AVE](https://github.com/YapengTian/AVE-ECCV18) and unzip it.

---


## 2. Preprocessing

### 2.1 Set Dataset Path

```sh
export DATASET_PATH=[YOUR_DATASET_PATH]
```
Replace `[YOUR_DATASET_PATH]` with your local dataset path.

---


## Spliting audio from mp4 files
```sh
mkdir ${DATASET_PATH}/WAV/
python3 data/scripts/mp4_to_wav.py ${DATASET_PATH}/AVE/ ${DATASET_PATH}/WAV/
```

```sh
# gen fbank
mkdir ${DATASET_PATH}/fbank/
python3 data/scripts/extract_fbank.py ${DATASET_PATH}/WAV/ ${DATASET_PATH}/fbank/

# gen image
mkdir ${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/
python3 data/scripts/video_preprocessing.py ${DATASET_PATH}/AVE/ ${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/ 1 ".mp4"

# - `${DATASET_PATH}/VideoFlash/`: input video folder
# - `${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/`: output image folder
# - `1`: keep 1 frame per second
# - `".flv"`: process video files ending with `.flv`
```

