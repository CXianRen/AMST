# CREMAD Dataset Usage Guide

## 1. Download Data

Please download the CREMAD dataset from [CREMAD GitHub](https://github.com/CheyneyComputerScience/CREMA-D) and unzip it.

---

## 2. Preprocessing

### 2.1 Set Dataset Path

```sh
export DATASET_PATH=[YOUR_DATASET_PATH]
```
Replace `[YOUR_DATASET_PATH]` with your local CREMAD dataset path.

---

### 2.2 Generate fbank Features and Save as .npy

```sh
mkdir -p ${DATASET_PATH}/fbank/
python3 data/scripts/extract_fbank.py ${DATASET_PATH}/AudioWAV/ ${DATASET_PATH}/fbank/
```

---

### 2.3 Convert flv/mp4 Videos to Images

```sh
mkdir -p ${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/
python3 data/scripts/video_preprocessing.py ${DATASET_PATH}/VideoFlash/ ${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/ 1 ".flv"

# - `${DATASET_PATH}/VideoFlash/`: input video folder
# - `${DATASET_PATH}/IMAGE_KEPT_1_PER_SEC/`: output image folder
# - `1`: keep 1 frame per second
# - `".flv"`: process video files ending with `.flv`
```


