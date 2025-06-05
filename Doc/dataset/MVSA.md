# MVSA-Single Dataset Usage Guide

## 1. Download Data

Please download the MVSA dataset from [MVSA](https://www.kaggle.com/datasets/vincemarcs/mvsasingle) and unzip it.

Or  

Download our prepared data here:[data prepared](https://drive.google.com/drive/folders/1x9TER3mc1sMgcALp7x_ooK65IjRHOaHN?usp=sharing)
For running, you just need fbank.zip, IMAGE_KEPT_X_PER_SEC.zip
---

## 2. Preprocessing

### 2.1 Set Dataset Path

```sh
export DATASET_PATH=[YOUR_DATASET_PATH]
```
Replace `[YOUR_DATASET_PATH]` with your local dataset path.

---


Each sample in this dataset consists of a single image and its corresponding text.  
No special processing is required for the visual modality.  
For the text modality, it is recommended to tokenize the text in advance to save time during runtime.


### 2.2 Tokenizing
```sh
python data/MVSA/new_extract_mvsa.py
```