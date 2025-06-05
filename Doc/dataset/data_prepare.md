Using scripts:
```sh
# generator
# gen_cre_ttv.py:   to split CREMA-D into training, testing set, validating set 

# converter
# extract_spec.py:  to convert audio into spectrum format. (Not used in MLA)
# extract_fbank.py: to convert audio into fbank format
# video_preprocessing.py: to convert video into images
```

## CREMA-D

### Step 1 generating 
(you might shkip this step, and use ours)  
For more details about parameters, ref [gen_cre_ttv.py](../AMST/data/CREMAD/gen_cre_ttv.py)
```sh
# python3 AMST/data/CREMAD/gen_cre_ttv.py [YOR_PATH]/CREMAD [YOR_PATH] 0.8 0.1 0.1 0
```

### Step 2 generating fbank data and save as .npy
```sh
# mkdir -p [YOUR_CREMAD_PATH]/fbank/
# python3 AMST/data/scripts/extract_fbank.py [YOUR_CREMAD_PATH]/AudioWAV/ [YOUR_CREMAD_PATH]/fbank/
```

### Step 3 conver flv/mp4 to images
```sh
# python3 AMST/data/scripts/video_preprocessing.py [YOUR_CREMAD_PATH]/VideoFlash/ [YOUR_OUTPUT_PATH]/ 1 ".flv"
# 1: keep 1 frame per second
# ".flv" video file ends with ".flv"

# eg. 
python3 AMST/data/scripts/video_preprocessing.py /scratch/dataset/CREMAD/VideoFlash/ /scratch/dataset/CREMAD/IMAGE_KEPT_1_PER_SEC/ 1 ".flv"
```

## AVE
### Step 1 spliting audio from mp4 files
```sh
# python3 AMST/data/scripts/mp4_to_wav.py [YOUR_AVE_PATH]/AVE/ [YOUR_AVE_PATH]/WAV/
# eg.
python3 AMST/data/scripts/mp4_to_wav.py /scratch/dataset/AVE/AVE/ /scratch/dataset/AVE/WAV/
```
And the remaining steps are same as that of CREMA-D

```sh
# gen fbank
mkdir /scratch/dataset/AVE/fbank/
python3 AMST/data/scripts/extract_fbank.py /scratch/dataset/AVE/WAV/ /scratch/dataset/AVE/fbank/

# gen image
python3 AMST/data/scripts/video_preprocessing.py /scratch/dataset/AVE/AVE /scratch/dataset/AVE/IMAGE_KEPT_1_PER_SEC/ 1 ".mp4"
```

## IEMOCAP
Download our prepared data here:[data prepared](https://drive.google.com/drive/folders/1x9TER3mc1sMgcALp7x_ooK65IjRHOaHN?usp=sharing)
For running, you just need fbank.zip, IMAGE_KEPT_X_PER_SEC.zip
```sh
#1. use gen_iemo_dataset.py to generat iemocap_all_sample.txt

#2. use extract_video_clips.py to generate video clips from raw videos

#3. use mp4_to_wav.py to generate wav files
#       eg.python3 AMST/data/scripts/mp4_to_wav.py ~/MP4 ~/WAV
#3.1 convert to fbank
#       eg.mkdir ~/fbank/
#          python3 AMST/data/scripts/extract_fbank.py ~/WAV/ ~/fbank/

#4. use video_processing.py to generate imgs
#       eg.python3 AMST/data/scripts/video_preprocessing.py ~/MP4 ~/IMAGE_KEPT_2_PER_SEC/ 2 ".mp4"

#5. (TODO?) use split_imgs.py to split imgs into 2 parts: speaking one and listening one. (for each sample, we will use the speaking one for aligning with text modality)

```

## UR-FUNNY
ref:https://github.com/ROC-HCI/UR-FUNNY/blob/master/README.md

Most steps are same as other datasets
```sh
export DATA_PATH="/mimer/NOBACKUP/groups/naiss2024-22-578/UR-FUNNY"
#1. use gen_urfunny_txt.py to generate ur_funny_x.txt

#2. use mp4_to_wav.py to generate wav files
AMST/data/scripts/mp4_to_wav.py ${DATA_PATH}/urfunny2_videos/ ${DATA_PATH}/WAV
#2.1 convert to fbank
mkdir ${DATA_PATH}/fbank/
python3 AMST/data/scripts/extract_fbank.py ${DATA_PATH}/WAV/ ${DATA_PATH}/fbank/

#3. use video_processing.py to generate imgs
python3 AMST/data/scripts/video_preprocessing.py ${DATA_PATH}/urfunny2_videos/ ${DATA_PATH}/IMAGE_KEPT_1_PER_SEC/ 1 ".mp4"

#4. extract text feature using bert pretrain model
cd AMST/data/UR-FUNNY
python extract_text_tokens.py
```