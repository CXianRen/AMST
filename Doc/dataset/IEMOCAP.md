# IEMOCAP Dataset Usage Guide

## 1. Download Data

Please download the dataset from [Link](https://sail.usc.edu/iemocap/iemocap_info.html) and unzip it.

---

## 2. Preprocessing

### 2.1 Preparing Data
```sh
#1. use gen_iemo_dataset.py to generat iemocap_all_sample.txt

#2. use extract_video_clips.py to generate video clips from raw videos

#3. use mp4_to_wav.py to generate wav files
#       eg.python3 data/scripts/mp4_to_wav.py ~/MP4 ~/WAV
#3.1 convert to fbank
#       eg.mkdir ~/fbank/
#          python3 data/scripts/extract_fbank.py ~/WAV/ ~/fbank/

#4. use video_processing.py to generate imgs
#       eg.python3 data/scripts/video_preprocessing.py ~/MP4 ~/IMAGE_KEPT_2_PER_SEC/ 2 ".mp4"

#5. using data/IEMOCAP/extract_text_tokens.py to generate text token
```


# More About dataset
+ Modalities- Audio, video, Motion Capture
+ Emotional content (10)- 
**angry, happy, sad, neutral, frustrated, excited, fearful, disgusted, excited, other**

**ten actors** were recorded in dyadic sessions (5 sessions with 2 subjects each). They were asked to
 perform three selected scripts with clear emotional content.


# File structure
```sh
├── Documentation
│   ├── corpus.dic
│   ├── FIVE_face_markers2.png
│   ├── HumaineInfo.txt
│   ├── phonemes.txt
│   └── timeinfo.txt
├── README.txt
├── Session1    # 5 Session, each session includes 2 person (1 male, 1 female)
│   ├── dialog
│   │   ├── avi               # raw video data (7 improvise + 7 script) * 2 = 28 videos
│   │   ├── EmoEvaluation     # xxx.txt are lable file
│   │   ├── lab               
│   │   ├── MOCAP_hand        # hand motion
│   │   ├── MOCAP_head        # head motion
│   │   ├── MOCAP_rotated     # motion
│   │   ├── transcriptions    # text data
│   │   └── wav               # raw wav data
│   └── sentences
│       ├── ForcedAlignment
│       ├── MOCAP_hand
│       ├── MOCAP_head
│       ├── MOCAP_rotated
│       └── wav
├── Session2
```

# Label example
```sh
[140.2250 - 145.6000]	Ses01F_impro05_F019	ang	[1.5000, 4.0000, 4.5000]
C-E1:	Anger;	()
C-E3:	Anger;	()
C-E4:	Anger;	()
C-F1:	Anger;	()
A-E3:	val 2; act 4; dom  4;	()
A-E4:	val 1; act 4; dom  5;	(threatening, belligerent)
A-F1:	val 1; act 5; dom  5;	()
```

# processing pipeline:
+ 1. Only focus on dialog
+ 2. Read label txt under dialog/EmoEvaluaion
+ 3. Get each sample(turn) info: start time to end time, also label
+ 4. Clip video and audio clips from raw audios
+ 5. Same process as CREMAD and AVE


As original data distrubution like below:
```sh
Class, Number
ang : 1103
dis : 2
exc : 1041
fea : 40
fru : 1849
hap : 595
neu : 1708
oth : 3
sad : 1084
sur : 107
xxx : 2507
```

And MLA only uses: 
```sh
#        train    test
Neutral  1014     337
Angry    685      200
Sad      642      230
Happy    977      340
```