#   This script is used to convert mp4 files to wav files.
#   The input is the path to the folder containing mp4 files.
#   The output is the path to the folder where the wav files will be saved.
#   The script will convert all mp4 files in the input folder to wav files and save them in the output folder.
#   The script uses ffmpeg to extract audio from the mp4 files.
#   The script is used as follows:
#       python mp4_to_wav.py path/to/mp4/files path/to/save/wav/files
#   Example:
#       python3 AMST/data/scripts/mp4_to_wav.py /scratch/dataset/AVE/AVE/ /scratch/dataset/AVE/WAV/

import os
from tqdm import tqdm
import ffmpeg

def extract_audio_from_video(video_file, output_audio_file):
        ffmpeg.input(video_file).output(
                output_audio_file, 
                q='0', 
                map='a', 
                loglevel='quiet').run()

def process(video_files_path, wav_save_path):
    videos = [video for video in os.listdir(video_files_path) 
              if video.endswith('.mp4')]
    
    # create the save path if not exists
    if not os.path.exists(wav_save_path):
        os.makedirs(wav_save_path)

    print("Number of videos: ", len(videos))
    for video_path in tqdm(videos):
        video_path = video_path.strip()
        video_name = video_path.split('/')[-1].split('.')[0]
        wav_file_path = os.path.join(wav_save_path, video_name+'.wav')
        # force to overwrite
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)
        extract_audio_from_video(os.path.join(video_files_path, video_path), wav_file_path)

if __name__ == "__main__":
    import sys
    video_files_path = sys.argv[1]
    wav_save_path = sys.argv[2]
    process(video_files_path, wav_save_path)
    print("Done! check the save path: ", wav_save_path)