# clip video

import os
import ffmpeg
from tqdm import tqdm

sample_txt_file = "./iemocap_all_sample.txt"

data_path = "/mimer/NOBACKUP/groups/naiss2024-22-578/IEMOCAP_full_release/"

video_clips_output_path = os.path.join("/tmp/", "MP4")

name_path_map = {
    "Ses01": "Session1",
    "Ses02": "Session2",
    "Ses03": "Session3",
    "Ses04": "Session4",
    "Ses05": "Session5",
}


def extract_video_segment(input_path, output_path, start_time, end_time):

    try:
        ffmpeg.input(input_path, 
                     ss=start_time, 
                     to=end_time
            ).output(output_path, 
            ).run(overwrite_output=True, quiet=True)
        # print(f" {output_path}")
    except Exception as e:
        print(f"{e}")


video_info_list = []
with open(sample_txt_file, 'r') as f:
    lines =  f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        sample_info = line.split("###")[0]
        sample_info = sample_info.strip()
        items = sample_info.split(",")
        name = items[0].strip()
        start_time = float(items[2].strip())
        end_time = float(items[3].strip())
        # print(name, start_time, end_time)
        
        session_name = name.split("_")[0][:5]
        # print(session_name)
        session_path = name_path_map[session_name]
        # print(session_path)
        
        video_name = name.rsplit("_", 1)[0]
        # print(video_name)
        video_path = os.path.join(data_path, 
                                  session_path,
                                  "dialog/avi/DivX",
                                  video_name+".avi")
        # print(video_path)
        output_path = os.path.join(
            video_clips_output_path,
            name + ".mp4"
        )
        # print(output_path)
        extract_video_segment(video_path, output_path, start_time, end_time)
        # break
        
