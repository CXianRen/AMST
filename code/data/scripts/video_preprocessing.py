# Usage: python video_preprocessing.py <videos_path> <output_path> 1
# Example: python video_preprocessing.py CREMAD/VideoFlash CREMAD/IMAGE_1PERSEC 1
# Parameters:
#   videos_path: path to the video folder
#   output_path: path to the output folder
#   1: keep 1 frame(s) per second
#   video_type: "[ .flv | .mp4 ]"

# import pandas as pd
import cv2
import os
# import pdb
from tqdm import tqdm

class videoReader(object):
    def __init__(self, video_path):
        self.video_path = video_path

        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames/self.fps)
        
        assert self.video_frames > 0, "Video not found: {}".format(self.video_path)


    def video2frame(self, frame_interval, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count +=1
            if count % frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()


    def video2frame_update(self, frame_save_path, frame_kept_per_second):
        self.frame_save_path = frame_save_path

        count = 0
        frame_interval = int(self.fps/ frame_kept_per_second)
        while(count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id<frame_interval*frame_kept_per_second and frame_id%frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            count += 1
        
        # check image number
        image_files = os.listdir(self.frame_save_path)
        assert len(image_files) > 0, "No image extracted: {}".format(self.frame_save_path)


class Handler(object):
    def __init__(self, videos_path, output_path, frame_kept_per_second= 1, video_type=".flv"):
        """
            videos_path: eg. CREMAD/Video
            frame_kept_per_second: eg. 1
            video_type: ".flv .mp4"
        """
        self.videos_path = videos_path
        self.output_path = output_path
        self.frame_kept_per_second = frame_kept_per_second

        # collect all videos under videos_path
        self.videos = [video for video in os.listdir(self.videos_path) if video.endswith(video_type)]

        print("Number of videos: ", len(self.videos))
        # print(self.videos[0])


    def extractImage(self):
        for each_video in tqdm(self.videos):
            video_path = os.path.join(self.videos_path, each_video)
            sample_name = each_video.split(".")[0]

            try:
                vr = videoReader(video_path)
                save_dir = os.path.join(self.output_path, sample_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                vr.video2frame_update(save_dir, self.frame_kept_per_second)
            except Exception as e:
                print("Error: ", e)
                print("Error in video: ", each_video)
                exit(1)


if __name__ == "__main__":
    # parse arguments
    import sys
    videos_path = sys.argv[1]
    output_path = sys.argv[2]
    kept_per_second = int(sys.argv[3])
    video_type = sys.argv[4]

    handler = Handler(
        videos_path = videos_path,
        output_path = output_path,
        frame_kept_per_second = kept_per_second,
        video_type = video_type)
    handler.extractImage()
    print("Done!, Images extracted to: ", handler.output_path)