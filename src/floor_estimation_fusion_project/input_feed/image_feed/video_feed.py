"""
"""
import acapture
import cv2
import os

from input_feed.image_feed.ImageFeed import ImageFeed
from main.flags_global import FLAGS


class VideoFeedAsync(ImageFeed):
    def __init__(self, frame_capture=False):
        self.cap = None
        self.main_dir = FLAGS.main_dir_path
        self.video_file_path = os.path.join(self.main_dir, FLAGS.image_input_video_path)
        self.check = False
        self.frame = []
        self.frame_capture = frame_capture

    def open_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_file_path)

    def close_video(self):
        self.cap.destroy()
        self.cap = None

    def get_frame(self):
        self.get_next_frame()
        return self.frame

    def get_next_frame(self):
        self.open_video()
        self.check, frame_raw = self.cap.read()
        if self.check:
            self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        else:
            self.frame = None

    def start_video_stream(self):
        while self.frame is not None:
            self.get_next_frame()

    def show_frame(self, repeat=False):
        if repeat is True:
            wait_delay = 1
        else:
            wait_delay = 200
        while True:
            cv2.imshow("VideoFrame", self.frame)
            k = cv2.waitKey(wait_delay)
            if not repeat or k == 27:
                break
        cv2.destroyWindow("VideoFrame")

    def center_crop(self):
        # :TODO need to be done preprocessing of input picture
        width = 1280
        height = 720    # Get dimensions
        new_width = 500
        new_height = 500

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        im = self.frame.crop((left, top, right, bottom))
        frame[y:y+h, x:x+w]
