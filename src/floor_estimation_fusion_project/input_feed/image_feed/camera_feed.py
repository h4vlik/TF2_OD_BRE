"""
"""
import acapture
import cv2
import os

from input_feed.image_feed.ImageFeed import ImageFeed
from main.flags_global import FLAGS


class CameraFeedAsync(ImageFeed):
    def __init__(self):
        self.cap = None
        self.check = False
        self.frame = []
        self.src = FLAGS.camera_device_used

    def open_camera_device(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.src)  # /dev/video0 on linux

    def close_camera_device(self):
        self.cap.destroy()
        self.cap = None

    def get_next_frame(self):
        self.open_camera_device()
        self.check, frame_raw = self.cap.read()
        if self.check:
            self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        else:
            self.frame = None

    def start_camera_stream(self):
        while self.frame is not None:
            self.get_next_frame()  # non-blocking

    def get_frame(self):
        self.get_next_frame()
        return self.frame

    def show_frame(self, repeat=False):
        if repeat is True:
            wait_delay = 1
        else:
            wait_delay = 0.1
        while True:
            cv2.imshow("CameraFrame", self.frame)
            k = cv2.waitKey(wait_delay)
            if not repeat or k == 27:
                break
        cv2.destroyWindow("CameraFrame")
