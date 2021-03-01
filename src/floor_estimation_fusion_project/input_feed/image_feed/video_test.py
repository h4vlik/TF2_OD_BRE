import acapture
import cv2
import os

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
video_file_path = os.path.join(main_dir, os.path.join(r'data\\Camera_input_data\\video', os.path.basename("A3_01.mp4")))
cap = cv2.VideoCapture(video_file_path)

check, frame = cap.read()

wait_delay = 1000

cv2.imshow("VideoFrame", frame)
k = cv2.waitKey(wait_delay)