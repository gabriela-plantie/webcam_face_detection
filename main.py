import cv2
from detector import *

import argparse

parser = argparse.ArgumentParser(description='Detect faces from WebCam')
parser.add_argument('-a', '--algorithm',
                    choices=['haar', 'mtcnn'],
                    nargs='?',
                    default='mtcnn',
                    help='Face detection algorithm')
parser.add_argument('-s', '--scale',
                    choices=range(1, 100),
                    type=int,
                    nargs='?',
                    default=60,
                    help='Face detection algorithm')

args = parser.parse_args()
selected_detector = {
    'haar': CascadeDetector,
    'mtcnn': MtcnnDetector,
}.get(args.algorithm)
detector = DetectorWithHistory(selected_detector())

video_capture = cv2.VideoCapture(0)
_, frame = video_capture.read()
(screen_h, screen_w, _) = frame.shape
scale_percent = args.scale  # percent of original size
screen_w = int(screen_w * scale_percent / 100)
screen_h = int(screen_h * scale_percent / 100)
dim = (screen_w, screen_h)
frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    detector.detect(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
