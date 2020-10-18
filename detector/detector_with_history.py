import cv2
import time


class DetectorWithHistory:

    def __init__(self, detector):
        self.frames_count = 0
        self.i = 0
        self.detector = detector
        self.last_detections = []
        self.last_frame_time = time.time()
        self.fps_pos = None
        self.fps = ""
        pass

    def detect(self, frame):
        self.frames_count += 1
        (screen_h, screen_w, _) = frame.shape
        if self.fps_pos is None:
            self.fps_pos = (int(screen_w*0.85), 30)
        if self.i == 0 or len(self.last_detections) > 10 or not self.detector.canDetectFullSizeObjects():
            start = self.last_frame_time
            self.last_detections = self.detector.detect_and_draw(frame)
            self.last_frame_time = time.time()
            seconds = self.last_frame_time - start
            self.fps = str("%.2f" % round(self.frames_count / seconds, 2))
            self.frames_count = 0
        else:
            new_detections = []
            for (x, y, w, h) in self.last_detections:
                margin_w = int(w * 1)
                margin_h = int(h * 1)
                (x0, y0, w0, h0) = (x - margin_w, y - margin_h, w + margin_w * 2, h + margin_h * 2)
                crop_img = frame[max(0, y0):min(y0 + h0, screen_w), max(0, x0):min(x0 + w0, screen_h)]
                detections = self.detector.detect_and_draw(crop_img)
                for (x1, y1, w1, h1) in detections:
                    new_detections.append((x0 + x1, y0 + y1, w1, h1))
            self.i = (self.i + 1) % 60
        cv2.putText(frame, self.fps, self.fps_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))

