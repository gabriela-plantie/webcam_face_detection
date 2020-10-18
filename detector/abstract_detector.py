from abc import ABC, abstractmethod
import cv2

class AbstractDetector(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def _detect(self, an_image):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def _draw_detection(self, an_image, a_detection):
        pass

    def get_boxes(self, detections):
        return detections

    def preprocessImage(self, an_image):
        return an_image

    def detect(self, an_image):
        found = self._detect(self.preprocessImage(an_image))
        return found  # self.separate_duplicates(found)

    def detect_and_draw(self, a_source_image):
        detections = self.detect(a_source_image)
        for detection in detections:
            self._draw_detection(a_source_image, detection)
        return self.get_boxes(detections)

    def canDetectFullSizeObjects(self):
        return False

    @classmethod
    def separate_duplicates(cls, found):
        found_filtered = []
        duplicates = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and cls.inside(r, q):
                    duplicates.append(r)
                    break
            else:
                found_filtered.append(r)
        return found_filtered, found

    @classmethod
    def inside(cls, r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    @classmethod
    def point_inside(cls, p, r):
        px, py = p
        rx, ry, rw, rh = r
        return rx < px < rx + rw and ry < py < ry + rh

    @classmethod
    def _draw_point(cls, an_image, a_point, a_color):
        (w, h, _) = an_image.shape
        if AbstractDetector.point_inside(a_point, (0, 0, w, h)):
            cv2.circle(an_image, a_point, 2, a_color, 2)

    @classmethod
    def _draw_rectangle(cls, image, rectangle, color):
        (x, y, w, h) = rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    @classmethod
    def _draw_text(cls, image, origin, text, color):

        cv2.putText(image, text, origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color)

