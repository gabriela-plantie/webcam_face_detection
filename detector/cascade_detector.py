from detector.abstract_detector import AbstractDetector
import cv2


class CascadeDetector(AbstractDetector):

    DEFAULT_MIN_NEIGHBORS = 5
    DEFAULT_SCALE_FACTOR = 1.1
    DEFAULT_MIN_SIZE = (30, 30)

    def __init__(self, scale_factor=DEFAULT_SCALE_FACTOR, min_neighbors=DEFAULT_MIN_NEIGHBORS,
                 min_size=DEFAULT_MIN_SIZE,
                 name='haar', classifier='./detector/haar_cascade/haarcascade_frontalface_default.xml'):
        super().__init__(name)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.classifier = cv2.CascadeClassifier(classifier)

    def parameters(self):
        return dict(scale_factor=self.scale_factor, min_neighbors=self.min_neighbors,
                    min_size=self.min_size)

    def preprocessImage(self, an_image):
        return cv2.cvtColor(an_image, cv2.COLOR_BGR2GRAY)

    def _detect(self, an_image):
        found = self.classifier.detectMultiScale(
            an_image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return found

    def _draw_detection(self, image, detection):
        CascadeDetector._draw_rectangle(image, detection, (0, 255, 0))
