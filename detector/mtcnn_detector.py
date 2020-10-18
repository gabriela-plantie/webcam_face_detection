from detector.abstract_detector import AbstractDetector

class MtcnnDetector(AbstractDetector):

    def __init__(self):
        import mtcnn
        super().__init__('mtcnn')
        self.classifier = mtcnn.MTCNN()

    def parameters(self):
        return dict()

    def _detect(self, an_image):
        return self.classifier.detect_faces(an_image)

    def _draw_detection(self, image, detection):
        MtcnnDetector._draw_rectangle(image, detection['box'], (0, 255, 0))
        self._draw_text(image,
                        (detection['box'][0], detection['box'][1]-10),
                        str("%.10f" % detection['confidence']),
                        (0, 255, 0))
        self._draw_keypoints(detection['keypoints'], image)

    def _draw_keypoints(self, keypoints, image):
        MtcnnDetector._draw_point(image, keypoints['left_eye'], (0, 255, 0))
        MtcnnDetector._draw_point(image, keypoints['right_eye'], (0, 255, 0))
        MtcnnDetector._draw_point(image, keypoints['nose'], (255, 255, 0))
        MtcnnDetector._draw_point(image, keypoints['mouth_left'], (0, 255, 255))
        MtcnnDetector._draw_point(image, keypoints['mouth_right'], (0, 255, 255))

    def get_boxes(self, detections):
        boxes = []
        for detection in detections:
            boxes.append(detection['box'])
        return boxes

    def canDetectFullSizeObjects(self):
        return True
