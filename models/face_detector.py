import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Production-ready face detector.
    Primary: MTCNN | Fallback: OpenCV Haar Cascade
    """

    def __init__(self):
        try:
            self.mtcnn = MTCNN()
            print("✅ MTCNN loaded successfully")
        except Exception as e:
            print(f"❌ MTCNN failed: {e}")
            self.mtcnn = None

        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.haar_cascade.empty():
            print("❌ Haar Cascade failed to load")
        else:
            print("✅ Haar Cascade fallback loaded")

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        if frame is None or frame.size == 0:
            return []

        try:
            if self.mtcnn is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.mtcnn.detect_faces(rgb_frame)
                if detections:
                    return detections

            # Fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) > 0:
                return [{'box': [x, y, w, h], 'confidence': 0.75, 'keypoints': {}} for (x, y, w, h) in faces]

            return []
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def draw_faces(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            x, y, w, h = det['box']
            color = (0, 255, 0) if det.get('keypoints') else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Face {det.get('confidence', 0):.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame


# ==================== TEST ====================
if __name__ == "__main__":
    print("🚀 Starting Face Detector Test...")
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam. Check Windows Camera privacy settings.")
        exit()

    print("🎥 Live window should appear now!")
    print("   Green = MTCNN | Orange = Haar")
    print("   Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_faces(frame)
        frame = detector.draw_faces(frame, detections)

        cv2.imshow("Face Detection Test (MTCNN + Haar)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test finished.")