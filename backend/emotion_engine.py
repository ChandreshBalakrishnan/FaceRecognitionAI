"""Face detection + emotion classification.

Two pieces:
  * FaceDetector - OpenCV Haar cascade, which ships with opencv-python, so there
    is nothing extra to download. Good enough for one person facing a webcam.
  * EmotionClassifier - the FER+ ONNX model. It takes a 64x64 grayscale face and
    returns 8 emotion probabilities.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from download_model import ensure_model

# Order is fixed by the FER+ model - do not reorder.
LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class FaceDetector:
    """Finds the largest face in a frame, with a little temporal smoothing."""

    def __init__(self, min_size=80, smooth=0.5):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")
        self.min_size = min_size
        self.smooth = smooth
        self._last_box = None

    def detect(self, gray):
        """Return (x, y, w, h) of the largest face, or None."""
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(self.min_size, self.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            self._last_box = None
            return None

        # Largest face wins - that is the person actually sitting at the screen.
        box = max(faces, key=lambda f: f[2] * f[3]).astype(float)

        if self._last_box is not None:
            a = self.smooth
            box = a * box + (1 - a) * self._last_box
        self._last_box = box
        return tuple(int(v) for v in box)


class EmotionClassifier:
    """FER+ ONNX wrapper. Input: a face crop. Output: dict of label -> probability."""

    def __init__(self, model_path=None):
        model_path = model_path or ensure_model(verbose=False)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def preprocess(gray_frame, box, pad=0.15):
        """Crop the face (with a little padding), make it 64x64 grayscale."""
        h, w = gray_frame.shape[:2]
        x, y, bw, bh = box
        px, py = int(bw * pad), int(bh * pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
        crop = gray_frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        crop = cv2.equalizeHist(crop)  # helps a lot under uneven room lighting
        return crop.astype(np.float32).reshape(1, 1, 64, 64)

    def predict(self, tensor):
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probs = softmax(logits)
        return {label: float(p) for label, p in zip(LABELS, probs)}


class EmotionEngine:
    """Convenience wrapper: frame in, scores out."""

    def __init__(self, model_path=None):
        self.detector = FaceDetector()
        self.classifier = EmotionClassifier(model_path)

    def process(self, frame_bgr):
        """Returns (scores_dict_or_None, face_box_or_None)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box = self.detector.detect(gray)
        if box is None:
            return None, None
        tensor = self.classifier.preprocess(gray, box)
        if tensor is None:
            return None, box
        return self.classifier.predict(tensor), box
