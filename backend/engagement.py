"""Engagement signals from OpenCV face landmarks.

Emotion tells you whether you dislike something. Engagement tells you whether
you are still *there* - and boredom, the thing you actually want to detect, has
no facial expression at all. It shows up as looking away, blinking more, and
yawning.

Three layers:

  OpenCVFaceAdapter   one frame -> raw Features (pose angles, gaze, eye openness)
  EngagementTracker   a stream of Features -> slow scores over time windows
  (the decider then fuses those scores with the emotion score)

The adapter is deliberately thin and all the interesting logic lives in the
tracker, because the tracker is what can be tested without a camera.

Why OpenCV and not MediaPipe: MediaPipe's face detector reaches for Metal on
Apple Silicon and aborts when the GPU service is not registered. This stack is
plain CPU OpenCV - YuNet for detection, the LBF 68-point model for landmarks -
and every piece of it is verified in test_pipeline.py against a real image.
"""

import math
import os
import time
from collections import deque

import cv2
import numpy as np

# 68-point iBUG landmark indices.
LEFT_EYE = (36, 37, 38, 39, 40, 41)
RIGHT_EYE = (42, 43, 44, 45, 46, 47)
INNER_MOUTH_V = ((61, 67), (62, 66), (63, 65))
INNER_MOUTH_H = (60, 64)

# Points used for head pose, and the canonical 3D face they correspond to (mm).
POSE_IDX = [30, 8, 36, 45, 48, 54]  # nose tip, chin, eye corners, mouth corners
MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye, outer corner
        (225.0, 170.0, -135.0),  # right eye, outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)

# The 3D model has Y up and Z toward the viewer; image coordinates have Y down.
FLIP = np.diag([1.0, -1.0, -1.0])


class Features:
    """What one frame tells us."""

    __slots__ = ("present", "yaw", "pitch", "roll", "gaze_x", "gaze_y", "openness",
                 "jaw_open", "bbox", "ear")

    def __init__(self, present=False, yaw=0.0, pitch=0.0, roll=0.0,
                 gaze_x=0.0, gaze_y=0.0, openness=1.0, jaw_open=0.0, bbox=None, ear=0.0):
        self.present = present
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.gaze_x = gaze_x
        self.gaze_y = gaze_y
        self.openness = openness
        self.jaw_open = jaw_open
        self.bbox = bbox  # (x, y, w, h) in pixels, reused for the emotion crop
        self.ear = ear  # raw eye aspect ratio, before normalisation

    def as_dict(self):
        return {
            "present": self.present,
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "gaze_x": round(self.gaze_x, 3),
            "gaze_y": round(self.gaze_y, 3),
            "openness": round(self.openness, 3),
            "jaw_open": round(self.jaw_open, 3),
            "ear": round(self.ear, 3),
        }


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def euler_from_rotation(r):
    """Yaw / pitch / roll in degrees from a 3x3 rotation matrix.

    Verified in test_pipeline.py by projecting a known pose and solving it back.
    """
    r = np.asarray(r, dtype=float)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = 0.0
    else:
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, _wrap(pitch), _wrap(roll)


def _wrap(angle):
    """A head is never pitched 176 degrees - that is the model's axis flip."""
    if angle > 90:
        return angle - 180
    if angle < -90:
        return angle + 180
    return angle


def eye_aspect_ratio(points, idx):
    """The classic blink metric: eye height over eye width."""
    p = [points[i] for i in idx]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
    return float(vertical / horizontal) if horizontal > 1e-6 else 0.0


def mouth_aspect_ratio(points):
    """Inner-lip height over width. Near 0 closed, well above 0.5 mid-yawn."""
    horizontal = np.linalg.norm(points[INNER_MOUTH_H[0]] - points[INNER_MOUTH_H[1]])
    if horizontal < 1e-6:
        return 0.0
    vertical = sum(np.linalg.norm(points[a] - points[b]) for a, b in INNER_MOUTH_V)
    return float(vertical / (3.0 * horizontal))


def head_pose(points, frame_w, frame_h):
    """Yaw / pitch / roll from six landmarks and a canonical 3D face."""
    focal = float(frame_w)
    camera = np.array(
        [[focal, 0, frame_w / 2.0], [0, focal, frame_h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    image_pts = np.array([points[i] for i in POSE_IDX], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D, image_pts, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rotation, _ = cv2.Rodrigues(rvec)
    return euler_from_rotation(FLIP @ rotation)


def pupil_offset(gray, points, idx, dark_fraction=0.35):
    """Roughly where the pupil sits inside the eye, as a fraction of eye half-width.

    Crude - it is the centroid of the darkest pixels in the eye region - but the
    iris is reliably the darkest thing there, and calibration recentres whatever
    bias this has. Returns None when the eye region is unusable.

    The threshold is anchored to the darkest pixel rather than a fixed
    percentile: a percentile assumes the iris takes up a predictable share of the
    crop, which fails whenever the eye is squinted or the crop runs wide.
    """
    eye = np.array([points[i] for i in idx], dtype=np.float64)
    x0, y0 = eye.min(axis=0)
    x1, y1 = eye.max(axis=0)
    pad = max(1.0, (x1 - x0) * 0.12)
    xa, ya = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    xb, yb = int(min(gray.shape[1], x1 + pad)), int(min(gray.shape[0], y1 + pad))
    roi = gray[ya:yb, xa:xb]
    if roi.size < 40:
        return None

    darkest = float(roi.min())
    bright = float(np.percentile(roi, 70))
    threshold = darkest + dark_fraction * max(1.0, bright - darkest)
    ys, xs = np.nonzero(roi <= threshold)
    if len(xs) < 5:
        return None

    cx, cy = xs.mean() + xa, ys.mean() + ya
    half_w = max(1e-3, (x1 - x0) / 2.0)
    half_h = max(1e-3, (y1 - y0) / 2.0)
    return float((cx - eye[:, 0].mean()) / half_w), float((cy - eye[:, 1].mean()) / half_h)


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------


class OpenCVFaceAdapter:
    """Detect a face, fit 68 landmarks, derive pose / gaze / eye / mouth features.

    All CPU, no GPU path, so nothing here can hit the Metal problem that made
    MediaPipe abort on Apple Silicon.
    """

    def __init__(self, config=None, detector_path=None, landmark_path=None):
        from download_model import ensure_landmarks, ensure_yunet

        config = config or {}
        cfg = config.get("engagement", {})
        # Your own wide-open eye aspect ratio, measured by calibrate.py. Eye shape
        # varies enough between people that a fixed constant misreads some faces
        # as permanently half-asleep.
        cal = config.get("calibration", {})
        self.ear_open = cal.get("ear_open", 0.30)
        # Measured by calibrate.py. The config value is only a starting guess -
        # the LBF landmark model does not close the eyelid fully, so a real blink
        # dips far less than a textbook eye aspect ratio would.
        self.ear_closed = cal.get("ear_closed", cfg.get("ear_closed_reference", 0.12))
        self.mar_yawn = cfg.get("mar_full_open", 0.50)

        self.detector = None
        self._detector_size = None
        try:
            self._detector_path = detector_path or ensure_yunet(verbose=False)
        except Exception:  # noqa: BLE001 - fall back to the bundled cascade
            self._detector_path = None

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)

        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(landmark_path or ensure_landmarks(verbose=False))

        self.last_points = None

    # ------------------------------------------------------------ detection

    def _ensure_detector(self, w, h):
        if self._detector_path is None:
            return None
        if self.detector is None or self._detector_size != (w, h):
            self.detector = cv2.FaceDetectorYN.create(
                self._detector_path, "", (w, h), 0.6, 0.3, 5000
            )
            self._detector_size = (w, h)
        return self.detector

    def _detect(self, frame_bgr, gray):
        h, w = gray.shape[:2]
        detector = self._ensure_detector(w, h)
        if detector is not None:
            _, faces = detector.detect(frame_bgr)
            if faces is not None and len(faces):
                # Largest face - that is the person at the screen.
                best = max(faces, key=lambda f: f[2] * f[3])
                x, y, bw, bh = (int(v) for v in best[:4])
                x, y = max(0, x), max(0, y)
                if bw > 20 and bh > 20:
                    return (x, y, min(bw, w - x), min(bh, h - y))

        boxes = self.cascade.detectMultiScale(gray, 1.15, 6, minSize=(80, 80))
        if len(boxes):
            return tuple(int(v) for v in max(boxes, key=lambda b: b[2] * b[3]))
        return None

    # -------------------------------------------------------------- process

    def process(self, frame_bgr, timestamp_ms=0):
        """Frame in (BGR, unmirrored), Features out."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box = self._detect(frame_bgr, gray)
        if box is None:
            self.last_points = None
            return Features(present=False)

        ok, fitted = self.facemark.fit(gray, np.array([box]))
        if not ok or not len(fitted):
            self.last_points = None
            # Detected but could not fit landmarks - still report presence so a
            # momentary fit failure is not mistaken for you leaving the room.
            return Features(present=True, bbox=box, openness=1.0)

        points = np.asarray(fitted[0][0], dtype=np.float64)
        self.last_points = points
        h, w = gray.shape[:2]

        yaw, pitch, roll = head_pose(points, w, h)

        ear = 0.5 * (
            eye_aspect_ratio(points, LEFT_EYE) + eye_aspect_ratio(points, RIGHT_EYE)
        )
        span = max(1e-3, self.ear_open - self.ear_closed)
        openness = float(np.clip((ear - self.ear_closed) / span, 0.0, 1.0))

        jaw_open = float(np.clip(mouth_aspect_ratio(points) / max(1e-3, self.mar_yawn), 0.0, 1.0))

        offsets = [
            pupil_offset(gray, points, LEFT_EYE),
            pupil_offset(gray, points, RIGHT_EYE),
        ]
        offsets = [o for o in offsets if o is not None]
        if offsets:
            gaze_x = sum(o[0] for o in offsets) / len(offsets)
            gaze_y = sum(o[1] for o in offsets) / len(offsets)
        else:
            gaze_x = gaze_y = 0.0

        return Features(
            present=True,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            openness=openness,
            jaw_open=jaw_open,
            bbox=box,
            ear=ear,
        )

    def close(self):
        pass


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


class _Window:
    """Time-bounded sliding window of (timestamp, value)."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.items = deque()

    def push(self, t, value):
        self.items.append((t, value))
        cutoff = t - self.seconds
        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()

    def values(self):
        return [v for _, v in self.items]

    def mean(self, default=0.0):
        vals = self.values()
        return sum(vals) / len(vals) if vals else default

    def span(self):
        if len(self.items) < 2:
            return 0.0
        return self.items[-1][0] - self.items[0][0]


class EngagementTracker:
    """Turns per-frame Features into slow, stable engagement scores."""

    def __init__(self, config):
        self.cfg = dict(config)
        eng = self.cfg.get("engagement", {})

        self.attention_window = _Window(eng.get("attention_window_seconds", 6.0))
        self.presence_window = _Window(eng.get("presence_window_seconds", 8.0))
        self.closure_window = _Window(eng.get("perclos_window_seconds", 30.0))
        self.blink_times = deque()
        self.blink_window_seconds = eng.get("blink_window_seconds", 60.0)

        self._eye_closed = False
        self._last_blink_t = -10.0
        self._last_yawn_t = -100.0
        self._jaw_open_since = None
        self._last_face_t = None

        self._sm_x = None
        self._sm_y = None
        self._on_screen = True
        self.look_x = 0.0
        self.look_y = 0.0

        self.attention = 1.0
        self.presence = 1.0
        self.seconds_since_face = 0.0
        self.absent = False
        self.blink_rate = float("nan")
        self.perclos = 0.0
        self.drowsiness = 0.0
        self.looking_away = False

    def calibration(self):
        return self.cfg.get("calibration", {})

    def update_config(self, patch):
        self.cfg.update(patch)

    # ---------------------------------------------------------------- update

    def update(self, features, now=None):
        now = now if now is not None else time.monotonic()
        eng = self.cfg.get("engagement", {})

        self.presence_window.push(now, 1.0 if features.present else 0.0)
        self.presence = self.presence_window.mean(default=1.0)

        if features.present:
            self._last_face_t = now
            self.seconds_since_face = 0.0
        elif self._last_face_t is None:
            self.seconds_since_face = 0.0
        else:
            self.seconds_since_face = now - self._last_face_t

        absent_after = eng.get("absent_after_seconds", 4.0)
        self.absent = self.seconds_since_face > absent_after

        if not features.present:
            # A face lost for a moment usually means you turned your head far
            # enough that the detector cannot see it any more - which is exactly
            # "not looking at the screen". Only a sustained loss means you left,
            # and that is handled by `absent`, which suppresses rather than fires.
            if self._last_face_t is not None and not self.absent:
                self.attention_window.push(now, 0.0)
                self.attention = self.attention_window.mean(default=1.0)
                self.looking_away = True
            return self.state()

        self._update_attention(features, now, eng)
        self._update_blinks(features, now, eng)
        self._update_yawns(features, now, eng)
        self.drowsiness = self._drowsiness(now)
        return self.state()

    def look_angles(self, features):
        """Where you appear to be looking, in degrees from your calibrated centre.

        Head pose does most of the work; pupil position is a smaller correction on
        top, because it is much the noisier of the two.
        """
        cal = self.calibration()
        gaze_gain = self.cfg.get("engagement", {}).get("gaze_degrees_per_unit", 20.0)
        look_x = (features.yaw - cal.get("yaw_center", 0.0)) + gaze_gain * (
            features.gaze_x - cal.get("gaze_x_center", 0.0)
        )
        look_y = (features.pitch - cal.get("pitch_center", 0.0)) + gaze_gain * (
            features.gaze_y - cal.get("gaze_y_center", 0.0)
        )
        return look_x, look_y

    def cone(self):
        """Half-angles of the 'still watching' cone, measured if calibration has them."""
        eng = self.cfg.get("engagement", {})
        cal = self.calibration()
        return (
            cal.get("cone_x_degrees") or eng.get("screen_cone_x_degrees", 26.0),
            cal.get("cone_y_degrees") or eng.get("screen_cone_y_degrees", 22.0),
        )

    def _update_attention(self, features, now, eng):
        raw_x, raw_y = self.look_angles(features)

        # Smooth the ANGLES, not just the on/off decision that follows. The
        # landmark fit jitters by several degrees frame to frame, and a raw angle
        # sitting near the edge of the cone flips in and out constantly - which
        # reads as "looking away" while you are staring straight at the screen.
        alpha = eng.get("pose_smoothing_alpha", 0.35)
        if self._sm_x is None:
            self._sm_x, self._sm_y = raw_x, raw_y
        else:
            self._sm_x = alpha * raw_x + (1 - alpha) * self._sm_x
            self._sm_y = alpha * raw_y + (1 - alpha) * self._sm_y
        self.look_x, self.look_y = self._sm_x, self._sm_y

        cone_x, cone_y = self.cone()

        # Hysteresis, for the same reason a thermostat has it. Leaving the cone
        # takes a bigger deviation than coming back into it, so an angle hovering
        # on the boundary settles instead of chattering.
        exit_mult = eng.get("cone_exit_multiplier", 1.35)
        if self._on_screen:
            self._on_screen = (
                abs(self.look_x) < cone_x * exit_mult and abs(self.look_y) < cone_y * exit_mult
            )
        else:
            self._on_screen = abs(self.look_x) < cone_x and abs(self.look_y) < cone_y

        self.attention_window.push(now, 1.0 if self._on_screen else 0.0)
        self.attention = self.attention_window.mean(default=1.0)
        self.looking_away = not self._on_screen

    def _update_blinks(self, features, now, eng):
        closed_at = eng.get("eye_closed_below", 0.35)
        open_at = eng.get("eye_open_above", 0.55)

        if self._eye_closed:
            if features.openness > open_at:
                self._eye_closed = False
        elif features.openness < closed_at:
            self._eye_closed = True
            if now - self._last_blink_t > eng.get("min_blink_gap_seconds", 0.12):
                self._last_blink_t = now
                self.blink_times.append(now)

        while self.blink_times and self.blink_times[0] < now - self.blink_window_seconds:
            self.blink_times.popleft()

        self.closure_window.push(
            now, 1.0 if features.openness < eng.get("perclos_below", 0.25) else 0.0
        )
        self.perclos = self.closure_window.mean(default=0.0)

        # Only trust the blink rate once there is a decent stretch of history,
        # otherwise the first few seconds report wild numbers.
        history = min(self.blink_window_seconds,
                      max(self.presence_window.span(), self.closure_window.span()))
        if history >= eng.get("min_blink_history_seconds", 20.0):
            self.blink_rate = len(self.blink_times) * 60.0 / history
        else:
            self.blink_rate = float("nan")

    def _update_yawns(self, features, now, eng):
        if features.jaw_open > eng.get("yawn_jaw_above", 0.5):
            if self._jaw_open_since is None:
                self._jaw_open_since = now
            elif now - self._jaw_open_since > eng.get("yawn_seconds", 1.4):
                self._last_yawn_t = now
                self._jaw_open_since = None
        else:
            self._jaw_open_since = None

    def _drowsiness(self, now):
        eng = self.cfg.get("engagement", {})
        cal = self.calibration()

        base_rate = cal.get("blink_rate_baseline", 18.0)
        excess = eng.get("blink_rate_excess", 18.0)
        if math.isnan(self.blink_rate):
            rate_component = 0.0
        else:
            rate_component = _clamp((self.blink_rate - base_rate) / max(1.0, excess))

        perclos_component = _clamp(self.perclos / max(1e-3, eng.get("perclos_high", 0.20)))
        yawn_component = 1.0 if now - self._last_yawn_t < eng.get("yawn_hold_seconds", 6.0) else 0.0

        return _clamp(
            eng.get("w_blink_rate", 0.5) * rate_component
            + eng.get("w_perclos", 0.7) * perclos_component
            + eng.get("w_yawn", 0.6) * yawn_component
        )

    def state(self):
        return {
            "attention": round(self.attention, 3),
            "presence": round(self.presence, 3),
            "absent": self.absent,
            "seconds_since_face": round(self.seconds_since_face, 1),
            "blink_rate": None if math.isnan(self.blink_rate) else round(self.blink_rate, 1),
            "perclos": round(self.perclos, 3),
            "drowsiness": round(self.drowsiness, 3),
            "looking_away": self.looking_away,
            "look_x": round(self.look_x, 1),
            "look_y": round(self.look_y, 1),
            "cone_x": round(self.cone()[0], 1),
            "cone_y": round(self.cone()[1], 1),
        }


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))
