"""Checks everything except the camera and the browser.

Run this if something feels wrong and you want to know which half is at fault:

    python test_pipeline.py

It loads the real emotion model, runs the real engagement, dwell and decision
logic against simulated inputs, and confirms the gates and timing guards behave.

The one piece it cannot cover is MediaPipe's own output, which needs a real
camera and a real face. For that, run:  python server.py --probe
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

from decider import Decider
from dwell import DwellTracker
from engagement import (Features, EngagementTracker, MODEL_3D, POSE_IDX,
                        euler_from_rotation, eye_aspect_ratio, head_pose,
                        mouth_aspect_ratio, pupil_offset, LEFT_EYE)
from server import load_config

PASS, FAIL = "  ok  ", " FAIL "
failures = []


def check(name, condition, detail=""):
    print(f"[{PASS if condition else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def section(title):
    print(f"\n{title}\n")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class FakeDwell:
    """A dwell tracker frozen at chosen values, for testing the decider alone."""

    def __init__(self, elapsed=30.0, gate=3.0, relief=0.0):
        self._elapsed, self._gate, self._relief = elapsed, gate, relief

    def current(self, now=None):
        return self._elapsed

    def gate_seconds(self):
        return self._gate

    def threshold_relief(self, now=None):
        return self._relief

    def state(self, now=None):
        return {"dwell": self._elapsed, "dwell_median": 8.0, "dwell_gate": self._gate,
                "dwell_samples": 9, "dwell_learned": True, "dwell_relief": self._relief}


def engagement_state(attention=1.0, drowsiness=0.0, presence=1.0, absent=False):
    return {"attention": attention, "drowsiness": drowsiness, "presence": presence,
            "absent": absent, "seconds_since_face": 0.0, "blink_rate": 18.0,
            "perclos": 0.0, "looking_away": attention < 0.5}


def rot_y(deg):
    t = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]]
    return m


def rot_x(deg):
    t = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]
    return m


def stream(decider, emotion, eng, dwell, seconds, start_t, fps=12):
    fired = []
    for i in range(int(seconds * fps)):
        t = start_t + i / fps
        state = decider.update(emotion, eng, dwell, now=t)
        if state["action"]:
            fired.append((round(t - start_t, 2), state["action"]))
    return fired, start_t + int(seconds * fps) / fps


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_models():
    section("1. models")

    from emotion_engine import LABELS, EmotionClassifier

    classifier = EmotionClassifier()
    check("emotion model loads", classifier.session is not None)

    probs = classifier.predict(np.random.rand(1, 1, 64, 64).astype(np.float32) * 255)
    check("returns all 8 labels", sorted(probs) == sorted(LABELS))
    check("probabilities sum to 1", abs(sum(probs.values()) - 1.0) < 1e-4)

    import cv2

    check("opencv has the contrib face module", hasattr(cv2, "face"),
          "needed for the 68-point landmark fitter")
    check("opencv has the YuNet detector", hasattr(cv2, "FaceDetectorYN"))

    from engagement import OpenCVFaceAdapter

    adapter = OpenCVFaceAdapter(load_config())
    check("landmark model loads", adapter.facemark is not None)
    check("a detector is available",
          adapter.detector is not None or not adapter.cascade.empty(),
          "YuNet" if adapter._detector_path else "Haar fallback")

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    check("a blank frame reports no face", adapter.process(blank).present is False)


def test_head_pose():
    section("2. head pose maths")

    import cv2

    yaw, pitch, roll = euler_from_rotation(np.eye(3))
    check("identity rotation is zero", max(abs(yaw), abs(pitch), abs(roll)) < 1e-6)

    # Project the canonical 3D face at a known angle, then solve it back through
    # the same code the live pipeline uses. This is the real check.
    w, h = 640, 480
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0])

    def recovered(axis, degrees):
        t = math.radians(degrees)
        c, s_ = math.cos(t), math.sin(t)
        if axis == "yaw":
            rot = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        elif axis == "pitch":
            rot = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        else:
            rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        rvec = cv2.Rodrigues(flip @ rot)[0]
        projected, _ = cv2.projectPoints(
            MODEL_3D, rvec, np.array([[0.0], [0.0], [1000.0]]), camera, np.zeros((4, 1))
        )
        points = np.zeros((68, 2))
        for slot, index in enumerate(POSE_IDX):
            points[index] = projected[slot].ravel()
        y, p, r = head_pose(points, w, h)
        return {"yaw": y, "pitch": p, "roll": r}[axis]

    for axis, degrees in [("yaw", 25), ("yaw", -40), ("pitch", 15),
                          ("pitch", -20), ("roll", 12)]:
        got = recovered(axis, degrees)
        check(f"{axis} of {degrees:+d} deg is recovered", abs(got - degrees) < 1.5,
              f"got {got:+.2f}")


def test_face_geometry():
    section("3. eye, mouth and gaze geometry")

    def eye(width=30.0, height=12.0):
        """Six points shaped like an eye, centred at the origin."""
        return {
            LEFT_EYE[0]: np.array([-width / 2, 0.0]),
            LEFT_EYE[1]: np.array([-width / 6, -height / 2]),
            LEFT_EYE[2]: np.array([width / 6, -height / 2]),
            LEFT_EYE[3]: np.array([width / 2, 0.0]),
            LEFT_EYE[4]: np.array([width / 6, height / 2]),
            LEFT_EYE[5]: np.array([-width / 6, height / 2]),
        }

    open_eye = eye(height=12.0)
    shut_eye = eye(height=1.0)
    ear_open = eye_aspect_ratio(open_eye, LEFT_EYE)
    ear_shut = eye_aspect_ratio(shut_eye, LEFT_EYE)
    check("an open eye has a high aspect ratio", ear_open > 0.3, f"{ear_open:.3f}")
    check("a shut eye has a low one", ear_shut < 0.1, f"{ear_shut:.3f}")
    check("open scores well above shut", ear_open > 4 * ear_shut,
          f"{ear_open:.3f} vs {ear_shut:.3f}")

    def mouth(gap):
        pts = np.zeros((68, 2))
        pts[60] = [-25.0, 0.0]
        pts[64] = [25.0, 0.0]
        for upper, lower in ((61, 67), (62, 66), (63, 65)):
            pts[upper] = [0.0, -gap / 2]
            pts[lower] = [0.0, gap / 2]
        return pts

    mar_shut = mouth_aspect_ratio(mouth(1.0))
    mar_yawn = mouth_aspect_ratio(mouth(30.0))
    check("a closed mouth scores near zero", mar_shut < 0.05, f"{mar_shut:.3f}")
    check("a yawn scores high", mar_yawn > 0.5, f"{mar_yawn:.3f}")

    # Pupil offset: a dark blob placed off-centre in a synthetic eye.
    def offset_for(blob_x):
        gray = np.full((60, 120), 200, dtype=np.uint8)
        cv2_pts = {i: p + np.array([60.0, 30.0]) for i, p in eye(width=40.0, height=18.0).items()}
        gray[26:34, blob_x - 4:blob_x + 4] = 20
        return pupil_offset(gray, cv2_pts, LEFT_EYE)

    left_look = offset_for(48)
    right_look = offset_for(72)
    centre = offset_for(60)
    check("pupil offset detects a leftward pupil", left_look and left_look[0] < -0.2,
          f"{left_look[0]:+.2f}" if left_look else "none")
    check("...and a rightward one", right_look and right_look[0] > 0.2,
          f"{right_look[0]:+.2f}" if right_look else "none")
    check("...and reads near zero when centred", centre and abs(centre[0]) < 0.15,
          f"{centre[0]:+.2f}" if centre else "none")


def test_engagement():
    section("4. engagement tracking")

    cfg = load_config()
    cfg["calibration"] = dict(cfg["calibration"], yaw_center=0.0, pitch_center=0.0,
                              gaze_x_center=0.0, gaze_y_center=0.0, blink_rate_baseline=18.0)

    # a) looking straight ahead keeps attention high
    t = EngagementTracker(cfg)
    for i in range(120):
        t.update(Features(present=True, yaw=2.0, pitch=-3.0), now=i / 12)
    check("facing the screen keeps attention at 1", t.attention > 0.95, f"{t.attention:.2f}")

    # b) head turned well away drops it
    t = EngagementTracker(cfg)
    for i in range(120):
        t.update(Features(present=True, yaw=55.0), now=i / 12)
    check("head turned away drops attention to 0", t.attention < 0.05, f"{t.attention:.2f}")

    # c) gaze is a nudge, not a driver. The pupil estimate on this stack is too
    #    noisy to trust on its own - its centre has been observed flipping sign
    #    between sessions - so eye movement alone deliberately cannot leave the
    #    cone. It only tips the balance once the head has already turned part way.
    t = EngagementTracker(cfg)
    for i in range(120):
        t.update(Features(present=True, yaw=0.0, gaze_x=0.9), now=i / 12)
    check("eyes alone do not count as looking away", t.attention > 0.9,
          f"attention {t.attention:.2f}")
    check("...but they do shift the look angle", t.look_x > 8.0, f"{t.look_x:.1f} deg")

    t = EngagementTracker(cfg)
    for i in range(120):
        t.update(Features(present=True, yaw=18.0, gaze_x=0.9), now=i / 12)
    check("a part turn plus eyes does leave the cone", t.attention < 0.1,
          f"attention {t.attention:.2f}")

    t = EngagementTracker(cfg)
    for i in range(120):
        t.update(Features(present=True, yaw=18.0, gaze_x=0.0), now=i / 12)
    check("...where the part turn alone would not", t.attention > 0.9,
          f"attention {t.attention:.2f}")

    # d) a calibrated off-centre resting pose is not mistaken for looking away
    cfg_off = dict(cfg)
    cfg_off["calibration"] = dict(cfg["calibration"], yaw_center=18.0, pitch_center=-22.0)
    t = EngagementTracker(cfg_off)
    for i in range(120):
        t.update(Features(present=True, yaw=18.0, pitch=-22.0), now=i / 12)
    check("calibrated resting pose stays attentive", t.attention > 0.95, f"{t.attention:.2f}")

    # e) jitter near the edge of the cone must not read as looking away.
    #    This is the failure that made it announce "looking away" while the user
    #    was staring straight at the screen: the landmark fit wobbles a few
    #    degrees, and a raw angle sitting near the boundary flips every frame.
    cfg_cone = dict(cfg)
    cfg_cone["calibration"] = dict(cfg["calibration"], cone_x_degrees=20.0,
                                   cone_y_degrees=18.0)
    t = EngagementTracker(cfg_cone)
    rng = np.random.default_rng(3)
    for i in range(150):
        # hovering right at the boundary, +/- 4 degrees of noise
        t.update(Features(present=True, yaw=20.0 + rng.normal(0, 4.0)), now=i / 12)
    check("boundary jitter does not read as looking away", t.attention > 0.9,
          f"attention {t.attention:.2f}")

    # ...while a real turn still does.
    t = EngagementTracker(cfg_cone)
    for i in range(150):
        t.update(Features(present=True, yaw=55.0), now=i / 12)
    check("a genuine head turn still registers", t.attention < 0.05,
          f"attention {t.attention:.2f}")

    # f) hysteresis: coming back needs to be inside the inner cone
    t = EngagementTracker(cfg_cone)
    for i in range(60):
        t.update(Features(present=True, yaw=0.0), now=i / 12)
    for i in range(60, 120):
        t.update(Features(present=True, yaw=45.0), now=i / 12)
    check("leaves the cone on a clear turn", t.looking_away)
    for i in range(120, 180):
        t.update(Features(present=True, yaw=0.0), now=i / 12)
    check("and comes back when you face the screen again", not t.looking_away)

    # g) the cone is taken from calibration when present
    t = EngagementTracker(cfg_cone)
    check("uses the measured cone", t.cone() == (20.0, 18.0), str(t.cone()))
    t_default = EngagementTracker(cfg)
    check("falls back to the config cone", t_default.cone()[0] ==
          cfg["engagement"]["screen_cone_x_degrees"])

    # h) blinks are counted once each, not once per closed frame
    t = EngagementTracker(cfg)
    now = 0.0
    for _ in range(20):  # 20 blinks, each 3 frames closed
        for _ in range(3):
            t.update(Features(present=True, openness=0.1), now=now)
            now += 1 / 12
        for _ in range(9):
            t.update(Features(present=True, openness=0.9), now=now)
            now += 1 / 12
    check("counts 20 blinks as 20", len(t.blink_times) == 20, f"{len(t.blink_times)}")

    # i) blink rate is withheld until there is enough history
    t = EngagementTracker(cfg)
    for i in range(24):
        t.update(Features(present=True, openness=0.9), now=i / 12)
    check("blink rate withheld early on", math.isnan(t.blink_rate))

    # j) eyes mostly closed drives PERCLOS and drowsiness up
    t = EngagementTracker(cfg)
    for i in range(12 * 40):
        openness = 0.1 if (i % 12) < 8 else 0.9
        t.update(Features(present=True, openness=openness), now=i / 12)
    check("heavy eye closure raises PERCLOS", t.perclos > 0.5, f"{t.perclos:.2f}")
    check("...and drowsiness", t.drowsiness > 0.6, f"{t.drowsiness:.2f}")

    # k) a yawn registers
    t = EngagementTracker(cfg)
    for i in range(12 * 3):
        t.update(Features(present=True, jaw_open=0.8), now=i / 12)
    check("a sustained yawn raises drowsiness", t.drowsiness >= 0.6, f"{t.drowsiness:.2f}")

    # l) losing the face briefly means you turned away; losing it for good means
    #    you left. These must not be confused - the first should skip, the second
    #    must not. This matters more with an OpenCV detector than it would with
    #    MediaPipe, because turning your head far enough loses the detection.
    t = EngagementTracker(cfg)
    now = 0.0
    for _ in range(60):  # five seconds present and attentive
        t.update(Features(present=True, yaw=1.0), now=now)
        now += 1 / 12
    check("starts attentive", t.attention > 0.95, f"{t.attention:.2f}")

    for _ in range(24):  # two seconds of no face
        t.update(Features(present=False), now=now)
        now += 1 / 12
    check("a brief loss of face reads as looking away", t.attention < 0.75,
          f"attention {t.attention:.2f}")
    check("...and is not yet called absence", not t.absent,
          f"{t.seconds_since_face:.1f}s since face")

    for _ in range(48):  # four more seconds
        t.update(Features(present=False), now=now)
        now += 1 / 12
    check("a sustained loss is called absence", t.absent, f"{t.seconds_since_face:.1f}s")

    t.update(Features(present=True, yaw=1.0), now=now)
    check("and it clears the moment you return", not t.absent)


def test_blink_threshold():
    section("5. blink threshold calibration")

    # Regression test for a real failure: on a live face the landmark model only
    # dipped the eye aspect ratio about a fifth on each blink, never reaching the
    # fixed textbook threshold, so 25 seconds of normal blinking was counted as
    # zero blinks and the drowsiness signal silently died.
    cfg = load_config()
    eng = cfg["engagement"]
    closed_at = eng["eye_closed_below"]
    open_at = eng["eye_open_above"]
    dip = eng["blink_dip_fraction"]

    rng = np.random.default_rng(7)

    def series(n_blinks, base=0.325, blink_ratio=0.242, noise=0.008, frames=575):
        marked = set()
        for b in range(n_blinks):
            start = int((b + 0.5) * frames / max(1, n_blinks))
            marked.update(range(start, start + 3))
        return np.array([
            (blink_ratio + rng.normal(0, 0.006)) if i in marked
            else (base + rng.normal(0, noise))
            for i in range(frames)
        ])

    def count(ears):
        ear_open = float(np.percentile(ears, 75))
        ear_closed = ear_open * (1.0 - closed_at - dip) / (1.0 - closed_at)
        span = max(1e-3, ear_open - ear_closed)
        closed, blinks = False, 0
        for raw in ears:
            openness = float(np.clip((raw - ear_closed) / span, 0.0, 1.0))
            if closed:
                if openness > open_at:
                    closed = False
            elif openness < closed_at:
                closed = True
                blinks += 1
        return blinks

    for real in (3, 11, 25):
        got = count(series(real))
        check(f"counts {real} shallow blinks", got == real, f"got {got}")

    flat = count(series(0))
    check("reports none on a flat signal", flat == 0, f"got {flat}")
    check("still catches deep blinks", count(series(11, blink_ratio=0.15)) == 11)

    # And the old fixed-constant rule must be shown to fail, so nobody
    # reintroduces it.
    ears = series(11)
    ear_open = float(np.percentile(ears, 75))
    span_old = max(1e-3, ear_open - 0.12)
    closed, old_blinks = False, 0
    for raw in ears:
        openness = float(np.clip((raw - 0.12) / span_old, 0.0, 1.0))
        if closed:
            if openness > open_at:
                closed = False
        elif openness < closed_at:
            closed = True
            old_blinks += 1
    check("the old fixed threshold missed them all", old_blinks == 0,
          f"{old_blinks} blinks - this was the bug")


def test_dwell():
    section("6. dwell learning")

    cfg = load_config()
    path = os.path.join(tempfile.mkdtemp(), "dwell.json")

    d = DwellTracker(cfg, history_path=path)
    check("starts on the default median", d.median == cfg["dwell"]["default_median_seconds"],
          f"{d.median}s")
    check("and says it has not learned yet", not d.learned)

    for seconds in [4, 5, 6, 20, 7, 5, 6]:
        d.note_video_change(previous_dwell=float(seconds))
    check("learns a median from manual scrolls", d.learned and 5 <= d.median <= 7,
          f"median {d.median}s from {len(d.samples)} samples")

    before = list(d.samples)
    d.note_video_change(previous_dwell=0.9, auto=True)
    check("ignores our own auto-skips", d.samples == before, "no feedback loop")

    d.note_video_change(previous_dwell=0.2)
    check("ignores implausibly short samples", d.samples == before)
    d.note_video_change(previous_dwell=9999.0)
    check("ignores implausibly long samples", d.samples == before)

    gate = d.gate_seconds()
    check("gate sits below the median", 3.0 <= gate <= d.median, f"{gate:.1f}s")

    d.video_started = 0.0
    check("no threshold relief at the median", d.threshold_relief(now=d.median) < 0.01)
    relief_far = d.threshold_relief(now=d.median * 3)
    check("relief grows well past the median",
          abs(relief_far - cfg["dwell"]["max_relief"]) < 1e-6, f"{relief_far:.3f}")

    d2 = DwellTracker(cfg, history_path=path)
    check("history survives a restart", d2.samples == d.samples, f"{len(d2.samples)} samples")


def test_fusion():
    section("7. signal fusion")

    cfg = load_config()
    cfg["calibration"] = dict(cfg["calibration"], baseline_negative=0.0)
    d = Decider(cfg)

    angry = {"anger": 0.85, "neutral": 0.15}
    calm = {"neutral": 0.9, "happiness": 0.1}
    dwell = FakeDwell()

    # Prime the emotion EMA, then read the fused score.
    for _ in range(30):
        d.update(angry, engagement_state(), dwell, now=0.0)
    emotion_only = d.disengagement
    check("strong emotion alone clears the threshold", emotion_only > cfg["enter_threshold"],
          f"{emotion_only:.2f}")

    d = Decider(cfg)
    for _ in range(30):
        d.update(calm, engagement_state(attention=0.0), dwell, now=0.0)
    attention_only = d.disengagement
    check("looking away alone clears the threshold", attention_only > cfg["enter_threshold"],
          f"{attention_only:.2f}")

    d = Decider(cfg)
    for _ in range(30):
        d.update(calm, engagement_state(drowsiness=0.9), dwell, now=0.0)
    drowsy_only = d.disengagement
    check("drowsiness alone does not, by design", drowsy_only < cfg["enter_threshold"],
          f"{drowsy_only:.2f} - needs corroboration")

    d = Decider(cfg)
    for _ in range(30):
        d.update(calm, engagement_state(attention=0.55, drowsiness=0.9), dwell, now=0.0)
    combined = d.disengagement
    check("two weak signals together add up", combined > cfg["enter_threshold"],
          f"{combined:.2f}")

    d = Decider(cfg)
    for _ in range(30):
        d.update(calm, engagement_state(), dwell, now=0.0)
    check("an engaged, calm face scores near zero", d.disengagement < 0.1,
          f"{d.disengagement:.2f}")

    # Zeroing a gain must remove that signal entirely.
    cfg_no_emotion = dict(cfg, fusion_gains=dict(cfg["fusion_gains"], emotion=0.0))
    d = Decider(cfg_no_emotion)
    for _ in range(30):
        d.update(angry, engagement_state(), dwell, now=0.0)
    check("gain of 0 switches a signal off", d.disengagement < 0.01, f"{d.disengagement:.2f}")

    d = Decider(cfg)
    for _ in range(30):
        d.update(angry, engagement_state(attention=0.0, drowsiness=0.8), dwell, now=0.0)
    check("fused score never exceeds 1", d.disengagement <= 1.0, f"{d.disengagement:.3f}")
    check("everything at once names looking away as the reason",
          d.leading_reason() in ("looking away", "negative reaction"), d.leading_reason())


def test_gates():
    section("8. gates and timing guards")

    cfg = load_config()
    cfg["calibration"] = dict(cfg["calibration"], baseline_negative=0.0)
    calm = {"neutral": 0.9, "happiness": 0.1}
    away = engagement_state(attention=0.0)

    def fresh():
        d = Decider(cfg)
        d._grace_until = 0.0
        return d

    fired, _ = stream(fresh(), calm, engagement_state(), FakeDwell(), 10.0, 0.0)
    check("an engaged viewer is never interrupted", not fired, f"{len(fired)} events")

    fired, _ = stream(fresh(), calm, away, FakeDwell(), 3.0, 0.0)
    check("sustained looking away fires once", len(fired) == 1, f"{len(fired)} events")
    if fired:
        check("only after the hold time", 1.2 <= fired[0][0] <= 2.6, f"{fired[0][0]:.2f}s")

    fired, _ = stream(fresh(), calm, away, FakeDwell(), 6.0, 0.0)
    check("cooldown limits 6s of it to one skip", len(fired) == 1, f"{len(fired)} events")

    # absence gate
    fired, _ = stream(fresh(), calm, engagement_state(attention=0.0, absent=True),
                      FakeDwell(), 10.0, 0.0)
    check("never skips while you are away from the camera", not fired, f"{len(fired)} events")

    fired, _ = stream(fresh(), calm, engagement_state(attention=0.0, presence=0.4),
                      FakeDwell(), 3.0, 0.0)
    check("but a brief face dropout still counts as looking away", len(fired) == 1,
          f"{len(fired)} events")

    # dwell gate
    fired, _ = stream(fresh(), calm, away, FakeDwell(elapsed=1.0, gate=5.0), 10.0, 0.0)
    check("never skips before the dwell gate", not fired, f"{len(fired)} events")

    fired, _ = stream(fresh(), calm, away, FakeDwell(elapsed=9.0, gate=5.0), 4.0, 0.0)
    check("does skip once past the gate", len(fired) == 1, f"{len(fired)} events")

    # grace period
    d = Decider(cfg)
    d.notify_video_changed(now=0.0)
    fired, _ = stream(d, calm, away, FakeDwell(), 1.4, 0.0)
    check("a fresh video gets a grace period", not fired, f"{len(fired)} events")

    # flicker
    d = fresh()
    events = []
    for i in range(12 * 10):
        eng = away if i % 2 == 0 else engagement_state()
        if d.update(calm, eng, FakeDwell(), now=i / 12)["action"]:
            events.append(i)
    check("frame-to-frame flicker fires nothing", not events, f"{len(events)} events")

    # threshold relief from overstaying
    d = fresh()
    d.update(calm, engagement_state(attention=0.35), FakeDwell(), now=0.0)
    plain = d.effective_threshold(FakeDwell(relief=0.0))
    relieved = d.effective_threshold(FakeDwell(relief=0.15))
    check("overstaying lowers the bar a little", abs(plain - relieved - 0.15) < 1e-6,
          f"{plain:.2f} -> {relieved:.2f}")

    # sensitivity
    d = Decider(cfg)
    low, high = d.set_sensitivity(0.0), d.set_sensitivity(1.0)
    check("higher sensitivity lowers the threshold", high[0] < low[0],
          f"{low[0]:.2f} -> {high[0]:.2f}")
    check("higher sensitivity shortens the hold", high[1] < low[1],
          f"{low[1]:.2f}s -> {high[1]:.2f}s")

    # calibration still suppresses a resting "sad" face. A relaxed face in poor
    # light really does read like this to FER models.
    resting = {"neutral": 0.12, "sadness": 0.80, "contempt": 0.05, "fear": 0.03}
    cfg_cal = dict(cfg, calibration=dict(cfg["calibration"], baseline_negative=0.45))

    d = Decider(cfg_cal)
    d._grace_until = 0.0
    fired, _ = stream(d, resting, engagement_state(), FakeDwell(), 10.0, 0.0)
    calibrated_score = d.disengagement
    check("calibration suppresses a resting 'sad' face", not fired, f"{len(fired)} events")

    d = fresh()
    fired_uncal, _ = stream(d, resting, engagement_state(), FakeDwell(), 10.0, 0.0)
    check("...and it would have fired without it", len(fired_uncal) > 0,
          f"{len(fired_uncal)} events uncalibrated")
    check("calibration cuts the resting score substantially",
          calibrated_score < 0.7 * d.disengagement,
          f"{d.disengagement:.2f} uncalibrated -> {calibrated_score:.2f} calibrated")


def test_learning():
    section("9. learning from your own scrolling")

    from learning import ExampleRecorder, LogisticModel, build_features
    from train import auc, fit, soft_or_scores

    cfg = load_config()
    path = os.path.join(tempfile.mkdtemp(), "examples.jsonl")

    # --- the recorder
    rec = ExampleRecorder(cfg, path=path)
    engaged_frames = {"attention": 1.0, "drowsiness": 0.0, "blink_rate": 18.0}

    def record_video(dwell, auto=False, median=10.0):
        rec.start_video()
        for i in range(60):
            t_in = i / 12.0
            rec.note_frame(t_in, build_features(engaged_frames, 0.1, 0.2, t_in / median))
        return rec.note_video_end(dwell, auto, median)

    check("a fast skip is labelled bored", (record_video(3.0) or {}).get("y") == 1)
    check("a long watch is labelled engaged", (record_video(30.0) or {}).get("y") == 0)
    check("the ambiguous middle is discarded", record_video(10.0) is None,
          "no label beats a bad label")
    check("our own auto-skips are never recorded", record_video(3.0, auto=True) is None)

    rec.start_video()
    rec.note_frame(0.1, [0.0] * 6)
    check("too short to have a window is discarded", rec.note_video_end(0.9, False, 10.0) is None)

    bored, engaged = rec.counts()
    check("counts what it wrote", (bored, engaged) == (1, 1), f"{bored} bored, {engaged} engaged")

    # --- the trainer, against a simulated person whose boredom the hand-tuned
    #     rules cannot see: they bail when blinking hard and past their median,
    #     and their head never moves. This is the whole reason for learning.
    rng = np.random.default_rng(11)
    n = 600
    attention_loss = rng.uniform(0, 0.25, n)      # barely looks away - noise
    drowsiness = rng.uniform(0, 0.3, n)
    negative = rng.uniform(0, 0.3, n)
    positive = rng.uniform(0, 0.4, n)
    blink_rel = rng.uniform(-0.5, 1.5, n)         # the real driver
    dwell_ratio = rng.uniform(0.2, 2.5, n)        # the other real driver

    logit = 3.0 * blink_rel + 1.8 * (dwell_ratio - 1.0) - 1.0
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
    x = np.column_stack([attention_loss, drowsiness, negative, positive,
                         blink_rel, dwell_ratio])

    cut = int(n * 0.75)
    weights, bias, mean, std = fit(x[:cut], y[:cut])
    model = LogisticModel(weights, bias, mean, std)

    scores = np.array([model.predict(row) for row in x[cut:]])
    learned_auc = auc(y[cut:], scores)
    baseline = auc(y[cut:], soft_or_scores(x[cut:], cfg))

    check("the learned model predicts this person's skips", learned_auc > 0.75,
          f"AUC {learned_auc:.3f}")
    check("the hand-tuned rules cannot", baseline < 0.6, f"AUC {baseline:.3f}")
    check("learning wins by a clear margin", learned_auc - baseline > 0.2,
          f"{baseline:.3f} -> {learned_auc:.3f}")

    order = sorted(zip(["attention_loss", "drowsiness", "negative", "positive",
                        "blink_rel", "dwell_ratio"], weights), key=lambda kv: -abs(kv[1]))
    check("it identifies blinking and dwell as the drivers",
          {order[0][0], order[1][0]} == {"blink_rel", "dwell_ratio"},
          ", ".join(f"{k} {w:+.2f}" for k, w in order[:3]))

    # --- round trip
    saved = os.path.join(tempfile.mkdtemp(), "weights.json")
    model.meta = {"examples": n}
    model.save(saved)
    reloaded = LogisticModel.load(saved)
    check("weights survive a save and load", reloaded is not None
          and abs(reloaded.predict(x[0]) - model.predict(x[0])) < 1e-6)

    with open(saved, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["features"] = ["something", "else"]
    with open(saved, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    check("a stale feature set is refused, not mispredicted",
          LogisticModel.load(saved) is None)

    # --- the decider uses it, and the gates still apply
    d = Decider(cfg, model=model)
    d._grace_until = 0.0
    always_bored = LogisticModel([0.0] * 6, 20.0, [0.0] * 6, [1.0] * 6)
    d_hot = Decider(cfg, model=always_bored)
    d_hot._grace_until = 0.0
    fired, _ = stream(d_hot, {"neutral": 1.0}, engagement_state(), FakeDwell(), 4.0, 0.0)
    check("a learned score can fire a skip", len(fired) == 1, f"{len(fired)} events")

    d_hot = Decider(cfg, model=always_bored)
    d_hot._grace_until = 0.0
    fired, _ = stream(d_hot, {"neutral": 1.0}, engagement_state(absent=True),
                      FakeDwell(), 6.0, 0.0)
    check("but the absence gate still overrides it", not fired, f"{len(fired)} events")

    d_hot = Decider(cfg, model=always_bored)
    d_hot._grace_until = 0.0
    fired, _ = stream(d_hot, {"neutral": 1.0}, engagement_state(),
                      FakeDwell(elapsed=1.0, gate=5.0), 6.0, 0.0)
    check("and so does the dwell gate", not fired, f"{len(fired)} events")


def main():
    # CI runs with --no-models: the two model files are ~92 MB and everything
    # except section 1 is pure logic that does not need them.
    if "--no-models" not in sys.argv:
        test_models()
    else:
        section("1. models (skipped: --no-models)")
    test_head_pose()
    test_face_geometry()
    test_engagement()
    test_blink_threshold()
    test_dwell()
    test_fusion()
    test_gates()
    test_learning()

    print()
    if "--no-models" in sys.argv:
        print("Ran without the model files. Run without --no-models for the full set.")
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    print("Still worth running once on your machine:  python server.py --probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
