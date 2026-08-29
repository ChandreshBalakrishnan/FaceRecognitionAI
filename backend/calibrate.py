"""One-time calibration. Run this before your first real session.

It measures four things about *you* at rest, all of which vary enough between
people and setups that fixed defaults would be wrong:

  resting expression   emotion models are trained on posed, exaggerated faces, so
                       a relaxed face scores a surprising amount of "sadness".
                       Uncalibrated, the extension would skip almost everything.
  head pose centre     nobody sits square to their webcam, and a laptop camera
                       below eye level means a permanent downward pitch.
  gaze centre          same idea for the eyes.
  eye shape            how wide your eyes actually are when open, in eye-aspect-
                       ratio terms. Eye shape varies enough that a fixed constant
                       reads some people as permanently half-asleep.
  blink rate           normal is anywhere from 8 to 30 blinks a minute. What
                       matters is a rise above *your* normal, not a textbook one.

    python calibrate.py
    python calibrate.py --seconds 40 --dry-run
"""

import argparse
import datetime
import time

import cv2
import numpy as np

from engagement import OpenCVFaceAdapter
from server import CONFIG_PATH, load_config, open_camera, save_config


def weighted(scores, weights):
    return sum(scores.get(k, 0.0) * w for k, w in weights.items())


def main():
    parser = argparse.ArgumentParser(description="Measure your resting baselines")
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="show results, do not save")
    parser.add_argument("--no-emotion", action="store_true", help="skip the emotion baseline")
    args = parser.parse_args()

    cfg = load_config()
    if args.camera is not None:
        cfg["camera_index"] = args.camera

    print("Loading models...")
    mesh = OpenCVFaceAdapter(cfg)

    classifier = None
    if cfg.get("use_emotion", True) and not args.no_emotion:
        from emotion_engine import EmotionClassifier

        classifier = EmotionClassifier()

    cap = open_camera(cfg["camera_index"], cfg["frame_width"])

    print(
        f"\nCalibrating for {args.seconds:.0f} seconds.\n"
        "Sit the way you normally sit while scrolling, and look at the screen the\n"
        "way you normally would. Relax your face - do not smile or frown on\n"
        "purpose, and blink normally. Press q to abort.\n"
    )

    neg_samples, yaws, pitches, gx, gy = [], [], [], [], []
    ear_series = []
    frames_with_face = 0
    start = time.monotonic()

    try:
        while time.monotonic() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                continue

            elapsed = time.monotonic() - start
            features = mesh.process(frame, int(elapsed * 1000))

            if features.present:
                ear_series.append((elapsed, features.ear))
                frames_with_face += 1
                yaws.append(features.yaw)
                pitches.append(features.pitch)
                gx.append(features.gaze_x)
                gy.append(features.gaze_y)

                if classifier is not None and features.bbox:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    tensor = classifier.preprocess(gray, features.bbox)
                    if tensor is not None:
                        neg_samples.append(
                            weighted(classifier.predict(tensor), cfg["negative_weights"])
                        )

            view = cv2.flip(frame, 1) if cfg.get("mirror_preview", True) else frame
            view = view.copy()
            if features.present and features.bbox:
                x, y, w, h = features.bbox
                if cfg.get("mirror_preview", True):
                    x = view.shape[1] - x - w
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.putText(
                view,
                f"Relax and look at the screen... {args.seconds - elapsed:4.1f}s",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
            )
            cv2.putText(
                view,
                f"face {frames_with_face} frames",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1,
            )
            cv2.imshow("Calibration", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Aborted.")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mesh.close()

    duration = time.monotonic() - start

    if frames_with_face < 40 or len(ear_series) < 40:
        print(
            f"\nOnly {frames_with_face} frames had a face in them - not enough to trust.\n"
            "Try again with more light on your face and the camera roughly at eye level."
        )
        return

    calibration = dict(cfg.get("calibration", {}))
    calibration["yaw_center"] = round(float(np.median(yaws)), 2)
    calibration["pitch_center"] = round(float(np.median(pitches)), 2)
    calibration["gaze_x_center"] = round(float(np.median(gx)), 3)
    calibration["gaze_y_center"] = round(float(np.median(gy)), 3)

    # Measure the "still watching" cone instead of assuming it.
    #
    # Every frame here is a frame of you watching the screen, so the spread of
    # your look angles IS the watching distribution. Whatever it turns out to be
    # - tight if the landmark fit is crisp, wide if it is noisy - the cone is set
    # to contain almost all of it. A hard-coded cone cannot do that: too narrow
    # and pose jitter reads as looking away while you are staring right at the
    # screen; too wide and genuinely turning your head goes unnoticed.
    gaze_gain = cfg["engagement"].get("gaze_degrees_per_unit", 20.0)
    look_x_raw = (np.array(yaws) - calibration["yaw_center"]) + gaze_gain * (
        np.array(gx) - calibration["gaze_x_center"]
    )
    look_y_raw = (np.array(pitches) - calibration["pitch_center"]) + gaze_gain * (
        np.array(gy) - calibration["gaze_y_center"]
    )

    # Smooth exactly as the live tracker will, so the spread we measure is the
    # spread it will actually see.
    alpha = cfg["engagement"].get("pose_smoothing_alpha", 0.35)

    def smooth(series):
        out, prev = [], None
        for value in series:
            prev = value if prev is None else alpha * value + (1 - alpha) * prev
            out.append(prev)
        return np.array(out)

    look_x = smooth(look_x_raw)
    look_y = smooth(look_y_raw)

    margin = cfg["engagement"].get("cone_margin", 1.30)
    spread_x = float(np.percentile(np.abs(look_x), 98))
    spread_y = float(np.percentile(np.abs(look_y), 98))
    calibration["cone_x_degrees"] = round(float(np.clip(spread_x * margin, 12.0, 45.0)), 1)
    calibration["cone_y_degrees"] = round(float(np.clip(spread_y * margin, 10.0, 40.0)), 1)

    # Both ends of your eye aspect ratio.
    #
    # The open end is measured: the 75th percentile, solidly "eyes open" without
    # chasing the widest outlier.
    #
    # The closed end is DERIVED from it, not measured, and that distinction
    # matters. The obvious approach - take a low percentile of the observed
    # ratios - fails badly, because blinks are only ~2% of frames, so any
    # percentile low enough to catch them lands in open-eye noise instead and the
    # detector then reports dozens of blinks that never happened. Anchoring to a
    # fixed textbook value fails the other way: the LBF landmark model does not
    # close the eyelid fully, so a real blink may only dip the ratio by a fifth,
    # never crossing the threshold, and the drowsiness signal silently reads zero.
    #
    # So: a blink is defined as a dip of `blink_dip_fraction` below YOUR open
    # ratio. That is scale-free, so it works whether your eyes are wide or narrow
    # and whether the landmark fit is crisp or soft.
    ears = np.array([e for _, e in ear_series])
    p5, p25, p50, p75, p95 = np.percentile(ears, [5, 25, 50, 75, 95])

    ear_open = float(np.clip(p75, 0.15, 0.45))
    closed_at = cfg["engagement"].get("eye_closed_below", 0.35)
    open_at = cfg["engagement"].get("eye_open_above", 0.55)
    dip = cfg["engagement"].get("blink_dip_fraction", 0.18)

    # Chosen so that normalised openness crosses `closed_at` exactly when the raw
    # ratio has fallen by `dip` from your open value.
    ear_closed = float(ear_open * (1.0 - closed_at - dip) / (1.0 - closed_at))
    trigger_ear = ear_open * (1.0 - dip)
    separation = ear_open - ear_closed

    calibration["ear_open"] = round(ear_open, 3)
    calibration["ear_closed"] = round(ear_closed, 3)
    span = max(1e-3, ear_open - ear_closed)

    blinks, closed = 0, False
    for _, raw in ear_series:
        openness = float(np.clip((raw - ear_closed) / span, 0.0, 1.0))
        if closed:
            if openness > open_at:
                closed = False
        elif openness < closed_at:
            closed = True
            blinks += 1

    blink_rate = blinks * 60.0 / max(1e-3, duration)
    # Clamped: a 25-second sample is short, so refuse to believe extreme numbers.
    calibration["blink_rate_baseline"] = round(float(np.clip(blink_rate, 6.0, 35.0)), 1)

    print("\n--- results ---")
    print(f"frames with a face   {frames_with_face}  over {duration:.0f}s")
    print(f"head yaw centre      {calibration['yaw_center']:+.1f} deg")
    print(f"head pitch centre    {calibration['pitch_center']:+.1f} deg")
    print(f"gaze centre          x {calibration['gaze_x_center']:+.3f}  "
          f"y {calibration['gaze_y_center']:+.3f}")
    print(f"look-angle spread    x +/-{spread_x:.1f} deg   y +/-{spread_y:.1f} deg  "
          "(98th pct while watching)")
    print(f"watching cone        x +/-{calibration['cone_x_degrees']:.1f} deg   "
          f"y +/-{calibration['cone_y_degrees']:.1f} deg")
    print(f"eye ratio spread     p5 {p5:.3f}  p25 {p25:.3f}  p50 {p50:.3f}  "
          f"p75 {p75:.3f}  p95 {p95:.3f}")
    print(f"  open reference     {calibration['ear_open']:.3f}")
    print(f"  closed reference   {calibration['ear_closed']:.3f}  "
          f"(separation {separation:.3f})")
    print(f"  blink triggers at  {trigger_ear:.3f}  "
          f"({dip:.0%} below your open ratio)")
    print(f"blinks               {blinks} in {duration:.0f}s "
          f"-> {blink_rate:.1f}/min (saved as {calibration['blink_rate_baseline']:.1f})")

    if spread_x > 20 or spread_y > 18:
        print(
            "\n  Your look angle wanders a lot even while watching, so the landmark fit\n"
            "  is noisy - usually dim or uneven light. The cone has been widened to\n"
            "  match, which stops false 'looking away', but it also means you will have\n"
            "  to turn your head further before a skip triggers."
        )

    if blink_rate < 4:
        print(
            "\n  Very few blinks detected. If that does not match reality, the landmark\n"
            "  fit is probably too soft to see your eyelids - lower\n"
            "  fusion_gains.drowsiness and lean on the other signals."
        )
    elif blink_rate > 45:
        print(
            "\n  That blink rate is implausibly high, which means noise in the eye\n"
            "  landmarks is being counted as blinks. Lower fusion_gains.drowsiness."
        )

    if neg_samples:
        arr = np.array(neg_samples)
        # 75th percentile, not the mean: sit above your normal fluctuation rather
        # than in the middle of it.
        baseline = float(np.clip(np.percentile(arr, 75), 0.0, 0.85))
        calibration["baseline_negative"] = round(baseline, 3)
        print(f"resting 'negative'   mean {arr.mean():.3f}  median {np.median(arr):.3f}  "
              f"p75 {baseline:.3f}  <- saved")
        if baseline > 0.6:
            print(
                "\n  That is a high resting score. It usually means poor lighting or a\n"
                "  steep camera angle rather than anything about your face."
            )
    else:
        print("resting 'negative'   skipped")

    if abs(calibration["pitch_center"]) > 20:
        print(
            "\n  Your head pitch is well off centre - most likely a laptop camera below\n"
            "  eye level. It is calibrated out, but raising the laptop will make gaze\n"
            "  tracking noticeably more accurate."
        )

    if args.dry_run:
        print("\n--dry-run: config.json not modified.")
        return

    calibration["calibrated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    cfg_on_disk = load_config()
    cfg_on_disk["calibration"] = calibration
    save_config(cfg_on_disk, CONFIG_PATH)
    print("\nSaved to config.json. Next:  python server.py")


if __name__ == "__main__":
    main()
