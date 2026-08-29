"""Emotion Scroll - local detection service (v2.1).

Reads your webcam, works out whether you are still engaged with what you are
watching, and tells the browser extension when to move on. Nothing leaves this
machine: the WebSocket server binds to 127.0.0.1 and no image or video is ever
stored or uploaded.

    python server.py                 # with a preview window
    python server.py --no-preview    # headless
    python server.py --camera 1      # pick a different camera
    python server.py --probe         # check the models on your machine and exit

Keys in the preview window:  q = quit,  space = arm/disarm,  r = reset
"""

import argparse
import asyncio
import json
import os
import platform
import sys
import threading
import time

import cv2

from decider import Decider
from dwell import DwellTracker
from engagement import EngagementTracker, OpenCVFaceAdapter
from learning import ExampleRecorder, LogisticModel

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")


# --------------------------------------------------------------------------
# WebSocket hub
# --------------------------------------------------------------------------


class Hub:
    """Owns the WebSocket server on its own asyncio loop in a background thread."""

    def __init__(self, host, port, on_client_message):
        self.host = host
        self.port = port
        self.on_client_message = on_client_message
        self.clients = set()
        self.loop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("WebSocket server failed to start")

    def _run(self):
        import websockets

        async def handler(ws):
            self.clients.add(ws)
            await ws.send(json.dumps({"type": "hello", "from": "emotion-scroll", "version": 2}))
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self.on_client_message(msg)
            except Exception:  # noqa: BLE001 - a dropped client is normal
                pass
            finally:
                self.clients.discard(ws)

        async def main():
            self.loop = asyncio.get_running_loop()
            async with websockets.serve(handler, self.host, self.port, ping_interval=20):
                self._ready.set()
                await asyncio.Future()  # run forever

        try:
            asyncio.run(main())
        except Exception as exc:  # noqa: BLE001
            print(f"[ws] server error: {exc}", file=sys.stderr)
            self._ready.set()

    def broadcast(self, payload):
        if not self.clients or self.loop is None:
            return
        data = json.dumps(payload)

        async def _send():
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send(data)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_send(), self.loop)


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


def open_camera(index, width):
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # last-ditch fallback
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}.\n"
            "On macOS, the first run must be allowed under System Settings > Privacy "
            "& Security > Camera for whichever app is running Python (Terminal, iTerm, "
            "VS Code...). Try --camera 1 if you have more than one camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 3 / 4))
    return cap


# --------------------------------------------------------------------------
# Preview overlay
# --------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, state, box, armed, fps, mirrored):
    h, w = frame.shape[:2]

    if box is not None:
        x, y, bw, bh = box
        if mirrored:
            x = w - x - bw
        hot = state["disengagement"] >= state["threshold"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 80, 255) if hot else (0, 200, 255), 2)

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 132), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

    status = "ARMED" if armed else "PAUSED"
    cv2.putText(frame, status, (12, 24), FONT, 0.6,
                (120, 255, 120) if armed else (150, 150, 150), 2)
    cv2.putText(frame, f"{fps:.0f} fps", (w - 78, 24), FONT, 0.5, (150, 150, 150), 1)

    comp = state.get("components", {})
    rows = [
        ("look away", comp.get("attention_loss", 0.0)),
        ("drowsy", comp.get("drowsiness", 0.0)),
        ("emotion", comp.get("emotion", 0.0)),
    ]
    for i, (label, value) in enumerate(rows):
        y = 46 + i * 18
        cv2.putText(frame, f"{label:>9s}", (12, y), FONT, 0.42, (185, 185, 185), 1)
        bar_x, bar_w = 96, 130
        cv2.rectangle(frame, (bar_x, y - 9), (bar_x + bar_w, y - 1), (55, 55, 55), -1)
        cv2.rectangle(frame, (bar_x, y - 9),
                      (bar_x + int(bar_w * min(1.0, value)), y - 1), (90, 170, 255), -1)
        cv2.putText(frame, f"{value:.2f}", (bar_x + bar_w + 8, y), FONT, 0.42, (185, 185, 185), 1)

    blink = state.get("blink_rate")
    blink_txt = "--" if blink is None else f"{blink:.0f}/min"
    cv2.putText(
        frame,
        f"attn {state.get('attention', 0):.2f}  "
        f"look {state.get('look_x', 0):+.0f},{state.get('look_y', 0):+.0f} "
        f"of {state.get('cone_x', 0):.0f},{state.get('cone_y', 0):.0f}deg  "
        f"blink {blink_txt}",
        (12, 112),
        FONT, 0.42, (150, 150, 150), 1,
    )
    dwell_txt = (
        f"dwell {state.get('dwell', 0):.1f}s / gate {state.get('dwell_gate', 0):.1f}s "
        f"(median {state.get('dwell_median', 0):.1f}s, n={state.get('dwell_samples', 0)})"
    )
    cv2.putText(frame, dwell_txt, (12, 128), FONT, 0.42, (150, 150, 150), 1)

    blocked = state.get("blocked_by")
    if blocked:
        cv2.putText(frame, f"waiting: {blocked}", (12, h - 22), FONT, 0.46, (120, 190, 255), 1)
    elif state["holding"] > 0:
        cv2.putText(
            frame,
            f"holding {state['holding']:.1f}/{state['hold_target']:.1f}s -> {state['reason']}",
            (12, h - 22), FONT, 0.46, (110, 200, 255), 1,
        )

    # Disengagement meter along the bottom, with the threshold marked.
    bar_w = int(w * min(1.0, state["disengagement"]))
    cv2.rectangle(frame, (0, h - 10), (w, h), (45, 45, 45), -1)
    cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 80, 255), -1)
    thresh_x = int(w * min(1.0, state["threshold"]))
    cv2.line(frame, (thresh_x, h - 14), (thresh_x, h), (255, 255, 255), 2)
    return frame


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------


def probe(cfg):
    """Confirm the whole vision stack works on this machine, then exit.

    Worth running once after install, and the first thing to run when something
    behaves oddly: it separates "the camera or the lighting is the problem" from
    "the logic is the problem" in about ten seconds.
    """
    print("Loading face detector and landmark model...")
    mesh = OpenCVFaceAdapter(cfg)
    print("  ok")

    print("Loading emotion model...")
    from emotion_engine import EmotionClassifier

    classifier = EmotionClassifier()
    print("  ok")

    cap = open_camera(cfg["camera_index"], cfg["frame_width"])
    print("\nLook at the camera. Sampling 40 frames...\n")

    seen = 0
    last = None
    for i in range(40):
        ok, frame = cap.read()
        if not ok:
            continue
        f = mesh.process(frame, i * 40)
        if f.present:
            seen += 1
            last = f
        time.sleep(0.03)
    cap.release()

    print(f"frames with a face: {seen}/40")
    if not last:
        print(
            "\nNo face found. Add light, put the camera near eye level, and check that "
            "nothing else is using the webcam."
        )
        return 1

    print("\nlatest frame:")
    for k, v in last.as_dict().items():
        print(f"  {k:10s} {v}")

    print(f"\nface detector: {'YuNet' if mesh.detector is not None else 'Haar cascade fallback'}")
    print(f"landmarks fitted: {'yes, 68 points' if mesh.last_points is not None else 'NO'}")

    if last.ear:
        print(f"\nyour eye aspect ratio, eyes open: {last.ear:.3f}")
        if last.ear < 0.18:
            print("  That is low. Either your eyes were closing, or the landmark fit is")
            print("  poor - add light and try again before trusting blink detection.")

    if last.bbox:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor = classifier.preprocess(gray, last.bbox)
        if tensor is not None:
            scores = classifier.predict(tensor)
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
            print("\nemotion model on that face: " + ", ".join(f"{k} {v:.2f}" for k, v in top))

    print("\nLooks good. Next:  python calibrate.py")
    return 0


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Emotion Scroll detection service")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument("--no-preview", action="store_true", help="run without a window")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--probe", action="store_true", help="check the models and exit")
    parser.add_argument("--verbose", action="store_true", help="log every decision")
    args = parser.parse_args()

    cfg = load_config()
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.port is not None:
        cfg["port"] = args.port

    if args.probe:
        sys.exit(probe(cfg))

    show_preview = cfg.get("show_preview", True) and not args.no_preview

    if not cfg.get("calibration", {}).get("calibrated_at"):
        print(
            "\n  Not calibrated yet. Run  python calibrate.py  first, or this will\n"
            "  misread your resting face, your normal sitting position, and your\n"
            "  eye shape.\n"
        )

    print("Loading models...")
    mesh = OpenCVFaceAdapter(cfg)
    classifier = None
    if cfg.get("use_emotion", True):
        from emotion_engine import EmotionClassifier

        classifier = EmotionClassifier()

    tracker = EngagementTracker(cfg)
    model = LogisticModel.load()
    decider = Decider(cfg, model=model)
    dwell = DwellTracker(cfg)
    recorder = ExampleRecorder(cfg)

    if model is not None:
        meta = model.meta or {}
        print(
            f"Using your trained model ({meta.get('examples', '?')} examples, "
            f"AUC {meta.get('test_auc', '?')})."
        )
    else:
        bored, engaged = recorder.counts()
        print(
            f"Using the hand-tuned rules. Training examples so far: "
            f"{bored} skipped early, {engaged} watched through."
        )
        if bored >= 20 and engaged >= 20:
            print("  Enough to try:  python train.py")

    shared = {"armed": True, "auto_skipped": False}
    lock = threading.Lock()

    def on_message(msg):
        kind = msg.get("type")
        if kind == "sensitivity":
            with lock:
                thr, hold = decider.set_sensitivity(msg.get("value", 0.5))
            print(f"[cfg] sensitivity -> threshold {thr:.2f}, hold {hold:.2f}s")
        elif kind == "video_changed":
            with lock:
                auto = bool(msg.get("auto", False)) or shared["auto_skipped"]
                previous = msg.get("dwell")
                if previous is None:
                    previous = dwell.current()
                # Record against the median as it stood BEFORE this video is
                # folded into it, so the label reflects what was normal at the
                # time rather than a median this video just moved.
                example = recorder.note_video_end(previous, auto, dwell.median)
                sample = dwell.note_video_change(previous_dwell=previous, auto=auto)
                recorder.start_video()
                shared["auto_skipped"] = False
                decider.notify_video_changed()
            if example is not None:
                kind_txt = "skipped early" if example["y"] == 1 else "watched through"
                print(f"[learn] {kind_txt} after {example['dwell']}s "
                      f"({recorder.written} examples)")
            if sample is not None and args.verbose:
                print(f"[dwell] watched {sample:.1f}s (median now {dwell.median:.1f}s)")
        elif kind == "enabled":
            with lock:
                shared["armed"] = bool(msg.get("value", True))
                decider.reset()
            print(f"[cfg] {'armed' if shared['armed'] else 'paused'} by extension")

    hub = Hub(cfg["host"], cfg["port"], on_message)
    hub.start()
    print(f"WebSocket listening on ws://{cfg['host']}:{cfg['port']}")

    cap = open_camera(cfg["camera_index"], cfg["frame_width"])
    print("Camera open. Load a YouTube Short and the extension will connect.\n")

    frame_interval = 1.0 / max(1, cfg.get("target_fps", 12))
    mirrored = cfg.get("mirror_preview", True)
    fps_ema = 0.0
    last_t = time.monotonic()
    started = time.monotonic()

    try:
        while True:
            loop_start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Analysis runs on the true camera image. Mirroring is display-only -
            # flipping first would invert the yaw and gaze signs.
            ts_ms = int((time.monotonic() - started) * 1000)
            features = mesh.process(frame, ts_ms)

            emotion_scores = None
            if classifier is not None and features.present and features.bbox:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tensor = classifier.preprocess(gray, features.bbox)
                if tensor is not None:
                    emotion_scores = classifier.predict(tensor)

            with lock:
                engagement = tracker.update(features)
                state = decider.update(emotion_scores, engagement, dwell)
                recorder.note_frame(dwell.current(), decider.features)
                armed = shared["armed"]

            action = state.pop("action", None)
            hub.broadcast({"type": "state", "armed": armed, **state})

            if action and armed:
                with lock:
                    shared["auto_skipped"] = True
                hub.broadcast({"type": "action", **action})
                print(
                    f"[skip] {action['reason']} score={action['score']} "
                    f"after {action['dwell']}s -> next video"
                )
            elif action and args.verbose:
                print(f"[skip suppressed - paused] {action}")

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            if show_preview:
                view = cv2.flip(frame, 1) if mirrored else frame.copy()
                view = draw_overlay(view, state, features.bbox, armed, fps_ema, mirrored)
                cv2.imshow("Emotion Scroll", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    with lock:
                        shared["armed"] = not shared["armed"]
                        decider.reset()
                    print(f"[key] {'armed' if shared['armed'] else 'paused'}")
                if key == ord("r"):
                    with lock:
                        decider.reset()

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        dwell.save()
        cap.release()
        mesh.close()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
