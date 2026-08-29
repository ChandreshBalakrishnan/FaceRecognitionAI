"""Downloads the three models this project needs into ./models/.

  emotion-ferplus-8.onnx              ~35 MB   facial emotion (8 classes)
  lbfmodel.yaml                       ~56 MB   68-point face landmarks
  face_detection_yunet_2023mar.onnx   ~230 KB  face detection

Run once after installing requirements:
    python download_model.py
"""

import os
import sys
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")
LANDMARKS_PATH = os.path.join(MODEL_DIR, "lbfmodel.yaml")
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")

# The ONNX model zoo stores this file with Git LFS. The "media." host serves the
# real bytes; the plain github.com/raw URL only returns a small pointer file.
URLS = [
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://media.githubusercontent.com/media/onnx/models/main/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
]

LANDMARKS_URLS = [
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
]

# opencv_zoo keeps its weights in Git LFS, so the "media." host is the one that
# serves real bytes - the plain raw URL returns a small pointer file.
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

MIN_BYTES = 20_000_000  # a valid model is ~35 MB; anything smaller is an error page
LANDMARKS_MIN_BYTES = 20_000_000  # the real file is ~56 MB
YUNET_MIN_BYTES = 100_000  # ~230 KB; a Git LFS pointer is a few hundred bytes


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100, count * block_size * 100 // total_size)
    sys.stdout.write(f"\r  downloading... {pct}%")
    sys.stdout.flush()


def _fetch(path, urls, min_bytes, label, manual_hint, verbose=True):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        if verbose:
            print(f"{label} already present: {path}")
        return path

    last_error = None
    for url in urls:
        try:
            if verbose:
                print(f"Fetching {label} from {url}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp, _progress if verbose else None)
            if verbose:
                print()
            if os.path.getsize(tmp) < min_bytes:
                os.remove(tmp)
                raise RuntimeError("downloaded file is too small (got an error page?)")
            os.replace(tmp, path)
            if verbose:
                print(f"Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
            return path
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            if verbose:
                print(f"  failed: {exc}")

    raise RuntimeError(
        f"Could not download {label}.\n{manual_hint}\n"
        f"Place it at {path}\nLast error: {last_error}"
    )


def ensure_model(verbose=True):
    """The FER+ emotion model."""
    return _fetch(
        MODEL_PATH,
        URLS,
        MIN_BYTES,
        "emotion model",
        "Download it manually from https://github.com/onnx/models -> "
        "validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        verbose,
    )


def ensure_landmarks(verbose=True):
    """The LBF 68-point facial landmark model."""
    return _fetch(
        LANDMARKS_PATH,
        LANDMARKS_URLS,
        LANDMARKS_MIN_BYTES,
        "landmark model",
        "Download lbfmodel.yaml manually from "
        "https://github.com/kurnianggoro/GSOC2017/tree/master/data",
        verbose,
    )


def ensure_yunet(verbose=True):
    """YuNet face detector - small, fast, and far better than Haar off-angle."""
    return _fetch(
        YUNET_PATH,
        YUNET_URLS,
        YUNET_MIN_BYTES,
        "face detector",
        "Download face_detection_yunet_2023mar.onnx manually from "
        "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        verbose,
    )


if __name__ == "__main__":
    ensure_model()
    print()
    ensure_landmarks()
    print()
    ensure_yunet()
    print("\nAll three models ready. Next:  python server.py --probe")
