"""Learning what boredom looks like on your face, from your own scrolling.

The hand-tuned fusion in decider.py encodes my guess: looking away matters a
lot, drowsiness a little, and so on. Those numbers were never measured on
anyone. They cannot distinguish "engaged and still" from "bored and still",
because nothing in a neutral face separates the two.

But you label boredom perfectly, hundreds of times an evening, every time you
flick past something after two seconds. This module captures those labels.

The framing is a genuine prediction problem:

    given how you looked a few seconds into a video, did you bail on it?

  * a video you scrolled past well before your median watch time  -> label 1
  * a video you stayed on well past your median                   -> label 0
  * anything in between                                           -> discarded,
    because an ambiguous label is worse than no label

Crucially the features come from a window EARLY in the video, not from the
moment you scrolled. The instant before a scroll is contaminated - you are
reaching for the trackpad and glancing down - and a model trained on that would
learn to detect scrolling, which is useless for predicting it.

Auto-skips are never recorded. They are our own doing, so learning from them
would just teach the model to agree with itself.
"""

import json
import math
import os

FEATURE_NAMES = [
    "attention_loss",
    "drowsiness",
    "negative",
    "positive",
    "blink_rel",
    "dwell_ratio",
]

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.join(HERE, "training_data.jsonl")
WEIGHTS_PATH = os.path.join(HERE, "model_weights.json")

# Window, in seconds since the video started, that the example is drawn from.
WINDOW_START = 0.7
WINDOW_END = 4.0
MIN_WINDOW_SECONDS = 0.8


def build_features(engagement, negative, positive, dwell_ratio, blink_baseline=18.0):
    """The feature vector, built identically at record time and at predict time.

    Keeping this in one function is the whole point - a mismatch between the two
    would be silent and would poison every prediction.
    """
    blink = engagement.get("blink_rate")
    if blink is None:
        blink_rel = 0.0
    else:
        blink_rel = (blink - blink_baseline) / max(1.0, blink_baseline)
        blink_rel = max(-1.0, min(2.0, blink_rel))

    return [
        1.0 - float(engagement.get("attention", 1.0)),
        float(engagement.get("drowsiness", 0.0)),
        float(negative),
        float(positive),
        blink_rel,
        min(3.0, float(dwell_ratio)),
    ]


class ExampleRecorder:
    """Buffers per-frame features for the current video and writes one example
    when the video ends, if the outcome was unambiguous."""

    def __init__(self, config, path=EXAMPLES_PATH):
        cfg = config.get("learning", {})
        self.path = path
        self.enabled = cfg.get("record", True)
        self.bored_below = cfg.get("bored_below_median_fraction", 0.6)
        self.engaged_above = cfg.get("engaged_above_median_fraction", 1.5)
        self.frames = []
        self.written = self._count_existing()

    def _count_existing(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0

    def start_video(self):
        self.frames = []

    def note_frame(self, seconds_into_video, features):
        if not self.enabled:
            return
        if seconds_into_video > WINDOW_END + 1.0:
            return  # nothing past the window is ever used, so do not buffer it
        self.frames.append((seconds_into_video, list(features)))

    def note_video_end(self, dwell, auto, median):
        """Returns the example dict written, or None."""
        frames, self.frames = self.frames, []
        if not self.enabled or auto or dwell is None or median <= 0:
            return None

        ratio = dwell / median
        if ratio < self.bored_below:
            label = 1
        elif ratio > self.engaged_above:
            label = 0
        else:
            return None  # the ambiguous middle - better no label than a bad one

        window_end = min(WINDOW_END, dwell - 0.3)
        picked = [f for t, f in frames if WINDOW_START <= t <= window_end]
        if len(picked) < 6:
            return None
        span = max(t for t, _ in frames if t <= window_end) - WINDOW_START
        if span < MIN_WINDOW_SECONDS:
            return None

        averaged = [sum(col) / len(col) for col in zip(*picked)]
        example = {
            "x": [round(v, 4) for v in averaged],
            "y": label,
            "dwell": round(dwell, 2),
            "median": round(median, 2),
            "frames": len(picked),
        }
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(example) + "\n")
            self.written += 1
        except OSError:
            return None
        return example

    def counts(self):
        """How many of each label are on disk, for progress reporting."""
        bored = engaged = 0
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        y = json.loads(line).get("y")
                    except json.JSONDecodeError:
                        continue
                    if y == 1:
                        bored += 1
                    elif y == 0:
                        engaged += 1
        except OSError:
            pass
        return bored, engaged


class LogisticModel:
    """A fitted logistic regression, standardised inputs and all."""

    def __init__(self, weights, bias, mean, std, meta=None):
        self.weights = list(weights)
        self.bias = float(bias)
        self.mean = list(mean)
        self.std = list(std)
        self.meta = meta or {}

    @classmethod
    def load(cls, path=WEIGHTS_PATH):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        if data.get("features") != FEATURE_NAMES:
            # The feature set changed since this was trained; the weights no
            # longer mean what they did, so refuse rather than mispredict.
            return None
        try:
            return cls(data["weights"], data["bias"], data["mean"], data["std"],
                       data.get("meta"))
        except KeyError:
            return None

    def save(self, path=WEIGHTS_PATH):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "features": FEATURE_NAMES,
                    "weights": [round(w, 5) for w in self.weights],
                    "bias": round(self.bias, 5),
                    "mean": [round(m, 5) for m in self.mean],
                    "std": [round(s, 5) for s in self.std],
                    "meta": self.meta,
                },
                fh,
                indent=2,
            )
            fh.write("\n")

    def predict(self, features):
        z = self.bias
        for value, weight, mean, std in zip(features, self.weights, self.mean, self.std):
            z += weight * (value - mean) / (std if std > 1e-9 else 1.0)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def contributions(self, features):
        """Signed contribution of each feature, for explaining a decision."""
        out = {}
        for name, value, weight, mean, std in zip(
            FEATURE_NAMES, features, self.weights, self.mean, self.std
        ):
            out[name] = weight * (value - mean) / (std if std > 1e-9 else 1.0)
        return out

    def leading_reason(self, features):
        contributions = self.contributions(features)
        if not contributions:
            return "learned"
        name = max(contributions.items(), key=lambda kv: kv[1])[0]
        return {
            "attention_loss": "looking away",
            "drowsiness": "bored / drowsy",
            "negative": "negative reaction",
            "positive": "not enjoying it",
            "blink_rel": "blinking a lot",
            "dwell_ratio": "watched long enough",
        }.get(name, name)
