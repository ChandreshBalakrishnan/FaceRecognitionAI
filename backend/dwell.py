"""How long you normally stay on a video, learned from your own behaviour.

Dwell time is not a reason to skip - it is the thing that stops the system
being obnoxious. Two jobs:

  1. A floor. No auto-skip until you have been on a video for a decent fraction
     of your own median watch time. Whatever your face is doing in the first
     second, you have not really seen the video yet.
  2. A nudge. Once you are well past your normal watch time and still look
     disengaged, the threshold drops a little. Sitting on a video far longer
     than you usually would, while looking away, is a strong signal.

The median comes from *manual* scrolls only. Counting our own auto-skips would
be a feedback loop: skip early, median drops, skip earlier.
"""

import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(HERE, "dwell_history.json")


class DwellTracker:
    def __init__(self, config, history_path=HISTORY_PATH):
        cfg = config.get("dwell", {})
        self.cfg = cfg
        self.history_path = history_path
        self.max_samples = cfg.get("max_samples", 60)
        self.min_sample_seconds = cfg.get("min_sample_seconds", 0.7)
        self.max_sample_seconds = cfg.get("max_sample_seconds", 180.0)
        self.default_median = cfg.get("default_median_seconds", 8.0)
        self.min_samples_to_trust = cfg.get("min_samples_to_trust", 5)

        self.samples = self._load()
        self.video_started = time.monotonic()

    # ------------------------------------------------------------ persistence

    def _load(self):
        try:
            with open(self.history_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            samples = [float(x) for x in data.get("samples", [])]
            return samples[-self.max_samples :]
        except (OSError, ValueError, TypeError):
            return []

    def save(self):
        try:
            with open(self.history_path, "w", encoding="utf-8") as fh:
                json.dump({"samples": self.samples[-self.max_samples :]}, fh)
        except OSError:
            pass  # a read-only folder should not take the whole app down

    # ----------------------------------------------------------------- events

    def note_video_change(self, previous_dwell=None, auto=False, now=None):
        """Called when the browser moves to a new video."""
        now = now if now is not None else time.monotonic()
        elapsed = previous_dwell if previous_dwell is not None else now - self.video_started
        self.video_started = now

        if auto:
            return None  # our own doing - not evidence about your taste
        if elapsed is None:
            return None
        if not (self.min_sample_seconds <= elapsed <= self.max_sample_seconds):
            return None

        self.samples.append(float(elapsed))
        self.samples = self.samples[-self.max_samples :]
        self.save()
        return float(elapsed)

    # ---------------------------------------------------------------- queries

    @property
    def median(self):
        if len(self.samples) < self.min_samples_to_trust:
            return self.default_median
        return statistics.median(self.samples)

    @property
    def learned(self):
        return len(self.samples) >= self.min_samples_to_trust

    def current(self, now=None):
        now = now if now is not None else time.monotonic()
        return max(0.0, now - self.video_started)

    def gate_seconds(self):
        """Nothing may fire before this many seconds into a video."""
        fraction = self.cfg.get("gate_fraction_of_median", 0.5)
        floor = self.cfg.get("gate_floor_seconds", 3.0)
        ceiling = self.cfg.get("gate_ceiling_seconds", 20.0)
        return min(ceiling, max(floor, fraction * self.median))

    def threshold_relief(self, now=None):
        """0 up to your median, rising to `max_relief` as you go well past it.

        Subtracted from the trigger threshold, so overstaying makes the system
        slightly more willing to move you along.
        """
        med = max(1e-3, self.median)
        over = (self.current(now) / med) - 1.0
        span = max(1e-3, self.cfg.get("relief_span_medians", 1.5))
        return _clamp(over / span) * self.cfg.get("max_relief", 0.15)

    def state(self, now=None):
        return {
            "dwell": round(self.current(now), 1),
            "dwell_median": round(self.median, 1),
            "dwell_gate": round(self.gate_seconds(), 1),
            "dwell_samples": len(self.samples),
            "dwell_learned": self.learned,
            "dwell_relief": round(self.threshold_relief(now), 3),
        }


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))
