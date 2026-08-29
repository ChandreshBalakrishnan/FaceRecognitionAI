"""Turns noisy per-frame signals into occasional decisions.

Three signals are fused into one disengagement score:

  attention loss   you are not looking at the screen (head pose + eye gaze)
  drowsiness       blink rate above your own baseline, eye closure, yawns
  negative emotion anger / disgust / sadness above your own baseline

They combine with a soft OR rather than an average:

    disengaged = 1 - (1-g1*s1)(1-g2*s2)(1-g3*s3)

An average would drown out a single strong signal - clear disgust with an
otherwise attentive face would come out middling and never fire. A soft OR says
"any one of these, on its own, is enough, and together they are more". Setting a
signal's gain to 0 in config.json removes it cleanly.

Two things gate the result rather than feeding it:

  presence   if you are not in front of the camera, do nothing at all. Skipping
             while you are out of the room just burns through the feed.
  dwell      nothing fires until you have been on the video long enough to have
             actually seen it.

Then the four timing guards from v1 still apply: baseline subtraction, EMA
smoothing, a sustained hold, and a cooldown plus per-video grace period.
"""

import time

from learning import build_features


class Decider:
    """Scores disengagement, either from the hand-tuned soft OR or, once one has
    been trained on your own scrolling, from a learned model. The gates and
    timing guards below are identical either way - the model replaces only the
    score, never the safety rails."""

    def __init__(self, config, model=None):
        self.cfg = dict(config)
        self.model = model
        self.features = [0.0] * 6
        self.reset()

    # ---------------------------------------------------------------- config

    def update_config(self, patch):
        self.cfg.update(patch)

    def set_sensitivity(self, sensitivity):
        """Map a 0..1 slider onto a threshold. Higher slider = skips more easily."""
        s = max(0.0, min(1.0, float(sensitivity)))
        self.cfg["enter_threshold"] = 0.75 - 0.43 * s
        self.cfg["hold_seconds"] = 1.8 - 1.1 * s
        return self.cfg["enter_threshold"], self.cfg["hold_seconds"]

    # ----------------------------------------------------------------- state

    def reset(self):
        self.neg = 0.0
        self.pos = 0.0
        self.disengagement = 0.0
        self.smoothed_scores = {}
        self.components = {"attention_loss": 0.0, "drowsiness": 0.0, "emotion": 0.0}
        self._above_since = None
        self._cooldown_until = 0.0
        self._grace_until = time.monotonic() + self.cfg.get("grace_seconds", 1.5)
        self._last_face_time = 0.0
        self._blocked_by = None

    def notify_video_changed(self, now=None):
        """New video started - give it a fair chance before judging it."""
        now = now if now is not None else time.monotonic()
        self._above_since = None
        self._grace_until = now + self.cfg.get("grace_seconds", 1.5)

    # -------------------------------------------------------------- emotion

    def _weighted(self, scores, weights):
        return sum(scores.get(k, 0.0) * w for k, w in weights.items())

    def _update_emotion(self, scores):
        alpha = self.cfg.get("smoothing_alpha", 0.3)
        for label, value in scores.items():
            prev = self.smoothed_scores.get(label, value)
            self.smoothed_scores[label] = alpha * value + (1 - alpha) * prev

        raw_neg = self._weighted(self.smoothed_scores, self.cfg["negative_weights"])
        raw_pos = self._weighted(self.smoothed_scores, self.cfg["positive_weights"])

        base = self.cfg.get("calibration", {}).get(
            "baseline_negative", self.cfg.get("baseline_negative", 0.0)
        )
        self.neg = max(0.0, (raw_neg - base) / max(1e-6, 1.0 - base))
        self.pos = raw_pos

    # --------------------------------------------------------------- fusion

    def _fuse(self, attention_loss, drowsiness):
        gains = self.cfg.get("fusion_gains", {})
        parts = {
            "attention_loss": (attention_loss, gains.get("attention", 0.85)),
            "drowsiness": (drowsiness, gains.get("drowsiness", 0.55)),
            "emotion": (self.neg, gains.get("emotion", 0.85)),
        }
        self.components = {k: round(v * g, 3) for k, (v, g) in parts.items()}

        product = 1.0
        for value, gain in parts.values():
            product *= 1.0 - _clamp(value) * _clamp(gain)
        return 1.0 - product

    def _components_from_model(self):
        """Bars for the display: what pushed THIS prediction up."""
        contributions = self.model.contributions(self.features)
        self.components = {
            "attention_loss": round(_clamp(contributions.get("attention_loss", 0.0) / 2.0), 3),
            "drowsiness": round(_clamp(contributions.get("drowsiness", 0.0) / 2.0), 3),
            "emotion": round(_clamp(contributions.get("negative", 0.0) / 2.0), 3),
        }

    # --------------------------------------------------------------- update

    def update(self, emotion_scores, engagement, dwell, now=None):
        """One frame. `engagement` is EngagementTracker.state(), `dwell` a DwellTracker."""
        now = now if now is not None else time.monotonic()
        cfg = self.cfg

        if emotion_scores is not None:
            self._last_face_time = now
            self._update_emotion(emotion_scores)
        else:
            gap = now - self._last_face_time
            if gap > cfg.get("no_face_reset_seconds", 0.8):
                self.neg *= 0.7
                self.pos *= 0.7

        median = getattr(dwell, "median", 8.0)
        self.features = build_features(
            engagement,
            self.neg,
            self.pos,
            dwell.current(now) / max(1e-3, median),
            self.cfg.get("calibration", {}).get("blink_rate_baseline", 18.0),
        )

        if self.model is not None:
            self.disengagement = self.model.predict(self.features)
            self._components_from_model()
        else:
            attention_loss = 1.0 - float(engagement.get("attention", 1.0))
            drowsiness = float(engagement.get("drowsiness", 0.0))
            self.disengagement = self._fuse(attention_loss, drowsiness)

        action = self._evaluate(now, engagement, dwell)
        return self._state(now, engagement, dwell, action)

    def _evaluate(self, now, engagement, dwell):
        cfg = self.cfg
        self._blocked_by = None

        # Absence suppresses. Note this is a *sustained* loss of your face, not a
        # brief one - the tracker reads a short disappearance as you having
        # turned away, which is a reason to skip, not a reason to stop.
        if engagement.get("absent", False):
            self._above_since = None
            self._blocked_by = "you are not at the camera"
            return None

        if now < self._grace_until:
            self._above_since = None
            self._blocked_by = "video just started"
            return None

        if now < self._cooldown_until:
            self._above_since = None
            self._blocked_by = "cooling down"
            return None

        elapsed = dwell.current(now)
        gate = dwell.gate_seconds()
        if elapsed < gate:
            self._above_since = None
            self._blocked_by = f"only {elapsed:.1f}s of {gate:.1f}s watched"
            return None

        threshold = self.effective_threshold(dwell, now)
        if self.disengagement < threshold:
            self._above_since = None
            return None

        if self._above_since is None:
            self._above_since = now
            return None

        held = now - self._above_since
        if held < cfg["hold_seconds"]:
            return None

        self._above_since = None
        self._cooldown_until = now + cfg["cooldown_seconds"]
        return {
            "action": "skip",
            "reason": self.leading_reason(),
            "held": round(held, 2),
            "score": round(self.disengagement, 3),
            "threshold": round(threshold, 3),
            "dwell": round(elapsed, 1),
            "components": dict(self.components),
            "emotion": self.dominant(),
        }

    def effective_threshold(self, dwell, now=None):
        base = self.cfg["enter_threshold"]
        return max(0.15, base - dwell.threshold_relief(now))

    # ------------------------------------------------------------- reporting

    def leading_reason(self):
        if self.model is not None:
            return self.model.leading_reason(self.features)
        if not self.components:
            return "unknown"
        name = max(self.components.items(), key=lambda kv: kv[1])[0]
        return {
            "attention_loss": "looking away",
            "drowsiness": "bored / drowsy",
            "emotion": "negative reaction",
        }.get(name, name)

    def dominant(self):
        if not self.smoothed_scores:
            return "unknown"
        return max(self.smoothed_scores.items(), key=lambda kv: kv[1])[0]

    def _state(self, now, engagement, dwell, action):
        state = {
            "face": bool(not engagement.get("absent", False) and self.smoothed_scores),
            "emotion": self.dominant(),
            "negative": round(self.neg, 3),
            "positive": round(self.pos, 3),
            "disengagement": round(self.disengagement, 3),
            "components": dict(self.components),
            "reason": self.leading_reason(),
            "threshold": round(self.effective_threshold(dwell, now), 3),
            "base_threshold": round(self.cfg["enter_threshold"], 3),
            "holding": round(now - self._above_since, 2) if self._above_since else 0.0,
            "hold_target": round(self.cfg["hold_seconds"], 2),
            "cooling": max(0.0, round(self._cooldown_until - now, 2)),
            "blocked_by": self._blocked_by,
            "scores": {k: round(v, 3) for k, v in self.smoothed_scores.items()},
            "action": action,
        }
        state.update(engagement)
        state.update(dwell.state(now))
        return state


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))
