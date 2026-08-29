# Emotion Scroll

Watches your face through your webcam and skips to the next YouTube Short when
you have stopped engaging with the one you are on. Everything runs locally — no
video, image, or score ever leaves your machine.

It reads four things: where you are looking, how you are blinking, your facial
expression, and how long you have stayed compared with your own normal. Then it
learns which of those actually predict *your* boredom, from your own scrolling.

<details>
<summary>How it got here</summary>

v1 used facial emotion alone. That catches you *disliking* something but misses
boredom — the far more common reason to move on, and one with no facial
expression at all.

v2 added gaze, head pose, blink behaviour and dwell time.

**v2.1 replaced MediaPipe with plain OpenCV.** MediaPipe's face detector reaches
for Metal on Apple Silicon and aborts when the GPU service is not registered —
`Check failed: service_ Service is unavailable`, inside its own code, before any
of this project runs. The replacement stack is CPU-only OpenCV: YuNet for
detection, the LBF 68-point model for landmarks, `solvePnP` for head pose,
classic eye and mouth aspect ratios for blinks and yawns. Nothing in it can
touch the GPU. Everything above the adapter is unchanged.

v3 added the learning pipeline, because no hand-tuned weighting of these signals
can separate "engaged and still" from "bored and still". See
[Learning what boredom looks like on *you*](#learning-what-boredom-looks-like-on-you).

</details>

```
  webcam ──► Python service (127.0.0.1:8765) ──► Chrome extension ──► YouTube Shorts
             YuNet + LBF landmarks + FER+           performs the skip,
             fusion, gates, timing                  reports how long you watched
```

All the thinking happens in Python so you can tune it without touching the
extension. No video, image, or score ever leaves your machine — the socket binds
to localhost only, and the only network calls are the one-time model downloads.

## Quick start

```bash
git clone <your-repo-url> emotion-scroll
cd emotion-scroll/backend

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python download_model.py     # ~92 MB, once
python server.py --probe     # confirm the camera and models work
python calibrate.py          # 25 seconds, required
python server.py
```

Then load `extension/` as an unpacked extension at `chrome://extensions`
(Developer mode → Load unpacked) and open any YouTube Short.

Requires Python 3.9+, Chrome, and a webcam. Built and used on macOS; the backend
is plain OpenCV so Linux and Windows should work, though only macOS camera
permissions are documented below.

## Repository layout

```
backend/
  server.py          the service: camera loop, WebSocket, preview window
  engagement.py      face detection, landmarks, head pose, gaze, blinks
  emotion_engine.py  FER+ emotion classification
  decider.py         fusion, gates, and the timing guards
  dwell.py           learned watch-time median
  learning.py        feature vector and labelled-example recorder
  train.py           fits the boredom model to your own scrolling
  calibrate.py       measures your baselines - run before first use
  test_pipeline.py   100 checks, no camera or browser needed
  config.json        every threshold, weight, and window
extension/
  manifest.json      MV3 extension, YouTube only
  background.js      owns the WebSocket, relays to tabs
  content.js         the heads-up display and the actual scrolling
  popup.html/js      on/off switch, sensitivity, live status
```

---

## The four signals

| signal | what it sees | how |
| --- | --- | --- |
| **looking away** | eyes and head not pointed at the screen | `solvePnP` head pose + pupil position within the eye, both measured against *your* calibrated resting pose |
| **drowsiness** | blinking more than your normal, eyes half-closed, yawning | eye aspect ratio vs. your baseline, PERCLOS, mouth aspect ratio |
| **emotion** | you actively dislike this | FER+ negative blend, above your resting baseline |
| **dwell** | how long you have been here vs. your own median | learned from your manual scrolls |

The first three fuse into one disengagement score. Dwell does not feed the
score — it gates it.

### Why they fuse with a soft OR, not an average

```
disengaged = 1 - (1 - g₁s₁)(1 - g₂s₂)(1 - g₃s₃)
```

An average would drown out a single strong signal: clear disgust on an otherwise
attentive face would come out middling and never fire. A soft OR says "any one of
these is enough on its own, and together they are more". Each gain lives in
`config.json`, and setting one to `0` removes that signal cleanly.

The tuned result, which the test suite pins down:

- looking away, alone → fires
- strong negative emotion, alone → fires
- **drowsiness alone → does not fire.** Half-closed eyes are too easily a dim
  room or a bad camera angle. It needs corroboration.
- moderate looking-away *plus* drowsiness → fires, though neither would alone

### How the blink threshold is set

A blink is defined as the eye aspect ratio dipping a fixed *fraction* below your
own open-eye value, rather than below an absolute number.

That is not fussiness. A fixed textbook threshold fails because the LBF landmark
model does not track the eyelid all the way shut — on a real face the ratio may
only fall by a fifth during a blink, never crossing the line, so a normal 25
seconds of blinking gets counted as zero and drowsiness silently reads 0 forever.
Measuring the closed value from a low percentile of observed frames fails the
other way: blinks are only about 2% of frames, so any percentile low enough to
catch them sits inside open-eye noise, and the detector then reports dozens of
blinks that never happened.

A relative dip is scale-free, so it works whether your eyes are wide or narrow
and whether the landmark fit is crisp or soft. `blink_dip_fraction` in
`config.json` controls it, and calibration prints the exact ratio at which your
blinks will trigger so you can check it against reality.

### Learning what boredom looks like on *you*

Everything above is my guess at what disengagement looks like. The gains were
never measured on anyone. And the honest limitation is that none of these signals
can tell "engaged and still" from "bored and still" — you can be thoroughly bored
while staring straight ahead with a neutral face.

But you label boredom perfectly, hundreds of times an evening, every time you
flick past something after two seconds. So the system records those labels and
can learn from them.

The framing is a real prediction problem: **given how you looked a few seconds
into a video, did you bail on it?**

- scrolled past well before your median watch time → labelled "bored"
- stayed well past your median → labelled "engaged"
- anything in between → discarded, because an ambiguous label is worse than none

Two details that matter. The features come from a window *early* in the video
(0.7–4s), never from the moment you scrolled — the instant before a scroll is
contaminated by you reaching for the trackpad, and a model trained on that would
learn to detect scrolling rather than predict it. And auto-skips are never
recorded, since learning from our own decisions would just teach the model to
agree with itself.

Nothing extra to run. Use it normally, and watch for `[learn]` lines in the
terminal. Then:

```bash
python train.py --stats     # how many examples so far
python train.py             # fit, report, save if it is actually good
```

It holds out 25% of your data, reports AUC against the hand-tuned rules it would
replace, and prints which signals it found to matter for you. **It refuses to
save a model that barely beats guessing, or one that loses to the hand-tuned
rules.** Restart `server.py` to pick up a saved model; it announces which one it
is using at startup.

The learned score replaces only the score. Every gate and timing guard —
absence, dwell, grace, hold, cooldown — applies exactly as before, so a bad
model can be wrong about *when* but cannot make it obnoxious.

Delete `model_weights.json` to go back to the hand-tuned rules; delete
`training_data.jsonl` to start collecting over.

### Losing your face means two different things

This matters more with an OpenCV detector than it would have with MediaPipe,
because turning your head far enough makes the detector lose you — and that is
exactly the moment you most want a skip. So:

- **face gone for under 4 seconds** → you turned away. Counts as looking away,
  and can fire a skip.
- **face gone for longer** → you left the room. Suppresses everything. Skipping
  while you are away just burns through your feed.

### Dwell, which is what stops it being obnoxious

Nothing fires until you have been on a video for a fraction of your own median
watch time. Whatever your face is doing in the first second, you have not really
seen it yet. Past your median, the threshold drops slightly — sitting on a video
far longer than you normally would while looking away is a real signal.

The median is learned only from **manual** scrolls. Counting auto-skips would be
a feedback loop: skip early → median drops → skip earlier. The extension tags
every video change with whether we caused it, and tagged ones are discarded.

Then the four v1 timing guards still apply on top: baseline subtraction, EMA
smoothing, a sustained hold, and a cooldown plus per-video grace period.

---

## Part 1 — Python service

Python 3.9 or newer. Check with `python3 --version`.

If you installed v1 or v2, clear out the old OpenCV and MediaPipe first — this
version needs the **contrib** OpenCV build for `cv2.face`, and having both
installed fails in confusing ways:

```bash
cd path/to/emotion-scroll/backend
source .venv/bin/activate          # if you already made one
pip uninstall -y opencv-python opencv-python-headless mediapipe
```

Fresh install:

```bash
cd path/to/emotion-scroll/backend

python3 -m venv .venv
source .venv/bin/activate          # your prompt should now start with (.venv)

pip install -r requirements.txt
python download_model.py
```

That fetches three models into `backend/models/`: FER+ for emotion (~35 MB), the
LBF landmark model (~56 MB), and YuNet for detection (~230 KB). Once only.

### Check it works on your machine

```bash
python server.py --probe
```

Samples 40 frames and reports what the stack actually sees: how many frames found
a face, your head angles, gaze, eye aspect ratio, which detector is active, and
what the emotion model makes of you. Run it once after installing, and first
whenever something behaves oddly — it separates a camera or lighting problem from
a logic problem in about ten seconds.

### Calibrate — do not skip this

```bash
python calibrate.py
```

Sit normally for 25 seconds, relaxed, looking at the screen as you usually would.
This measures five things that vary too much between people and setups for
defaults to work:

- **resting expression** — emotion models are trained on posed, theatrical faces,
  so a relaxed face scores a surprising amount of "sadness". Uncalibrated, this
  skips nearly everything.
- **head pose centre** — nobody sits square to their webcam, and a laptop camera
  below eye level means a permanent downward pitch that would otherwise read as
  "looking away" forever.
- **gaze centre** — same idea for the eyes.
- **the watching cone** — how much your look angle naturally wanders while you
  are watching. Measured, not assumed: a fixed cone is either too narrow (pose
  jitter reads as looking away while you stare straight ahead) or too wide
  (turning your head goes unnoticed).
- **eye shape** — how wide your eyes are when open, in aspect-ratio terms, which
  then sets the blink threshold. This one is load-bearing: see below.
- **blink rate** — normal is anywhere from 8 to 30 a minute. What matters is a
  rise above *your* normal.

macOS will ask for camera permission on behalf of whatever app runs Python —
Terminal, iTerm, VS Code. If you miss the prompt: System Settings → Privacy &
Security → Camera.

### Run it

```bash
python server.py
```

The preview window shows your face, a bar for each fused signal, the combined
score against the threshold, your blink rate, and your dwell versus your learned
median. When nothing is firing it tells you which gate is holding it back.

| key | does |
| --- | --- |
| `q` | quit |
| `space` | arm / disarm |
| `r` | reset the smoothing state |

Flags: `--no-preview`, `--camera 1`, `--probe`, `--verbose`.

---

## Part 2 — Chrome extension

1. Open `chrome://extensions`.
2. Turn on **Developer mode**, top right.
3. Click **Load unpacked**.
4. Select the `emotion-scroll/extension` folder. The folder, not a file.
5. Pin it with the puzzle-piece icon.

With the service running, the badge reads **on**. Open any YouTube Short and a
panel appears bottom-right with the signal bars, the combined meter, and what it
is currently waiting for.

| shortcut | does |
| --- | --- |
| `alt` + `E` | arm / disarm |
| `alt` + `H` | hide the panel |

Upgrading from v1: remove the old extension and load this folder fresh. Coming
from v2: the extension is unchanged, but reload it anyway to be safe.

---

## Tuning

Everything is in `backend/config.json`; restart `server.py` after editing.

| setting | meaning | try this if… |
| --- | --- | --- |
| `enter_threshold` | combined score needed (0–1) | **skipping too much** → raise to 0.65 |
| `hold_seconds` | how long it must stay there | **skipping too much** → raise to 1.8 |
| `fusion_gains.attention` | weight on looking away | it skips when you glance away → lower to 0.6 |
| `fusion_gains.drowsiness` | weight on blinks/yawns | dim room → lower to 0.35 |
| `fusion_gains.emotion` | weight on facial emotion | set `0` to drop emotion entirely |
| `engagement.pose_smoothing_alpha` | how much the look angle is smoothed | **false "looking away"** → lower to 0.2 |
| `engagement.cone_exit_multiplier` | how much further you must turn to leave the cone | still flickering → raise to 1.6 |
| `engagement.cone_margin` | slack added to your measured cone at calibration | → raise to 1.5 and recalibrate |
| `engagement.gaze_degrees_per_unit` | how much pupil position counts vs. head pose | gaze looks noisy → lower to 12, or 0 to use head pose alone |
| `engagement.blink_dip_fraction` | how far the eye ratio must dip to count as a blink | **0 blinks reported** → lower to 0.12; **absurd blink rate** → raise to 0.25 |
| `engagement.absent_after_seconds` | how long a lost face means you left | you turn away a lot → raise to 6 |
| `engagement.attention_window_seconds` | how long attention is averaged | feels sluggish → lower to 4 |
| `dwell.gate_fraction_of_median` | how much of your normal watch time is protected | interrupts too early → raise to 0.7 |
| `dwell.max_relief` | how much overstaying lowers the bar | set `0` to switch off |
| `calibration.*` | your measured baselines | re-run `calibrate.py` rather than editing by hand |

The popup's sensitivity slider overrides `enter_threshold` and `hold_seconds`
live — use it for quick adjustments, the file for permanent ones.

`backend/dwell_history.json` holds your learned watch times,
`backend/training_data.jsonl` your labelled examples, and
`backend/model_weights.json` a fitted model. Delete any of them to start that
part over.

Sanity check with no camera and no browser:

```bash
python test_pipeline.py               # everything
python test_pipeline.py --no-models   # skips the two large model downloads
```

100 checks: the models load, head pose recovers known angles from projected
geometry, eye and mouth ratios behave, pupil offset tracks a moving pupil, the
blink threshold counts shallow blinks without inventing them on a flat signal,
attention tracking works, dwell learns and ignores auto-skips, the fusion has
the properties above, and every gate and timing guard holds.

---

## When it misbehaves

**Badge never says "on".** The service is not running, or something else holds
port 8765. Click "Reconnect to service" in the popup.

**`cv2.face` does not exist / `createFacemarkLBF` missing.** You have plain
`opencv-python` installed. Uninstall it and reinstall `opencv-contrib-python` —
see the top of Part 1.

**It says you are looking away while you are watching.** Recalibrate first: the
cone is measured from your own watching data, so a calibration taken in a
different position than you actually sit in will be the wrong shape. Then watch
the `look ±x,±y of X,Ydeg` line in the preview — if the numbers wander close to
the cone while you are looking straight at the screen, your landmark fit is
noisy; raise `cone_margin` to 1.5 and recalibrate, or lower
`gaze_degrees_per_unit` toward 0.

**It thinks you are always looking away.** Almost always calibration — re-run it.
`--probe` shows your live head angles; if `yaw`/`pitch` sit far from your
calibrated centres while you are looking at the screen, calibration was taken in
a different position from the one you actually sit in.

**It never fires.** Watch the preview: which bar is moving, and what does the
"waiting" line say? Usually the dwell gate, doing its job. Raise the sensitivity
slider or lower `dwell.gate_fraction_of_median`.

**Drowsiness sits high all the time.** Dim room, most likely — the eye-openness
signal gets unreliable. Lower `fusion_gains.drowsiness`, or add light. Check that
`--probe` reports an eye aspect ratio above about 0.20 with your eyes open.

**Blink rate shows "—".** Expected for the first 20 seconds; it refuses to report
a rate until it has enough history.

**Calibration reports 0 blinks.** Your eyelids move less, in eye-aspect-ratio
terms, than the threshold expects — usually a soft landmark fit in dim light.
Lower `engagement.blink_dip_fraction` to 0.12 and calibrate again. Until that
reads a plausible number, drowsiness contributes nothing; the other two signals
still work.

**The panel updates but nothing scrolls.** YouTube changed its markup. Open
DevTools on the Shorts page — `content.js` logs a warning when all three
navigation strategies fail.

---

## Honest limits

- **Gaze is coarse.** Pupil position is the centroid of the darkest pixels in the
  eye region — good enough for "at the screen or not", nowhere near good enough
  for which part of the screen. Head pose does most of the work; gaze is a nudge.
  If it looks noisy, lower `gaze_degrees_per_unit` and lean on head pose.
- **Head pose comes from a generic 3D face.** The maths is exact — verified by
  projecting known poses and solving them back — but the model face is not your
  face, so absolute angles carry a few degrees of bias. Calibration removes the
  constant part.
- **LBF landmarks are older and weaker than MediaPipe's mesh**, especially at
  steep angles and in poor light. YuNet detection is genuinely good; the landmark
  fit on top of it is the weaker link.
- **Boredom is inferred, not seen.** Looking away and blinking more are
  *correlates* of disengagement, and much better ones than facial expression, but
  you can be riveted while stone-faced.
- **Lighting and camera angle dominate everything.** A laptop camera angled up at
  your chin is the worst case for every signal at once.

## Where to take it next

- **More features for the learner.** Stillness and fidgeting, time of day, and
  how long you have been scrolling are all cheap to add to the feature vector
  and plausibly carry signal the current six do not.
- **Scroll-back-after-skip as a negative label.** If you skip and immediately
  scroll back, that is a labelled false positive going to waste.
- **Instagram Reels and TikTok**: the backend needs no changes. Add the domain to
  `manifest.json` and add strategies to the `strategies` array in `content.js`.
- **Better landmarks without MediaPipe.** If the LBF fit proves to be the weak
  link, an ONNX landmark model would drop straight into `OpenCVFaceAdapter`
  without touching anything else.

## Privacy

The webcam feed is read, analysed in memory, and discarded frame by frame.
Nothing is written to disk except your learned dwell times (a list of numbers)
and your calibration, nothing is sent anywhere, and the WebSocket refuses
connections from outside 127.0.0.1.

## Contributing

`python test_pipeline.py` must pass before anything is merged; CI runs it on
every push along with `pyflakes` and a syntax check of the extension. If you are
adding a signal, add it to `FEATURE_NAMES` in `learning.py` as well as to the
tracker — the recorder and the predictor share one feature builder on purpose,
and any existing `model_weights.json` is automatically refused once the feature
set changes rather than being silently mispredicted.

## License

MIT — see [LICENSE](LICENSE).

The models are downloaded separately and carry their own terms:
[FER+](https://github.com/onnx/models) (MIT),
[LBF landmarks](https://github.com/kurnianggoro/GSOC2017) (BSD),
[YuNet](https://github.com/opencv/opencv_zoo) (Apache 2.0).
