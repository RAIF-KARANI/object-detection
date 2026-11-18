# 03 — How ByteTrack Works

Everything here matches the Ultralytics code installed in this project
(`.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py`, ultralytics 8.4.215).

## 1. Why tracking? Detection alone isn't enough

YOLO looks at each frame **independently**. Frame 10 says "3 people", frame 11 says
"3 people" — but it doesn't know that the person on the left in frame 11 is the *same*
person as in frame 10. To count people crossing a line, or measure how long someone
stays, we need a stable **ID per person** across frames. That's **multi-object
tracking (MOT)**: "tracking-by-detection" = detect in every frame, then link the
detections over time.

## 2. The big picture (one frame)

```
 frame t
   │
   ▼
 YOLO26n ──► detections (box + score), conf ≥ 0.10
   │
   ▼  split by score
 ┌──────────────────────────┐   ┌─────────────────────────────────┐
 │ HIGH: score ≥ 0.25       │   │ LOW: 0.10 < score < 0.25        │
 └────────────┬─────────────┘   └───────────────┬─────────────────┘
              │                                 │
 existing tracks ──(1) Kalman predict──► predicted boxes
              │                                 │
              ▼                                 │
  (2)+(3) STAGE 1: match HIGH dets to           │
          ALL tracks (tracked + lost)           │
          cost = 1 − IoU·score, Hungarian       │
              │ unmatched tracks                │
              ▼                                 ▼
  (4) STAGE 2: match LOW dets to still-unmatched "tracked" tracks
          cost = 1 − IoU, threshold 0.5
              │                              │
   still unmatched tracks           unmatched HIGH dets
     → mark LOST                    → (5) NEW track if score ≥ 0.25
   lost > 30 frames → REMOVED
              │
              ▼
 output: boxes with track IDs
```

## 3. Step 1 — Kalman filter: predict where each track will be

Each track keeps a **state** of 8 numbers (`KalmanFilterXYAH`):

```
(x, y, a, h,  vx, vy, va, vh)
 centre, aspect ratio (w/h), height,  and the velocity of each
```

The Kalman filter assumes **constant velocity** between frames. It does two things:

- **Predict** (before matching): move the box by its velocity.
  Example: centre x = 150 px, vx = +10 px/frame → predicted x = 160 px.
- **Update** (after matching): blend the prediction with the new detection, weighted by
  how uncertain each one is. If the prediction says 160 and the detection says 164,
  and both are about equally trusted, the new estimate is about 162. The filter also
  updates the velocity from this. If a track goes unmatched, it keeps predicting
  (coasting) but grows more uncertain.

Why predict? Comparing new detections to *where people will be*, not where they *were*,
makes matching work even when people move quickly.

## 4. Step 2 — IoU cost matrix

For every (track, detection) pair, compute IoU between the track's **predicted** box and
the detection box (see `02_HOW_YOLO_WORKS.md` for IoU). Turn it into a **cost**
(lower = better match):

```
cost = 1 − IoU
```

With `fuse_score: True` (stage 1 only), the detection's confidence is mixed in:

```
cost = 1 − IoU × score
```

So a confident, well-overlapping detection is "cheaper" to match than a weak one.

## 5. Step 3 — Hungarian algorithm: best one-to-one matching

Given the cost matrix, we want each track paired with **at most one** detection (and
vice versa), with the **smallest total cost**. Greedy "each track grabs its best box"
can fail when two tracks want the same box; the **Hungarian algorithm** (and the
faster Jonker–Volgenant algorithm that Ultralytics uses via the `lap` package,
`lap.lapjv`) solves this optimally. Pairs whose cost is above a threshold are rejected
even if they're the "best" available — no match is better than a bad match.

## 6. Step 4 — Two-stage association (ByteTrack's key idea)

Most trackers throw away low-confidence detections. ByteTrack's insight: a
**low-score box is often a real person who is partly hidden** (behind a shelf, another
shopper, a trolley). Their score drops, but the box is still in the right place.

- **Stage 1:** high-score detections (`≥ track_high_thresh = 0.25`) are matched to
  **all** tracks — active ones and recently lost ones (so a lost person can be
  re-found). Cost = `1 − IoU·score`, accepted if cost ≤ `match_thresh = 0.8`.
- **Stage 2:** low-score detections (`> track_low_thresh = 0.1` and `< 0.25`) are matched
  only to **tracks still active but unmatched** after stage 1. Cost = `1 − IoU` (score
  not mixed in — it would push all costs too high), accepted if cost ≤ **0.5** (i.e.
  IoU ≥ 0.5; this 0.5 is fixed in the code, not in the YAML).

Stage 2 only uses a low box to *continue an existing track* — it never starts a new one
from it. That's how low boxes rescue occluded people without adding false people.

## 7. Step 5 — Creating and deleting tracks

- **New track:** a high-score detection left unmatched after both stages starts a new
  track **only if its score ≥ `new_track_thresh` (0.25)**. It begins **unconfirmed**: it
  must be matched again in the next frame (cost ≤ 0.7) or it's deleted. This filters out
  one-frame false detections. Ultralytics hides unconfirmed tracks from the output.
- **Lost track:** an active track unmatched in both stages is marked **lost**. It keeps
  being predicted by the Kalman filter and can be re-found in stage 1 with its **old ID**.
- **Removed track:** if a track stays lost for more than `track_buffer` (30) frames it is
  removed for good. If that person reappears later, they get a **new** ID.

## 8. Small numeric example

Two existing tracks (after Kalman prediction) and three new detections, boxes as
(x1, y1, x2, y2). Every box is 100 × 200 = 20 000 px².

| | Box | Score |
|---|---|---|
| Track 1 (predicted) | (100, 100, 200, 300) | — |
| Track 2 (predicted) | (300, 100, 400, 300) | — |
| D1 | (110, 105, 210, 305) | 0.90 (high) |
| D2 | (290, 110, 390, 310) | **0.15 (low — person half-hidden)** |
| D3 | (500, 120, 600, 320) | 0.80 (high) |

IoUs:

- T1–D1: overlap x 110→200 = 90, y 105→300 = 195 → 17 550; union 40 000 − 17 550 = 22 450 → **IoU 0.78**
- T2–D2: overlap x 300→390 = 90, y 110→300 = 190 → 17 100; union 22 900 → **IoU 0.75**
- every other pair: no overlap → IoU 0

**Stage 1** (high dets D1, D3; cost = 1 − IoU·score):

```
          D1                      D3
 T1   1 − 0.78·0.90 = 0.30      1.00
 T2   1.00                      1.00
```

Hungarian picks T1↔D1 (cost 0.30 ≤ 0.8 ✓). T2 and D3 are left over (cost 1.00 > 0.8).

**Stage 2** (low det D2 vs leftover T2; cost = 1 − IoU):

```
          D2
 T2   1 − 0.75 = 0.25   ≤ 0.5 ✓ → T2 keeps its ID thanks to a 0.15-score box
```

**New tracks:** D3 is unmatched with score 0.80 ≥ 0.25 → new track ID 3 (unconfirmed
until it matches again next frame).

**Same frame with `conf = 0.50`:** D2 (0.15) is deleted *before* the tracker sees it.
Stage 2 has nothing to work with → T2 is marked **lost**. If the person stays hidden
more than 30 frames, T2 is removed and they come back as a **new ID** — an **ID switch**,
which in a footfall counter can mean the same shopper is counted twice.

## 9. `bytetrack.yaml` thresholds (installed values)

| Setting | Value | Meaning |
|---|---|---|
| `track_high_thresh` | 0.25 | Score splitting "high" (stage 1) from "low" (stage 2) detections |
| `track_low_thresh` | 0.1 | Below this, a detection is ignored completely |
| `new_track_thresh` | 0.25 | Minimum score for an unmatched detection to start a new track |
| `track_buffer` | 30 | Frames a lost track is remembered (30 frames ≈ 1 s at 30 FPS) |
| `match_thresh` | 0.8 | Maximum stage-1 cost (`1 − IoU·score`) to accept a match |
| `fuse_score` | True | Mix detection score into the stage-1 cost |

Hard-coded in `byte_tracker.py`: stage-2 cost limit **0.5**, unconfirmed-track cost limit **0.7**.

## 10. Why the detection `conf` threshold must be low

The detector's `conf` filter runs **before** the tracker. ByteTrack's stage 2 needs boxes
scoring between 0.1 and 0.25. If `conf` is higher than 0.1, those boxes never reach the
tracker:

```
 conf = 0.10  →  tracker receives 0.10 … 1.00  → stage 1 AND stage 2 work
 conf = 0.25  →  tracker receives 0.25 … 1.00  → stage 2 gets nothing
 conf = 0.50  →  tracker receives 0.50 … 1.00  → stage 2 gets nothing, AND fewer stage-1 boxes
```

That's why Ultralytics' own `model.track()` defaults `conf` to **0.1** (in
`ultralytics/engine/model.py`: *"trackers need low-confidence input"*), and why this
project sets `CONF = 0.10` in `config.py`. Phase 4 measures what actually happens to
track counts when `conf` is raised.

## 11. How it maps to our code

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.10, ...)
ids = results[0].boxes.id      # one integer ID per tracked person (None if no tracks)
```

`persist=True` = keep the same tracker object (with its Kalman states and lost-track
memory) between calls. Without it, every call would start a fresh tracker and IDs would
restart each frame.
