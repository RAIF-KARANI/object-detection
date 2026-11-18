# 04 — Counting Logic (`counter.py`)

This is the part I wrote myself on top of YOLO26n + ByteTrack. It does **not** use
Ultralytics' `solutions.ObjectCounter` or heatmap classes. ByteTrack gives me, every frame,
a list of `(track_id, box)`; `counter.py` turns that into footfall numbers.

```
 every frame:  ByteTrack ──► [(ID 4, box), (ID 7, box), ...]
                                   │  centre of each box
                                   ▼
                     LineCounter.update(ids, centres, timestamp)
                       ├─ which side of the line?  (cross product sign)
                       ├─ did the side flip?        → IN / OUT
                       ├─ already counted?          (one count per ID per direction)
                       ├─ trail                     (last 30 centres)
                       ├─ first/last seen           → time in frame
                       └─ how many boxes now?       → in view / peak / average / unique
                                   ▼
        draw line, arrow, trails, boxes, then the counts panel ── at the end ──► CSV + summary
```

## 1. The counting line

Defined in `config.py` as two points **as fractions of the frame**, so the same
setting works at any resolution:

```python
LINE_START = (0.5, 0.0)   # A: top edge, half-way across
LINE_END   = (0.5, 1.0)   # B: bottom edge, half-way across
```

`main.py` converts them to pixels once it knows the video size
(e.g. 640×480 → A = (320, 0), B = (320, 480)).

**Why vertical (my choice):** the live test camera faces me, so walking left/right past
it is the natural "doorway" movement. The first sample-clip run (section 12) used a
horizontal line `(0.0, 0.5) → (1.0, 0.5)` instead. For my own doorway video I'll place
the line where the doorway is.

## 2. Which side of the line is a point on? — the cross product

For the line from **A** to **B** and a point **P**, take two vectors:
`AB = B − A` and `AP = P − A`. Their 2D cross product is a single number:

```
cross = (Bx − Ax)·(Py − Ay) − (By − Ay)·(Px − Ax)
```

- `cross > 0` → P is on one side   → side **+1**
- `cross < 0` → P is on the other  → side **−1**
- `cross = 0` → P is exactly on the line → side **0**

Geometrically, `|cross|` is the area of the parallelogram spanned by AB and AP; its
**sign** says whether you turn left or right going from AB to AP. We only need the sign —
no square roots, no angles, just 2 multiplications and a subtraction per person per frame.

### Worked example (same numbers as `test_counter.py`)

Line A = (100, 200) → B = (500, 200), so `AB = (400, 0)`.

**P = (300, 150)** (above the line — remember y grows downwards):

```
AP    = (300 − 100, 150 − 200) = (200, −50)
cross = 400·(−50) − 0·200 = −20 000   → side −1
```

**P = (300, 260)** (below the line):

```
AP    = (200, 60)
cross = 400·60 − 0·200 = +24 000      → side +1
```

**P = (300, 200)** → `cross = 400·0 − 0·200 = 0` → on the line.

A diagonal check, line (0, 0) → (4, 4):

```
P = (4, 0):  cross = 4·0 − 4·4 = −16  → −1
P = (0, 4):  cross = 4·4 − 4·0 = +16  → +1
```

## 3. Detecting a crossing: the sign flips

The counter remembers each ID's **last side**. A crossing happens when the side this
frame is different from the last one:

| last side | side now | result |
|---|---|---|
| none yet | ±1 | nothing (first time we see this ID) |
| −1 | −1 | nothing (same side) |
| **−1** | **+1** | **IN** |
| **+1** | **−1** | **OUT** |
| −1 | 0 | nothing yet — exactly on the line, decide next frame |

**Why remember the last *non-zero* side?** If a centre lands exactly on the line (side 0)
and I stored that 0, the next frame would compare +1 with 0 and the crossing would be
missed. Skipping zeros means −1 → 0 → +1 still counts as one IN (tested).

## 4. IN vs OUT direction

**IN = from side −1 to side +1.** Which way that is on screen depends on the order of
A and B. Swapping A and B flips IN and OUT.

- Line drawn **left → right** (horizontal): side +1 is **below** the line → **IN = moving down**.
- Line drawn **top → bottom** (my default, vertical at x = W/2): `AB = (0, H)`, so
  `cross = 0·(Py − 0) − H·(Px − W/2) = −H·(Px − W/2)`. That's positive when `Px < W/2`,
  so side +1 is the **left half** → **IN = moving right → left on screen**.
  (OpenCV doesn't mirror the camera, so on a front-facing camera that's me walking to *my* right.)

So I never have to work this out in my head, `counter.draw()` draws an arrow labelled
**IN** from the middle of the line towards the +1 side. The arrow direction is
`(−dy, dx)` where `(dx, dy) = B − A`: that vector is perpendicular to AB, and its cross
product with AB is `dx² + dy² > 0`, so it always points to the + side.

## 5. Only count between the line's ends

The cross product treats the line as **infinitely long**. Someone walking past the
*end* of a doorway line would otherwise be counted. So a crossing only counts if the
point is alongside the segment, checked with a **dot product** (projection):

```
t = (AP · AB) / |AB|²      t = 0 at A, t = 1 at B
count only if 0 ≤ t ≤ 1
```

Example, A = (100, 200), B = (500, 200), |AB|² = 160 000:
- P = (300, 250): t = (200·400 + 50·0) / 160 000 = **0.5** → inside ✔
- P = (600, 250): t = (500·400) / 160 000 = **1.25** → past B ✘

## 6. Avoiding double counts

Each direction has a Python **set** of IDs already counted (`counted_in`,
`counted_out`). A set can't hold duplicates, and the counter checks
`if track_id not in counted` before counting. So:

- a person whose box centre **jitters** back and forth on the line
  (down, up, down, up …) counts **at most 1 IN and 1 OUT** (tested);
- someone who goes in and later comes back out counts as 1 IN **and** 1 OUT — correct.

**Limitation:** this counts *track IDs*, not people. If ByteTrack loses someone and
gives them a new ID (an **ID switch**), the new ID can be counted again. That's why the
tracker's stability (doc 03) matters for counting accuracy.

## 7. Which point of the person?

I use the **centre of the bounding box**. Simple and stable. An alternative is the
**bottom-centre** (the feet), which matches "where the person is standing" better on a
floor line seen at an angle; see `07_LIMITATIONS_AND_IMPROVEMENTS.md`.

## 8. Trails

Each ID keeps its last `TRAIL_LENGTH = 30` centre points in a
`collections.deque(maxlen=30)`. When the 31st point is added, the oldest is dropped
automatically. `draw()` joins consecutive points with short lines, in the ID's colour,
for IDs visible in the current frame — a quick visual check that tracking is smooth.

## 9. Time in frame

For every ID the counter stores `first_seen` and `last_seen` timestamps (seconds):

```
seconds_in_frame = last_seen − first_seen
```

Where the timestamp comes from:

- **Video file:** `timestamp = frame_index ÷ video_FPS`. So this is exactly
  *frames ÷ FPS*. Example: an ID first seen on frame 30 and last seen on frame 120 of a
  30 FPS video → (120 − 30) ÷ 30 = **3.0 s**.
- **Live camera:** the real clock (`time.perf_counter()`). A camera's *reported* FPS is
  often wrong — this laptop's camera reports 30 FPS but the Phase 2 run delivered 21.7 —
  and using 30 would make every time about 40% too short.

`frames_seen` (how many frames the ID was actually detected) is also saved; if it's much
smaller than `seconds_in_frame × FPS`, the person was hidden for part of that time.

## 10. Whole-frame counts (occupancy)

The line only tells me who walked **through** a point. Retail also asks *"how busy is
this area?"* — so the counter also counts people in the **whole frame**:

| number | how it's computed | question it answers |
|---|---|---|
| **In view** | `len(track_ids)` this frame | How many people are here right now? |
| **Peak** | `max(peak, in_view)` every frame | When was it busiest, and how busy? |
| **Average in view** | sum of `in_view` over all frames ÷ number of frames | How busy is it typically? |
| **Unique** | number of distinct track IDs seen (`len(first_seen)`) | Roughly how many different people passed? |

Details that matter:

- `update()` is called on **every** frame, including frames with nobody in them (empty
  list), so empty frames pull the average down correctly. Example from the test:
  frames with 2, 3, 0, 2 people → average (2 + 3 + 0 + 2) ÷ 4 = **1.75**.
- **In view is the most reliable number**: it only needs a detection in that frame, not a
  stable ID. It *does* include false detections (e.g. my chair in the live test) and it
  misses people fully hidden behind others.
- **Unique is the least reliable**: it counts *track IDs*. Every ID switch adds a fake
  extra person, and every false track (the chair) too. In my 71-second solo camera test
  there were 14 IDs for 1 real person. Better tracking / re-identification fixes this
  (doc 07).
- On screen, Unique is the running total so far; the end-of-run summary shows the final one.
- The counts are drawn on a **dark panel after the boxes**, so ID labels can't cover them
  in a crowded scene (they did in the first version).

## 11. CSV output

Every run writes `outputs/tracks_<source>_<date>_<time>.csv`, one row per track ID:

| column | meaning |
|---|---|
| `track_id` | ByteTrack ID |
| `first_seen_s`, `last_seen_s` | timestamps in seconds |
| `seconds_in_frame` | last − first |
| `frames_seen` | number of frames the ID was present |
| `direction_crossed` | `IN`, `OUT`, `IN+OUT` or empty |

## 12. Tests (`test_counter.py`)

Run `python test_counter.py`. 12 checks on made-up points, all passing:

| test | checks |
|---|---|
| `test_side_of_line` | the worked cross-product examples above |
| `test_crossing_direction` | −1→+1 = IN, +1→−1 = OUT, others none |
| `test_is_within_segment` | t = 0.5 and 0 inside, t = 1.25 outside |
| `test_simple_crossing_counts_in_once` | walking down across → exactly one IN |
| `test_jitter_on_line_is_not_double_counted` | wobbling → 1 IN + 1 OUT max |
| `test_crossing_past_end_of_line_is_ignored` | crossing beyond B → no count |
| `test_point_exactly_on_line_still_counts` | −1 → 0 → +1 → one IN |
| `test_two_people_counted_separately` | two IDs, opposite directions, same frames |
| `test_time_in_frame_and_rows` | 0.0 → 1.0 s, CSV row contents |
| `test_vertical_line_in_is_right_to_left` | default vertical line: left half is +1, right→left = IN |
| `test_trail_keeps_only_last_n_points` | deque keeps only the last N |
| `test_whole_frame_counts` | in view, peak, average (with an empty frame) = 1.75, unique |

## 13. Real runs (sample clip)

`python main.py --source sample --save` — 62 frames, **horizontal** line across the middle
(the default at that time; since changed to vertical):
**IN = 2, OUT = 2** (IDs 7 and 11 IN; IDs 4 and 2 OUT), 31 confirmed track IDs.
Running it twice gave identical counts.

Same clip with the **vertical** line + whole-frame counts:
**IN = 0, OUT = 0**, **peak 26 in view, average 23.9 in view, 31 unique IDs**
(identical on two runs; peak 26 also matches the separate Phase 2 check).
Zero crossings because people in this clip mostly walk up/down the frame, which crosses
a horizontal line but not a vertical one — **where the line goes decides what is measured**.

Accuracy against a hand count is checked in Phase 4 on my own doorway video.
