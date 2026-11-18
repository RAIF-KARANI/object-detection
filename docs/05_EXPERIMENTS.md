# 05 — Experiment Plan

Two experiments plus an accuracy check, all run by `benchmark.py` and saved to
`outputs/results.csv`. Results (real numbers only) are in `06_RESULTS.md`.

## Test data

**CAVIAR shopping-centre corridor clips** (EC-funded CAVIAR project, IST 2001 37540;
`homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/`). A fixed camera mounted high at the end of a
real shopping-centre corridor in Lisbon — close to how a retail footfall camera is set up.

| clip | frames (25 FPS) | length | real people (ground truth) |
|---|---|---|---|
| ShopAssistant2cor | 3700 | 148 s | 18 |
| ThreePastShop1cor | 1650 | 66 s | 14 |
| EnterExitCrossingPaths1cor | 383 | 15 s | 5 |
| WalkByShop1cor | 2360 | 94 s | 17 |

Resolution 384×288. Each clip comes with a **hand-labelled ground-truth XML**: for every
frame, every person's box and a person ID that stays the same across frames.

Why not my own phone video (the original plan)? My camera only ever saw one person, and
CAVIAR's ground truth is better than counting by hand: it's frame-by-frame and done by
the dataset authors.

## Fixed settings (unless an experiment changes them)

YOLO26n, `imgsz=640`, `classes=[0]` (person), ByteTrack (`bytetrack.yaml` defaults),
`conf=0.10`, counting line across the corridor `(0.0, 0.6) → (1.0, 0.6)`, IN = toward camera.

## How time is measured

- `benchmark.py` runs **headless** (no window, no video writing), so it measures
  processing, not screen drawing.
- The clock covers `model.track()` **plus copying the boxes from the GPU to the CPU**.
  The copy forces Python to wait for the GPU to finish, so GPU time isn't under-counted
  (GPU work is asynchronous).
- Not timed: reading/decoding the video frame (same for every setting).
- The **first 10 frames are skipped** (warm-up): the first frame alone took 6.2 s in
  Phase 3 because of one-time CUDA start-up.
- Ultralytics also reports its own split per frame (`results[0].speed`): preprocess,
  inference, postprocess. Whatever remains of the total = ByteTrack + Python overhead.
- `ms_per_frame` = mean over timed frames; `fps = 1000 / ms_per_frame`.
- **Machine must be quiet** (no other GPU/CPU-heavy programs). Learned the hard way — see
  TROUBLESHOOTING #7.

## Experiment 1 — GPU vs CPU

- **Question:** how much faster is the RTX 4060 laptop GPU than the CPU
  (Intel Core i7-13650HX, 6 performance + 8 efficiency cores)?
- **How:** first 500 frames of ShopAssistant2cor, `--device 0` vs `--device cpu`,
  **3 runs each**, everything else identical. Fresh model (and tracker) for every run.
- **Measured:** ms/frame, FPS, time split (preprocess / inference / postprocess /
  tracking + other), and the counts (to check the device doesn't change the answers).
- **Why it matters:** in-store edge devices often have no large GPU. Real-time needs
  each frame processed faster than the camera delivers it (25 FPS → 40 ms per frame).

## Experiment 2 — Detection confidence threshold

- **Question:** what happens when `conf` is raised from 0.10 to 0.25 and 0.50?
- **How:** all four clips, full length, GPU, `conf` = 0.10 / 0.25 / 0.50.
- **Measured:** FPS, number of unique track IDs, IN/OUT counts, people in view.
- **Expected effect (from how ByteTrack works, doc 03):** ByteTrack's second stage
  rescues partly hidden people using boxes scoring between 0.1 and 0.25. With
  `conf = 0.25` or `0.50` those boxes are deleted *before* the tracker sees them, so a
  person whose score dips (occlusion, far away) loses their track and comes back with a
  new ID. Higher `conf` also removes some false detections. So the expectation is: fewer
  false boxes, but *less stable* tracks — we measure which effect wins.

## Accuracy check (against ground truth)

For each clip, the ground-truth boxes are fed through **my own `LineCounter`** with the
same line and rules. That gives the *true* answer for exactly the same task, so system
and truth are compared like for like.

| metric | system | truth | meaning |
|---|---|---|---|
| IN / OUT | tracker's line counts | counter on ground-truth tracks | footfall accuracy |
| unique IDs vs real people | `unique_ids` | `gt_people` | `ids_per_person` = IDs ÷ people; 1.0 = perfect, higher = ID switches |
| peak / average in view | from tracker | from ground truth | occupancy accuracy |
| `in_view_mae` | — | — | mean over all frames of \|system people-in-view − true people-in-view\| |

The `conf = 0.10` rows (the default setting) are the system's accuracy; the other rows show
how accuracy changes with the threshold.

Known caveats:

- CAVIAR's labellers mark people who are only partly visible and very far away; a ~20 px
  tall person at the end of the corridor is hard for any detector at this resolution.
- Ground-truth and system centres use the same line, but box sizes differ slightly
  (human vs model), so a person walking exactly along the line can be counted by one and
  not the other.
- Four clips is a small sample; results show tendencies, not a definitive accuracy figure.
