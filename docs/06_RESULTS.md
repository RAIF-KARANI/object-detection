# 06 — Results

Every number here comes from a run on my machine (2026-11-18). Raw data:
`outputs/results.csv` (written by `benchmark.py`). Plan and method: `05_EXPERIMENTS.md`.

**Machine:** ASUS ROG Strix 16, NVIDIA RTX 4060 Laptop GPU (8 GB), Intel Core i7-13650HX
(6 performance + 8 efficiency cores, 20 threads), Windows 11, on AC power, "Turbo" plan.
**Software:** Python 3.12.9, torch 2.15.0+cu132, ultralytics 8.4.215, YOLO26n, ByteTrack.
**Data:** CAVIAR shopping-centre corridor clips, 384×288 @ 25 FPS, with ground truth.

---

## Experiment 1 — GPU vs CPU

ShopAssistant2cor, first 500 frames, first 10 frames not timed, `conf = 0.10`,
`imgsz = 640`, quiet machine.

| device | runs | ms per frame | FPS | inference ms | preprocess / postprocess / tracking+other ms |
|---|---|---|---|---|---|
| **GPU (RTX 4060)** | 3 | **11.5 / 11.8 / 11.5** (mean 11.6) | **84.6 – 87.3** | 8.1 – 8.3 | 0.7 / 1.3 – 1.4 / 1.3 – 1.4 |
| **CPU (i7-13650HX)** | 5 | **26.9 / 58.5 / 113.5 / 97.7 / 27.0** (median 58.5) | 8.8 – 37.2 | 24.0 – 104.2 | 1.0 – 2.5 / 0.7 – 2.1 / 1.2 – 4.8 |

(CPU runs 1–3 are in `results.csv`; runs 4–5 were an extra check with clock monitoring,
logged in the progress log.)

Both devices produced **identical counts** in every run (15 track IDs, peak 7, average 5.19
in view) — the device changes speed, not answers.

**Interpretation**

- The GPU is **2.3× faster than the CPU's best run and ~5× faster than its median**, and,
  just as important, **predictable**: 11.5–11.8 ms every time. The CPU varied 4× between
  identical runs. Real-time needs a guaranteed budget: at 25 FPS a frame arrives every
  40 ms; the GPU always fits, the CPU sometimes doesn't.
- **Why the CPU varies:** clock monitoring during two back-to-back CPU runs showed ~100–120%
  of base clock during the slow run (97.7 ms) and ~165% during the fast one (27.0 ms). The
  clock alone (1.65×) doesn't explain a 3.6× gap; this hybrid CPU also moves work between
  fast and efficiency cores, which is the likely rest of the difference (not proven).
- **Where the time goes:** inference is ~70% of the GPU frame time (8.2 of 11.6 ms);
  **ByteTrack + Python overhead is only ~1.3 ms**. Tracking is cheap; the network is the cost.
- 8 ms of GPU inference is slow for a 6.1-GFLOP model on an RTX 4060. Likely reason: with
  one small image at a time, the GPU waits on hundreds of separate per-layer launches from
  Python rather than computing. Exporting to TensorRT/ONNX (layer fusion, FP16) is the
  standard fix — see doc 07. (Not tested here.)
- Earlier attempts gave 17–44 ms (GPU) and 29–115 ms (CPU) because **another `main.py` run
  was using the GPU at the same time** — those runs were discarded (TROUBLESHOOTING #7).

## Experiment 2 — Detection confidence threshold

All four clips, full length, GPU.

| clip | conf | IN (true) | OUT (true) | unique IDs (real people) | IDs per person | avg in view (true) | in-view error (MAE) |
|---|---|---|---|---|---|---|---|
| ShopAssistant2cor | **0.10** | 10 (9) | 4 (4) | 82 (18) | 4.56 | 3.87 (3.23) | 0.98 |
| | 0.25 | 10 (9) | 4 (4) | 78 (18) | 4.33 | 3.50 (3.23) | 0.73 |
| | 0.50 | 8 (9) | 4 (4) | 41 (18) | 2.28 | 2.81 (3.23) | 0.60 |
| ThreePastShop1cor | **0.10** | 5 (5) | 2 (3) | 45 (14) | 3.21 | 5.63 (5.84) | 0.80 |
| | 0.25 | 5 (5) | 1 (3) | 43 (14) | 3.07 | 5.34 (5.84) | 0.87 |
| | 0.50 | 5 (5) | 1 (3) | 29 (14) | 2.07 | 4.21 (5.84) | 1.66 |
| EnterExitCrossingPaths1cor | **0.10** | 1 (1) | 1 (1) | 8 (5) | 1.60 | 4.39 (3.62) | 0.84 |
| | 0.25 | 1 (1) | 1 (1) | 8 (5) | 1.60 | 4.29 (3.62) | 0.80 |
| | 0.50 | 1 (1) | 1 (1) | 7 (5) | 1.40 | 3.44 (3.62) | 0.44 |
| WalkByShop1cor | **0.10** | 4 (5) | 4 (4) | 46 (17) | 2.71 | 3.99 (4.81) | 0.88 |
| | 0.25 | 4 (5) | 4 (4) | 42 (17) | 2.47 | 3.85 (4.81) | 1.00 |
| | 0.50 | 4 (5) | 4 (4) | 31 (17) | 1.82 | 3.30 (4.81) | 1.51 |

Peak in view (system vs true), conf 0.10: 9 vs 7, 9 vs 8, 6 vs 5, 7 vs 7.

**Speed vs conf:** on the runs made while the machine was quiet, `conf` made no difference:
ShopAssistant2cor 11.3 / 11.1 / 10.9 ms, ThreePastShop1cor 11.1 / 11.2 / 11.1 ms. Later runs
slowed to 14.8–43.9 ms with GPU inference time rising, while the machine was in use; that
slowdown is unrelated to `conf` and those timings are not used for conclusions (counts are
unaffected — detections are deterministic, as every repeated run showed).

**Interpretation**

- **Raising `conf` reduces track IDs** (82 → 78 → 41 on ShopAssistant2cor). I expected the
  opposite (removing ByteTrack's low-score boxes → more lost tracks → more new IDs). The
  data shows the bigger effect is different: at 0.50, **many people are never detected at
  all** — average people-in-view falls *below* the truth on three clips (e.g. 4.21 vs
  5.84) — and short-lived tracks from borderline detections never start. Fewer IDs
  partly means *fewer people tracked*, not *better tracking*.
- **ByteTrack's second stage (0.10 → 0.25)** made only a small difference here: IDs fell by
  0–4 per clip, and ThreePastShop1cor lost one true OUT (2 → 1, truth 3). The CAVIAR
  corridor has moderate occlusion; a crowded supermarket entrance would rely on it more.
- **Line counts are robust to `conf`**: people crossing the line (60% down the frame) are
  close and large, so they score high at any threshold. Far-away people — who are above
  the line — are the ones lost at high `conf`.
- **Best threshold depends on the goal:** 0.10 gives the best occupancy on the busier clips
  (MAE 0.80 / 0.88 vs 1.66 / 1.51 at 0.50); 0.50 looks better on the two clips where 0.10
  over-counts (ShopAssistant2cor, EnterExitCrossingPaths1cor) but under-counts the rest.
  **0.10 (the default) is the safest overall choice** and matches what ByteTrack expects.

## Accuracy check (conf 0.10, the default)

| metric | result |
|---|---|
| **Line crossings** | **29 of 32 correct per clip**: IN 20 counted / 20 true, OUT 11 / 12. Errors: +1 IN, −1 IN, −1 OUT (the totals partly cancel; per-clip error is 3 of 32 ≈ **9%**) |
| **People in view** | off by **0.80 – 0.98 people per frame** on average; peak off by 0–2 |
| **Unique people** | **1.6 – 4.6 track IDs per real person** — not usable as a "unique visitors" count without re-identification |

What it means for a footfall product: **line counts and occupancy are usable; "unique
visitors" from track IDs is not.** That matches the earlier visual checks.

## Other runs (visual checks, not benchmarks)

| run | result | where |
|---|---|---|
| Ultralytics sample clip (2 s, crowd) | peak 26, average 23.9 in view, 31 IDs | progress log 3.5, 3.7 |
| Live camera 1 (71.5 s, me only) | IN 4 / OUT 3, 14 IDs for 1 person: close-up camera → jitter, chair false positive | progress log 3.6 |
| CAVIAR clips with `main.py` | same counts as `benchmark.py` | progress log 3.8 |
| Handheld crowd phone video (not stored) | IN/OUT meaningless (camera moves), crowd under-counted | progress log 3.9 |

## Screenshots

Already in the repo:

- `outputs/caviar_contact_sheet.jpg` — 9 annotated frames from the four CAVIAR clips
  (counting line, IN arrow, boxes, IDs, time in frame, trails, counts panel).

Checklist for the report — save each in `outputs/screenshots/` (create the folder):

| # | file name | what to capture | how |
|---|---|---|---|
| 1 | `01_gpu_check.png` | `python check_gpu.py` output showing `CUDA available : True` and the RTX 4060 | screenshot of the terminal |
| 2 | `02_demo_window.png` | the live VisionTrack window on a CAVIAR clip, mid-crossing, panel readable | `python main.py --source videos\ThreePastShop1cor.mpg`, press Win+Shift+S around 45 s |
| 3 | `03_crossing_log.png` | terminal lines `ID … crossed IN/OUT` and the 3-line summary | same run as #2, after it ends |
| 4 | `04_tracks_csv.png` | a per-track CSV opened in Excel / VS Code | open `outputs/tracks_ShopAssistant2cor_*.csv` |
| 5 | `05_tests_pass.png` | `python test_counter.py` → `All 12 tests passed` | terminal |
| 6 | `06_benchmark.png` | `python benchmark.py --experiment device` output (GPU vs CPU) | on an idle machine |
| 7 | `07_results_table.png` | `outputs/results.csv` conf rows (in / gt_in, unique_ids / gt_people, in_view_mae) | open in Excel, freeze the header |
| 8 | `08_failure_case.png` *(optional)* | a jitter or ID-switch moment on CAVIAR to illustrate doc 07 | pause the `--save` video at a crossing pair |

Don't include frames that show my face or the third-party phone video.
