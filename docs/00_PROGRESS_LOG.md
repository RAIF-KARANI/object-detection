# 00 — Progress Log

A running record of every step: what was done, the commands, the result, and what I learned.

---

## Phase 1 — Environment setup (2026-11-18)

### 1.1 Check the machine — ~18:05

- **Commands:** `py -0p`, `git --version`, `nvidia-smi`
- **Result:** Python 3.13.1 (default) and 3.12 installed, **no 3.11**. Git 2.55.0.
  RTX 4060 Laptop GPU (8 GB), driver 621.01, supports CUDA up to 13.5.
- **Decision:** use Python 3.12 (supported by all libraries; avoids an extra install).
- **Learned:** the "CUDA Version" in `nvidia-smi` is the *maximum* CUDA the driver can
  run, not something I need to install. PyTorch brings its own CUDA runtime.

### 1.2 Create the virtual environment

- **Commands:** `py -3.12 -m venv .venv`, `python -m pip install --upgrade pip`
- **Result:** venv on Python 3.12.9, pip 26.3.
- **Learned:** a venv isolates this project's packages; activate with
  `.\.venv\Scripts\Activate.ps1` (or call `.venv\Scripts\python.exe` directly).

### 1.3 Choose and install the PyTorch CUDA build

- **Commands:** `pip index versions torch --index-url https://download.pytorch.org/whl/cu1XX`
  for cu126/cu128/cu130/cu132, then
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132`
- **Result:** installed `torch-2.15.0+cu132`, `torchvision-0.29.2+cu132`.
  (The check loop exited with code 255 — harmless, see TROUBLESHOOTING #2.)
- **Learned:** plain `pip install torch` on Windows gives a CPU-only build. Rule: pick the
  newest CUDA build ≤ the driver's CUDA version. Install torchvision from the same index.

### 1.4 Install the rest of the stack

- **Command:** `pip install ultralytics opencv-python numpy pandas`
- **Result:** ultralytics 8.4.215, opencv-python 5.1.0.95, numpy 2.5.4, pandas 3.1.1.
  torch stayed `+cu132` (not replaced by a CPU build).
- **Learned:** ultralytics pulls in its own helpers (matplotlib, scipy, polars…); we don't import them.

### 1.5 Verify the GPU

- **Command:** `python check_gpu.py`
- **Result:** `CUDA available: True`, `NVIDIA GeForce RTX 4060 Laptop GPU`, 8.0 GB,
  a test matrix multiplication ran on `cuda:0`.
- **Learned:** `is_available()` only proves the GPU is *visible*; running a real tensor
  operation proves it *computes*. GPU work is asynchronous, so `torch.cuda.synchronize()`
  waits for it to finish (this matters later for honest timing in benchmarks).

### 1.6 Project scaffolding

- **Created:** `requirements.txt` (pinned versions + CUDA extra index), a Python
  `.gitignore` (replaced GitHub's ActionScript template), `videos/`, `outputs/`, `docs/`.
- **Learned:** Git doesn't track empty folders, hence the `.gitkeep` files. Ignored: the
  venv, model weights, large videos and output videos; CSVs and screenshots are kept as evidence.

### 1.7 Docs + commit

- **Created:** `docs/01_SETUP.md`, `docs/TROUBLESHOOTING.md`, this log.
- **Commit:** "Phase 1: environment setup (Python 3.12 venv, PyTorch CUDA, deps, docs)".
- The repo already existed (initial GitHub commit), so `git init` was not needed.

---

## Phase 2 — First detection + tracking demo (2026-11-18)

### 2.1 Read the installed tracker code

- **Looked at:** `ultralytics/cfg/trackers/bytetrack.yaml`, `trackers/byte_tracker.py`,
  `trackers/track.py`, `trackers/utils/matching.py`, `engine/model.py`, `cfg/models/26/yolo26.yaml`.
- **Learned:**
  - `model.track()` sets `conf=0.1` by default because trackers need low-confidence boxes.
  - ByteTrack splits boxes at 0.25 (high) / 0.1 (low); stage 2 uses IoU-only cost with a
    fixed 0.5 limit; new tracks need score ≥ 0.25 and must be confirmed in the next frame.
  - The Kalman state is (x, y, aspect, height) + their velocities.
  - YOLO26n: 2.57 M params, 6.1 GFLOPs, `end2end: True` (NMS-free), `reg_max: 1` (no DFL).

### 2.2 `config.py` and `main.py`

- **Created:** `config.py` (all settings), `main.py` (OpenCV frame loop → `model.track(persist=True)`
  → draw boxes + IDs + FPS → `q` to quit). Sources: webcam `0`, `sample`, file path, or URL
  (URLs are downloaded once into `videos/`).
- **Design choice:** own the frame loop (instead of letting Ultralytics read the video) so
  Phase 3's counter can run on every frame; draw boxes ourselves to see where IDs live
  (`results[0].boxes.id`).

### 2.3 Install `lap`

- **Command:** `pip install lap` → `lap 0.5.14`, pinned in `requirements.txt`.
- **Why:** ByteTrack's matching calls `lap.lapjv` (Jonker–Volgenant assignment solver,
  same job as the Hungarian algorithm). Ultralytics would otherwise auto-install it on first track.

### 2.4 First run on the sample clip

- **Command:** `python main.py --source sample`
- **Result:** exit code 0. First run downloaded `yolo26n.pt` (5.3 MB) and
  `videos/solutions_ci_demo.mp4` (0.3 MB). Window showed boxes with IDs and FPS.
- **Headless sanity check** (same settings, no window — *not* the Phase 4 benchmark):
  clip is 640×360 @ 30 FPS, only **62 frames (~2 s)**; up to **26 people** in one frame;
  **31 unique track IDs**; `model.track()` averaged **18.0 ms/frame (≈ 56 FPS)** on the
  RTX 4060, excluding 5 warm-up frames.
- **Learned:** the sample clip is too short for a good benchmark — Phase 4 needs a longer video.

### 2.5 Webcam run

- **Command:** `python main.py`
- **First attempt:** exit code 0 with no output — but a clean exit can't prove frames were
  shown, so `main.py` was given an end-of-run summary line (frames, average FPS, unique IDs).
- **Second attempt result:** `1250 frames in 57.6 s (avg 21.7 FPS), 3 unique track IDs: [1, 6, 8]`
- **Interpretation:**
  - Webcam input works end to end on the GPU.
  - 21.7 FPS end-to-end is lower than the ~56 FPS model speed on the sample clip, so the
    limit is probably the webcam itself (laptop webcams often deliver ≤ 30 FPS, less in dim
    light) rather than the model — to be checked in Phase 4, not assumed.
  - 3 visible IDs in ~1 minute: new IDs appear when a person is lost for more than
    `track_buffer` (30 frames) or a second person/false detection is confirmed.
  - The gaps (no IDs 2–5, 7) are expected: ByteTrack hands out an ID when a candidate
    track starts, and candidates that are never confirmed in the next frame are deleted,
    so their IDs are never shown.
- **Learned:** an exit code only proves "no crash"; print evidence of what the program did.

### 2.6 Docs

- **Created:** `docs/02_HOW_YOLO_WORKS.md`, `docs/03_HOW_BYTETRACK_WORKS.md`.

---

## Phase 3 — My own counting logic (2026-11-18)

### 3.1 Switch to the other camera — ~19:04

- **Commands:** `Get-PnpDevice -Class Camera`, then an OpenCV probe of indices 0–4 and one
  still frame from each working index.
- **Result:** two physical cameras (Kreo Owl Camera, USB2.0 HD UVC WebCam) + the NVIDIA
  Broadcast virtual camera. OpenCV: index 0 = side/room view (used in Phase 2),
  index 1 = front-facing close-up, index 2 = black virtual camera (reports −1 FPS).
- **Decision:** default `SOURCE = "1"` in `config.py`; any index works with `--source N`.
- **Learned:** OpenCV only knows cameras by number, so look at a frame to know which is which.

### 3.2 `counter.py`

- **Created:** pure functions `side_of_line` (cross-product sign), `is_within_segment`
  (dot-product projection), `crossing_direction` (sign flip → IN/OUT), and the
  `LineCounter` class (last side, trails, first/last seen, counted sets, drawing, CSV rows).
- **Design decisions:** store the last *non-zero* side; count only between the line's
  ends; once per ID per direction; timestamps = frame ÷ FPS for files, real clock for cameras.

### 3.3 `main.py` + `config.py`

- **Added:** `--source --device --conf --imgsz --save`; line as fractions of the frame
  (`LINE_START/LINE_END`), `TRAIL_LENGTH`, `OUTPUT_DIR`; console line per crossing;
  per-track CSV every run; annotated MP4 with `--save`.
- **Bug caught before running:** `cap.get(FPS) or 30` doesn't catch −1 (camera index 2
  reports −1) → changed to `if video_fps <= 0`.

### 3.4 Unit checks

- **Command:** `python test_counter.py` → **All 10 tests passed** (cross-product examples,
  flips, segment ends, jitter, exactly-on-line, two people, time in frame, trail length).

### 3.5 Run on the sample clip

- **Command:** `python main.py --source sample --save`
- **Result:** IN = 2 (IDs 7, 11), OUT = 2 (IDs 4, 2), 31 confirmed IDs. Identical on a
  second run. Checked a saved frame: line, IN arrow (pointing down), boxes, IDs,
  time-in-frame labels, trails and totals all drawn correctly.
- **Surprise:** first summary said **5.2 FPS**. Timed it: frame 0 takes **6,202 ms**
  (one-time CUDA/tracker start-up), frame 1 267 ms, then ~45 ms. On a 62-frame clip the
  start-up dominates the average. Summary now also prints FPS **after the first frame**:
  32.5 FPS (including display + video writing).
- **Learned:** always separate warm-up from steady-state when quoting speed (Phase 4 does this properly).

### 3.6 Counting-line decision + live camera test

- **Decision (mine):** vertical line down the middle (`(0.5, 0.0) → (0.5, 1.0)`) so walking
  left/right past camera 1 crosses it. IN = right → left on screen (added a test proving it:
  **11/11 tests pass**).
- **Command:** `python main.py --save` (camera 1, front-facing close-up)
- **Result:** `1906 frames in 71.5 s (avg 26.7 FPS, 28.5 FPS after the first frame),
  14 unique track IDs, IN=4, OUT=3`. CSV: `outputs/tracks_camera1_20261118_191142.csv`.
- **My hand count:** TODO: fill in
- **What the saved video showed (checked frame by frame):**
  - **ID 1 OUT then IN 0.3 s apart:** I stood up right in front of the lens; the box filled
    the frame and its centre wobbled on the line (jitter). The once-per-direction rule capped it.
  - **ID 9 was my chair** — a false "person" tracked for 6.3 s at `conf=0.10`. Not counted
    (never crossed), but a real false positive.
  - **ID 56 OUT then IN:** sitting close, the large box changed shape and its centre
    crossed the middle line without me really walking.
  - IDs 2, 6, 10, 14, 36, 40 lived 1–5 frames: partial bodies at the frame edge / flicker.
- **Learned:** the counter logic is right, but a close-up camera is a bad counting setup:
  huge boxes clipped at the frame edges make centres jump. Real footfall cameras are
  mounted high/overhead and far from the door — film the Phase 4 video from a distance.
  (Frames from this run show my face, so they are not saved in the repo.)
