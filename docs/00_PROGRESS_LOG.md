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
