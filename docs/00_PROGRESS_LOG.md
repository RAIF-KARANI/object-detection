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
