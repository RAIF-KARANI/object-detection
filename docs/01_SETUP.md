# 01 — Setup (Windows 11 + NVIDIA GPU)

This is the exact setup used for VisionTrack, so anyone can reproduce it.

## Machine used

| Item | Value |
|---|---|
| OS | Windows 11 Home |
| Laptop | ASUS ROG Strix 16 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |
| NVIDIA driver | 621.01 (supports CUDA up to 13.5) |
| Python | 3.12.9 |
| Shell | PowerShell |

### Why Python 3.12 and not 3.11?

The original plan said 3.11, but the machine already had 3.12 and 3.13 installed.
PyTorch (CUDA builds), Ultralytics and OpenCV all fully support 3.12, and nothing in
this project depends on a 3.11-only feature, so 3.12 was used instead of installing
another interpreter. (3.13 was avoided because some libraries added support for it later.)

## 1. Check the prerequisites

```powershell
py -0p            # lists every installed Python version
git --version
nvidia-smi        # shows the GPU, driver version and "CUDA ... Version" (max CUDA the driver supports)
```

## 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

A **virtual environment** is a private folder of packages for this project only, so
its library versions cannot clash with other Python projects.

> If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"), run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call
> `.\.venv\Scripts\python.exe` directly instead of activating.

## 3. Install PyTorch with CUDA

**Important:** a plain `pip install torch` on Windows installs the **CPU-only** build.
The GPU build comes from PyTorch's own package server, chosen with `--index-url`.

How to choose the right command:

1. Go to <https://pytorch.org/get-started/locally/> and select
   *Stable → Windows → Pip → Python → CUDA*.
2. Pick the **newest CUDA build that is ≤ the "CUDA Version" shown by `nvidia-smi`**.
   You do *not* need to install the CUDA Toolkit separately — the PyTorch wheel ships
   its own CUDA runtime; only the NVIDIA driver is needed.

Available at install time: cu126 (torch 2.14.1), cu128, cu130 (2.14.1), cu132 (2.15.0).
The driver supports 13.5, so **cu132** was chosen:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

`torchvision` is installed in the same command because Ultralytics needs it; if it were
later pulled from normal PyPI it could drag in a CPU-only torch and replace the GPU one.

## 4. Install the rest

```powershell
pip install ultralytics opencv-python numpy pandas
```

Or install everything (including the CUDA torch) in one go:

```powershell
pip install -r requirements.txt
```

## 5. What each package is for

| Package | Version | Used for |
|---|---|---|
| `torch` | 2.15.0+cu132 | Deep-learning engine that runs the YOLO network on the GPU |
| `torchvision` | 0.29.2+cu132 | Image operations used internally by Ultralytics |
| `ultralytics` | 8.4.215 | YOLO26n detector and the built-in ByteTrack tracker |
| `opencv-python` | 5.1.0.95 | Reading webcam/video, drawing boxes/lines/text, showing the window |
| `numpy` | 2.5.4 | Array maths — every video frame is a NumPy array |
| `pandas` | 3.1.1 | Writing results to CSV |
| `lap` | 0.5.14 | Assignment solver ByteTrack uses to match tracks to detections (added in Phase 2: `pip install lap`) |

Ultralytics also installs its own dependencies (matplotlib, scipy, polars, psutil,
requests, …). The project code does not import them directly.

## 6. Verify the GPU

```powershell
python check_gpu.py
```

Output on this machine:

```
PyTorch version : 2.15.0+cu132
CUDA build      : 13.2
CUDA available  : True
GPU name        : NVIDIA GeForce RTX 4060 Laptop GPU
GPU memory      : 8.0 GB
Test matmul ran on: cuda:0 | result shape: (1000, 1000)
```

How to read it:

- `+cu132` in the version → this is the CUDA build (a CPU build shows `+cpu`).
- `CUDA available: True` → driver and GPU are usable from Python.
- The test matrix multiplication ran on `cuda:0` → the GPU actually computes, not just "is visible".

If `CUDA available` is `False`, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
