# Troubleshooting

Every problem hit while building VisionTrack, in plain words, with the fix.

---

## 1. Python 3.11 was not installed (Phase 1)

**Symptom:** `py -0p` listed only Python 3.13 and 3.12.

**What it means:** the plan asked for 3.11, but that interpreter wasn't on the machine.

**Fix / decision:** used the existing Python 3.12 (`py -3.12 -m venv .venv`). All
project libraries support 3.12. To install 3.11 instead, you would run
`winget install -e --id Python.Python.3.11` and reopen the terminal.

---

## 2. `pip index versions` returned exit code 255 (Phase 1)

**Symptom:** while checking which PyTorch CUDA builds exist, the command ended with
exit code 255.

**What it means:** one of the lookups in the loop failed quietly (its error output was
hidden), so the last command's exit code was non-zero. The four results we needed
(cu126, cu128, cu130, cu132) had all printed correctly.

**Fix:** none needed — it was a harmless failure in an *informational* command, not in
the install. The install itself succeeded afterwards.

---

## 3. ByteTrack needs the `lap` package (Phase 2)

**Symptom (found by reading the code, before it caused an error):**
`ultralytics/trackers/utils/matching.py` does `import lap`, and `lap` was not installed.

**What it means:** `lap` solves the "assignment problem" — pairing each track with at most
one detection at the lowest total cost. Without it, Ultralytics auto-installs it on the
first tracking run.

**Fix:** installed it on purpose and pinned it: `pip install lap` → `lap==0.5.14` in
`requirements.txt`.

---

## 4. Some cameras report a frame rate of −1 (Phase 3)

**Symptom:** the camera probe showed index 2 (NVIDIA Broadcast virtual camera) reporting
`-1` FPS.

**What it means:** OpenCV couldn't read the frame rate. `fps or 30.0` only replaces `0`;
in Python `-1` counts as "true", so −1 would have been used (negative timestamps, broken
saved video).

**Fix:** `if video_fps <= 0: video_fps = 30.0` in `main.py`.

---

## 5. Average FPS looked far too low on the short sample clip (Phase 3)

**Symptom:** summary said 5.2 FPS on the 62-frame sample clip, vs ~56 FPS in Phase 2.

**What it means:** the very first `model.track()` call took 6.2 s (CUDA start-up, loading
GPU kernels, creating the tracker). On a 2-second clip that one-time cost dominates the
average. Not a real slowdown.

**Fix:** the summary now also reports FPS after the first frame (32.5 FPS on this run,
including display and video writing). Benchmarks in Phase 4 skip warm-up frames.

---

## Reference: GPU not detected (`torch.cuda.is_available()` is `False`)

Did not happen on this machine, but these are the usual causes, in order:

1. **CPU-only torch installed** — `python -c "import torch; print(torch.__version__)"`
   shows `+cpu` or no suffix. Fix: `pip uninstall torch torchvision`, then reinstall with
   the `--index-url https://download.pytorch.org/whl/cu1XX` command from `01_SETUP.md`.
2. **Driver too old** for the chosen CUDA build — `nvidia-smi` "CUDA Version" is lower
   than the build (e.g. driver 12.4, build cu132). Fix: update the NVIDIA driver, or pick
   an older build such as cu126.
3. **Laptop running on the integrated Intel GPU only** (e.g. ASUS Armoury Crate "Eco"
   mode disables the NVIDIA GPU). Fix: switch GPU mode to "Standard"/"Optimized".
4. Can't fix quickly → run everything with `--device cpu`; it works, just slower.
