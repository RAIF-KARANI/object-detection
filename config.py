"""All VisionTrack settings in one place.

main.py (and later benchmark.py) import these values. Command-line options
can override some of them at run time.
"""

# --- Model ------------------------------------------------------------------
# YOLO26 "nano": the smallest/fastest YOLO26 model (2.57M parameters).
# Ultralytics downloads the weights file automatically on first use.
MODEL = "yolo26n.pt"

# --- Input ------------------------------------------------------------------
# "0" = default webcam. Can also be a video file path or an http(s) URL.
SOURCE = "0"

# Ultralytics' short public demo clip (people walking), used when no own video exists.
SAMPLE_VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"

# --- Inference --------------------------------------------------------------
DEVICE = 0       # 0 = first NVIDIA GPU, "cpu" = processor only
IMGSZ = 640      # frames are resized so the longest side is 640 px before entering the network
CLASSES = [0]    # COCO class 0 = "person"; every other class is ignored

# Detection confidence threshold. Kept LOW (0.10) on purpose:
# ByteTrack's second stage uses boxes scoring between track_low_thresh (0.1)
# and track_high_thresh (0.25) to keep tracks alive through partial occlusion.
# If we set conf=0.5 here, those boxes would be thrown away before the tracker sees them.
CONF = 0.10

# --- Tracking ---------------------------------------------------------------
# Ultralytics' built-in ByteTrack configuration file (thresholds explained in docs/03).
TRACKER = "bytetrack.yaml"
