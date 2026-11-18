"""All VisionTrack settings in one place.

main.py (and later benchmark.py) import these values. Command-line options
can override some of them at run time.
"""

# --- Model ------------------------------------------------------------------
# YOLO26 "nano": the smallest/fastest YOLO26 model (2.57M parameters).
# Ultralytics downloads the weights file automatically on first use.
MODEL = "yolo26n.pt"

# --- Input ------------------------------------------------------------------
# A digit = camera index. On this laptop: 0 = side/room view, 1 = front-facing close-up,
# 2 = NVIDIA Broadcast virtual camera (black unless the Broadcast app is running).
# Can also be "sample", a video file path, or an http(s) URL.
SOURCE = "1"

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

# --- Counting line (my own logic, counter.py) ---------------------------------
# Two points A -> B as FRACTIONS of the frame (x = 0 left .. 1 right, y = 0 top .. 1 bottom),
# so the same setting works for any resolution.
# Default (my choice for the CAVIAR shopping-centre corridor clips): HORIZONTAL line across
# the corridor floor at 60% height, drawn left -> right. People walk toward/away from the camera,
# so they cross it.
# Direction: IN = crossing from the negative to the positive side of A->B.
# For a left -> right line the + side is BELOW it, so IN = moving DOWN = walking TOWARD the camera.
# (Earlier camera test used a vertical line (0.5, 0.0) -> (0.5, 1.0): IN = right -> left on screen.)
# Swap A and B to flip IN/OUT.
LINE_START = (0.0, 0.6)
LINE_END = (1.0, 0.6)

# Number of recent centre points drawn as each person's trail.
TRAIL_LENGTH = 30

# Where CSVs and saved videos go.
OUTPUT_DIR = "outputs"
