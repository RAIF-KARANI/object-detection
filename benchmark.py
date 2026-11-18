"""VisionTrack experiments: speed (GPU vs CPU) and confidence threshold vs accuracy.

Runs headless (no window), so it measures processing time, not screen drawing.
Accuracy uses the CAVIAR hand-labelled ground truth (videos/ground_truth/*.xml).

Usage:
    python benchmark.py --experiment device     # GPU vs CPU on the first 500 frames
    python benchmark.py --experiment conf       # conf 0.10 / 0.25 / 0.50 on all clips vs ground truth
    python benchmark.py --experiment all
Results are written to outputs/results.csv (rows of the same experiment are replaced).
"""
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

import config
from counter import LineCounter

# CAVIAR corridor clip -> its ground-truth XML file
CLIPS = {
    "ShopAssistant2cor": "csa2gt",
    "ThreePastShop1cor": "c3ps1gt",
    "EnterExitCrossingPaths1cor": "ceecp1gt",
    "WalkByShop1cor": "cwbs1gt",
}
RESULTS_CSV = Path(config.OUTPUT_DIR) / "results.csv"
WARMUP_FRAMES = 10            # first frames include one-time GPU start-up; not timed
DEVICE_CLIP = "ShopAssistant2cor"
CONF_VALUES = [0.10, 0.25, 0.50]


def line_in_pixels(width, height):
    """Convert config's fractional counting line to pixel points for this frame size."""
    a = (config.LINE_START[0] * width, config.LINE_START[1] * height)
    b = (config.LINE_END[0] * width, config.LINE_END[1] * height)
    return a, b


def video_path(clip):
    """Path of a CAVIAR clip in videos/."""
    return f"videos/{clip}.mpg"


def load_ground_truth(clip):
    """Read a CAVIAR XML file -> {frame_number: (list of person IDs, list of (xc, yc) centres)}."""
    root = ET.parse(f"videos/ground_truth/{CLIPS[clip]}.xml").getroot()
    gt = {}
    for frame in root.findall("frame"):
        ids, centres = [], []
        for obj in frame.findall("objectlist/object"):
            box = obj.find("box")
            ids.append(int(obj.get("id")))
            centres.append((float(box.get("xc")), float(box.get("yc"))))   # already box centres
        gt[int(frame.get("number"))] = (ids, centres)
    return gt


def count_ground_truth(clip, width, height, fps):
    """Run MY LineCounter on the human-labelled boxes -> the TRUE counts for this clip.

    Same line, same rules as the real system, so the comparison is like for like.
    Returns (dict of true counts, list of true people-in-view per frame).
    """
    gt = load_ground_truth(clip)
    counter = LineCounter(*line_in_pixels(width, height))
    in_view = []
    for n in range(len(gt)):
        ids, centres = gt[n]
        counter.update(ids, centres, n / fps)
        in_view.append(counter.in_view)
    truth = {"gt_in": counter.in_count, "gt_out": counter.out_count, "gt_people": counter.unique_count,
             "gt_peak_in_view": counter.peak_in_view, "gt_avg_in_view": round(counter.average_in_view, 2)}
    return truth, in_view


def run_tracker(clip, device, conf, max_frames=None):
    """Run YOLO26n + ByteTrack + LineCounter on a clip without a window.

    Returns (dict of measurements, list of people-in-view per frame).
    """
    model = YOLO(config.MODEL)                # fresh model -> fresh tracker (no state from earlier runs)
    cap = cv2.VideoCapture(video_path(clip))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    counter = LineCounter(*line_in_pixels(width, height))

    frame_times, in_view = [], []
    stage_ms = []                             # Ultralytics' own split: preprocess / inference / postprocess
    n = 0
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        results = model.track(frame, persist=True, tracker=config.TRACKER, classes=config.CLASSES,
                              conf=conf, imgsz=config.IMGSZ, device=device, verbose=False)
        boxes = results[0].boxes
        if boxes.id is not None:
            # .cpu() makes Python wait until the GPU has finished, so the timing below is honest
            ids, xyxy = boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist()
        else:
            ids, xyxy = [], []
        frame_times.append(time.perf_counter() - start)
        speed = results[0].speed              # e.g. {'preprocess': 1.2, 'inference': 9.8, 'postprocess': 0.6} ms
        stage_ms.append((speed["preprocess"], speed["inference"], speed["postprocess"]))

        centres = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in xyxy]
        counter.update(ids, centres, n / fps)
        in_view.append(counter.in_view)
        n += 1
    cap.release()

    timed = frame_times[WARMUP_FRAMES:]       # skip warm-up frames
    ms = 1000 * sum(timed) / len(timed)
    stages = stage_ms[WARMUP_FRAMES:]
    pre, inf, post = (sum(s[i] for s in stages) / len(stages) for i in range(3))
    measured = {"frames": n, "ms_per_frame": round(ms, 1), "fps": round(1000 / ms, 1),
                "preprocess_ms": round(pre, 1), "inference_ms": round(inf, 1), "postprocess_ms": round(post, 1),
                # whatever is left: ByteTrack + Python/Ultralytics overhead + copying results
                "tracking_other_ms": round(ms - pre - inf - post, 1),
                "in": counter.in_count, "out": counter.out_count, "unique_ids": counter.unique_count,
                "peak_in_view": counter.peak_in_view, "avg_in_view": round(counter.average_in_view, 2)}
    return measured, in_view


def experiment_device(frames, repeats):
    """Experiment 1: same clip, same settings, GPU vs CPU, repeated (laptop speed varies run to run)."""
    rows = []
    for device in [0, "cpu"]:
        name = "gpu" if device == 0 else "cpu"
        for run in range(1, repeats + 1):
            measured, _ = run_tracker(DEVICE_CLIP, device, config.CONF, max_frames=frames)
            row = {"experiment": "device", "clip": DEVICE_CLIP, "device": name, "run": run,
                   "conf": config.CONF, **measured}
            print(row)
            rows.append(row)
        ms = [r["ms_per_frame"] for r in rows if r["device"] == name]
        print(f"{name}: mean {sum(ms) / len(ms):.1f} ms/frame (min {min(ms)}, max {max(ms)}) over {len(ms)} runs")
    return rows


def experiment_conf():
    """Experiment 2: conf 0.10 / 0.25 / 0.50 on every clip, compared with ground truth."""
    rows = []
    for clip in CLIPS:
        cap = cv2.VideoCapture(video_path(clip))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        truth, gt_in_view = count_ground_truth(clip, width, height, fps)

        for conf in CONF_VALUES:
            measured, in_view = run_tracker(clip, 0, conf)
            # Mean absolute error of "people in view", frame by frame
            pairs = list(zip(in_view, gt_in_view))
            mae = sum(abs(p - g) for p, g in pairs) / len(pairs)
            row = {"experiment": "conf", "clip": clip, "device": "gpu", "conf": conf, **measured, **truth,
                   "in_view_mae": round(mae, 2),
                   "ids_per_person": round(measured["unique_ids"] / truth["gt_people"], 2)}
            print(row)
            rows.append(row)
    return rows


def save_rows(rows, experiment):
    """Write rows to outputs/results.csv, replacing older rows of the same experiment."""
    new = pd.DataFrame(rows)
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        new = pd.concat([old[old["experiment"] != experiment], new], ignore_index=True)
    new.to_csv(RESULTS_CSV, index=False)
    print(f"Saved {len(rows)} '{experiment}' rows to {RESULTS_CSV}")


def main():
    """Parse options and run the chosen experiment(s)."""
    parser = argparse.ArgumentParser(description="VisionTrack experiments")
    parser.add_argument("--experiment", choices=["device", "conf", "all"], default="all")
    parser.add_argument("--frames", type=int, default=500, help="frames used for the GPU vs CPU test")
    parser.add_argument("--repeats", type=int, default=3, help="runs per device in the GPU vs CPU test")
    args = parser.parse_args()
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)

    if args.experiment in ("device", "all"):
        save_rows(experiment_device(args.frames, args.repeats), "device")
    if args.experiment in ("conf", "all"):
        save_rows(experiment_conf(), "conf")


if __name__ == "__main__":
    main()
