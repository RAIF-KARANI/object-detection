"""VisionTrack — people detection (YOLO26n) + tracking (ByteTrack) + my own line counter.

Usage:
    python main.py                                  # default camera (config.SOURCE)
    python main.py --source 0                       # another camera
    python main.py --source sample --save           # demo clip, also save annotated video
    python main.py --source videos/door.mp4 --device cpu --conf 0.25 --imgsz 640
Press "q" in the video window to quit. A per-track CSV is always written to outputs/.
"""
import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

import config
from counter import LineCounter, colour_for_id

WINDOW_NAME = "VisionTrack"


def parse_args():
    """Read command-line options; defaults come from config.py."""
    parser = argparse.ArgumentParser(description="People detection, tracking and line counting")
    parser.add_argument("--source", default=config.SOURCE,
                        help='camera index ("0", "1"), "sample", a file path, or an http(s) URL')
    parser.add_argument("--device", default=str(config.DEVICE),
                        help='"0" for the first NVIDIA GPU, or "cpu"')
    parser.add_argument("--conf", type=float, default=config.CONF,
                        help="detection confidence threshold (keep low for ByteTrack)")
    parser.add_argument("--imgsz", type=int, default=config.IMGSZ,
                        help="inference image size in pixels")
    parser.add_argument("--save", action="store_true",
                        help="also save the annotated video to outputs/")
    return parser.parse_args()


def resolve_source(source):
    """Turn the --source text into something cv2.VideoCapture can open.

    - a digit like "0"  -> integer camera index
    - "sample"          -> the Ultralytics demo clip URL
    - an http(s) URL    -> downloaded once into videos/, then the local path is used
    - anything else     -> treated as a file path
    """
    if source.isdigit():
        return int(source)                     # VideoCapture(0) means "camera number 0"
    if source == "sample":
        source = config.SAMPLE_VIDEO_URL
    if source.startswith(("http://", "https://")):
        local = Path("videos") / Path(source).name
        if not local.exists():                 # download only the first time
            print(f"Downloading {source} -> {local}")
            urllib.request.urlretrieve(source, local)
        return str(local)
    return source


def draw_boxes(frame, track_ids, boxes_xyxy, counter):
    """Draw each tracked person's box and an 'ID n  t s' label (time in frame)."""
    for track_id, (x1, y1, x2, y2) in zip(track_ids, boxes_xyxy):
        colour = colour_for_id(track_id)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))   # pixel coords must be integers
        cv2.rectangle(frame, p1, p2, colour, 2)
        label = f"ID {track_id}  {counter.seconds_in_frame(track_id):.1f}s"
        cv2.putText(frame, label, (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def main():
    """Open the source, then detect + track + count + display frame by frame."""
    args = parse_args()
    source = resolve_source(args.source)
    device = int(args.device) if args.device.isdigit() else args.device   # "0" -> 0, "cpu" stays
    is_camera = isinstance(source, int)

    model = YOLO(config.MODEL)                 # loads (and on first run downloads) yolo26n.pt
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:                                 # some cameras report 0 or -1
        video_fps = 30.0

    # Convert the line from fractions (config) to pixels for this video's size
    line_start = (config.LINE_START[0] * width, config.LINE_START[1] * height)
    line_end = (config.LINE_END[0] * width, config.LINE_END[1] * height)
    counter = LineCounter(line_start, line_end, config.TRAIL_LENGTH)

    # Output file names: e.g. outputs/camera1_20261118_191500.csv
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)
    name = f"camera{source}" if is_camera else Path(source).stem
    stamp = time.strftime("%Y%m%d_%H%M%S")
    writer = None
    if args.save:
        video_path = out_dir / f"{name}_{stamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")       # codec for .mp4 files
        writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))

    fps_smooth = 0.0                           # running average so the FPS number doesn't flicker
    prev_time = time.perf_counter()
    start_time = prev_time
    frame_index = 0

    while True:
        ok, frame = cap.read()                 # frame = H x W x 3 NumPy array (BGR colour order)
        if not ok:                             # end of video file, or camera disconnected
            break

        # Time of this frame in seconds: video files use frame number / FPS (exact);
        # live cameras use the real clock, because their reported FPS is often wrong.
        timestamp = time.perf_counter() - start_time if is_camera else frame_index / video_fps

        # Detect + track. persist=True keeps the SAME ByteTrack tracker between calls.
        results = model.track(
            frame,
            persist=True,
            tracker=config.TRACKER,
            classes=config.CLASSES,            # people only
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes.id is not None:               # None when no confirmed tracks this frame
            track_ids = boxes.id.int().cpu().tolist()
            boxes_xyxy = boxes.xyxy.cpu().tolist()
        else:
            track_ids, boxes_xyxy = [], []

        # Centre of each box = the point we test against the counting line
        centres = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes_xyxy]
        for track_id, direction in counter.update(track_ids, centres, timestamp):
            print(f"[{timestamp:6.1f}s] ID {track_id} crossed {direction}  "
                  f"(IN={counter.in_count}, OUT={counter.out_count})")

        counter.draw(frame, track_ids)                 # line + IN arrow + trails
        draw_boxes(frame, track_ids, boxes_xyxy, counter)
        counter.draw_stats(frame)                      # counts panel last, so boxes can't cover it

        # FPS = 1 / time taken for this whole loop iteration (read + model + drawing)
        now = time.perf_counter()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if writer is not None:
            writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        frame_index += 1
        if frame_index == 1:
            warm_time = time.perf_counter()    # the 1st frame includes one-time GPU start-up (seconds)
        # waitKey(1) lets the window refresh and checks the keyboard for 1 ms
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()                              # free the camera / close the file
    if writer is not None:
        writer.release()                       # finish writing the .mp4 file
    cv2.destroyAllWindows()

    # Per-track results -> CSV
    csv_path = out_dir / f"tracks_{name}_{stamp}.csv"
    pd.DataFrame(counter.rows()).to_csv(csv_path, index=False)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    avg_fps = frame_index / elapsed if elapsed > 0 else 0.0
    # Speed without the first frame's start-up cost (that's the speed you see while it runs)
    steady_fps = (frame_index - 1) / (end_time - warm_time) if frame_index > 1 else 0.0
    print(f"Summary: {frame_index} frames in {elapsed:.1f} s (avg {avg_fps:.1f} FPS, "
          f"{steady_fps:.1f} FPS after the first frame)")
    print(f"Line counts:  IN={counter.in_count}, OUT={counter.out_count}")
    print(f"Whole frame:  peak {counter.peak_in_view} people in view, "
          f"average {counter.average_in_view:.1f} in view, {counter.unique_count} unique track IDs")
    print(f"Saved per-track CSV: {csv_path}")
    if writer is not None:
        print(f"Saved annotated video: {video_path}")


if __name__ == "__main__":
    main()
