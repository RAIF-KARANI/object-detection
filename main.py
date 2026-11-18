"""VisionTrack — real-time people detection (YOLO26n) + tracking (ByteTrack).

Usage:
    python main.py                      # webcam
    python main.py --source sample      # Ultralytics demo clip
    python main.py --source videos/my_clip.mp4
Press "q" in the video window to quit.
"""
import argparse
import time
import urllib.request
from pathlib import Path

import cv2
from ultralytics import YOLO

import config

WINDOW_NAME = "VisionTrack"


def parse_args():
    """Read command-line options (more are added in Phase 3)."""
    parser = argparse.ArgumentParser(description="People detection + tracking")
    parser.add_argument(
        "--source",
        default=config.SOURCE,
        help='"0" for webcam, "sample" for the demo clip, a file path, or an http(s) URL',
    )
    return parser.parse_args()


def resolve_source(source):
    """Turn the --source text into something cv2.VideoCapture can open.

    - a digit like "0"  -> integer webcam index
    - "sample"          -> the Ultralytics demo clip URL
    - an http(s) URL    -> downloaded once into videos/, then the local path is used
    - anything else     -> treated as a file path
    """
    if source.isdigit():
        return int(source)                     # VideoCapture(0) means "webcam number 0"
    if source == "sample":
        source = config.SAMPLE_VIDEO_URL
    if source.startswith(("http://", "https://")):
        local = Path("videos") / Path(source).name
        if not local.exists():                 # download only the first time
            print(f"Downloading {source} -> {local}")
            urllib.request.urlretrieve(source, local)
        return str(local)
    return source


def draw_tracks(frame, boxes):
    """Draw a box and 'ID n' label for every tracked person on the frame."""
    if boxes.id is None:                       # no confirmed tracks in this frame
        return
    # .cpu() moves tensors from GPU memory to normal memory; .tolist() makes plain Python numbers
    for (x1, y1, x2, y2), track_id, conf in zip(
        boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist(), boxes.conf.cpu().tolist()
    ):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))   # pixel coords must be integers
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)       # green box, 2 px thick (colours are B, G, R)
        label = f"ID {track_id}  {conf:.2f}"
        cv2.putText(frame, label, (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    """Open the video source, then detect + track + display frame by frame."""
    args = parse_args()
    source = resolve_source(args.source)

    model = YOLO(config.MODEL)                 # loads (and on first run downloads) yolo26n.pt
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps_smooth = 0.0                           # running average so the FPS number doesn't flicker
    prev_time = time.perf_counter()
    start_time = prev_time
    frame_count = 0                            # for the summary printed at the end
    seen_ids = set()                           # every track ID that appeared at least once

    while True:
        ok, frame = cap.read()                 # frame = H x W x 3 NumPy array (BGR colour order)
        if not ok:                             # end of video file, or camera disconnected
            break

        # Detect + track in one call. persist=True keeps the SAME ByteTrack tracker
        # between calls, so IDs carry over from frame to frame.
        results = model.track(
            frame,
            persist=True,
            tracker=config.TRACKER,
            classes=config.CLASSES,            # people only
            conf=config.CONF,                  # low on purpose, see config.py
            imgsz=config.IMGSZ,
            device=config.DEVICE,
            verbose=False,                     # don't print a log line for every frame
        )
        draw_tracks(frame, results[0].boxes)   # results is a list with one entry per image; we sent one
        frame_count += 1
        if results[0].boxes.id is not None:
            seen_ids.update(results[0].boxes.id.int().cpu().tolist())

        # FPS = 1 / time taken for this whole loop iteration (read + model + drawing)
        now = time.perf_counter()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        # waitKey(1) lets the window refresh and checks the keyboard for 1 ms
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()                              # free the camera / close the file
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Summary: {frame_count} frames in {elapsed:.1f} s (avg {avg_fps:.1f} FPS), "
          f"{len(seen_ids)} unique track IDs: {sorted(seen_ids)}")


if __name__ == "__main__":
    main()
