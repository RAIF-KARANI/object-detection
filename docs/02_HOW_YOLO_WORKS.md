# 02 — How YOLO Works (and what's special about YOLO26)

## 1. What object detection is

**Classification** answers *"what is in this image?"* → `person`.
**Detection** answers *"what is where?"* → a list of boxes, each with a class and a confidence:

```
[ (x1=112, y1=40, x2=210, y2=330, class=person, conf=0.91),
  (x1=402, y1=55, x2=470, y2=310, class=person, conf=0.64) ]
```

`(x1, y1)` is the top-left corner, `(x2, y2)` the bottom-right, in pixels. In OpenCV
images, **y grows downwards** (0 is the top row).

## 2. "You Only Look Once"

Older detectors ran in two stages: first propose regions that *might* contain objects,
then classify each region. YOLO does it in **one pass** of a single neural network over
the whole image: every location predicts boxes and classes at the same time. One pass =
fast enough for real-time video.

## 3. Grid / feature maps

The network (a CNN) shrinks the image step by step while making it "deeper" (more
channels = more learned features: edges → textures → body parts → whole people).

YOLO26 makes predictions on **three feature maps** with strides 8, 16 and 32
(from `ultralytics/cfg/models/26/yolo26.yaml`: outputs P3/8, P4/16, P5/32).
For a 640×640 input:

```
 640x640 image
      │
      ▼
 ┌──────────┐   backbone: extracts features
 │ CNN      │
 └──────────┘
      │  neck: mixes fine detail with big-picture context
      ▼
 ┌──────────────┬──────────────┬──────────────┐
 │ P3: 80 x 80  │ P4: 40 x 40  │ P5: 20 x 20  │   ← three "grids"
 │ stride 8     │ stride 16    │ stride 32    │
 │ small people │ medium       │ large/close  │
 └──────────────┴──────────────┴──────────────┘
      │
      ▼
 head: at EVERY grid cell, predict
   • 4 box numbers  (where is the box relative to this cell?)
   • 80 class scores (one per COCO class: person, car, dog, ...)

 total cells = 80·80 + 40·40 + 20·20 = 6400 + 1600 + 400 = 8400 candidate predictions
```

Each grid cell "looks at" a patch of the image. On the 80×80 grid, one cell covers
8×8 pixels — good for small/far people. On the 20×20 grid a cell covers 32×32 pixels
and sees more context — good for big/close people.

## 4. Bounding boxes

Each cell predicts 4 numbers describing a box: how far the box's left, top, right and
bottom edges are from the cell's centre. These are converted back to pixel
coordinates `(x1, y1, x2, y2)` in the original frame.

YOLO26 predicts these 4 distances **directly** (`reg_max: 1` in the config). Earlier
YOLOs (v8/v11) used "DFL", which predicted a small probability distribution per edge;
dropping it makes the head simpler and easier to export to edge devices.

## 5. Class scores and confidence

For each cell the network outputs 80 numbers squeezed between 0 and 1 (sigmoid), one per
class. The **confidence** of a prediction is the score of its best class.

We only care about people, so `classes=[0]` keeps predictions whose class is `person`
(COCO class 0) and drops the rest.

The **confidence threshold** (`conf`) then removes weak predictions: `conf=0.25` keeps only
boxes scoring ≥ 0.25. We use **0.10** in tracking mode — see `03_HOW_BYTETRACK_WORKS.md`
for why.

## 6. IoU — Intersection over Union

IoU measures how much two boxes overlap, from 0 (no overlap) to 1 (identical):

```
            area of overlap
   IoU = ─────────────────────
          area of both combined
```

```
   (50,50)
     ┌────────────┐
     │     A      │
     │      (100,100)
     │       ┌────┼───────┐
     │       │////│       │
     └───────┼────┘       │
             │  (150,150) │
             │      B     │
             └────────────┘
                       (200,200)
```

**Worked example**

- Box A = (50, 50) → (150, 150) → width 100, height 100, **area 10 000**
- Box B = (100, 100) → (200, 200) → width 100, height 100, **area 10 000**
- Overlap rectangle:
  - left = max(50, 100) = 100, right = min(150, 200) = 150 → width 50
  - top = max(50, 100) = 100, bottom = min(150, 200) = 150 → height 50
  - **intersection = 50 × 50 = 2 500**
- **union** = 10 000 + 10 000 − 2 500 = **17 500** (subtract the overlap so it isn't counted twice)
- **IoU = 2 500 / 17 500 ≈ 0.143**

Only 14% overlap → these are probably two different objects. IoU is used in two places
in this project: removing duplicate detections (NMS, below) and matching detections to
tracks (ByteTrack).

## 7. NMS — Non-Maximum Suppression (the classic way)

Because 8400 cells all predict, one person usually gets **several overlapping boxes**
from neighbouring cells. Classic YOLO (v3–v8, v11) cleans this up after the network with NMS:

1. Sort all boxes by confidence, highest first.
2. Keep the top box.
3. Delete every other box whose IoU with it is above a threshold (e.g. 0.5) — they're duplicates.
4. Repeat with the next remaining box.

```
 before NMS                 after NMS
 ┌───┐┌───┐                 ┌───┐
 │0.9││0.8│  same person    │0.9│
 └───┘└───┘  (IoU 0.8)      └───┘
```

## 8. YOLO26: NMS-free, end-to-end

YOLO26 is **end-to-end** (`end2end: True` in `yolo26.yaml`). During training, its head
is taught a *one-to-one* rule: each real object is assigned to exactly **one** best
prediction, and the others learn to output low scores. So at inference the network
already produces one box per person, and **no NMS step is needed** — Ultralytics just
takes the top-scoring predictions (up to `max_det = 300`) above `conf`.

Why that helps on **edge devices** (small computers inside cameras or in-store boxes):

- NMS is a loop whose work depends on how many boxes there are — slower in crowds,
  so latency is unpredictable. End-to-end gives a fixed, predictable cost.
- NMS is awkward to run on accelerators and is often done on the CPU or left out of
  exported models. Without it, the whole model exports cleanly to ONNX / TensorRT /
  CoreML as one graph.
- Fewer post-processing steps = less code to port and less to go wrong on the device.

## 9. YOLO26n in numbers (from the installed config)

| | YOLO26n |
|---|---|
| Parameters | 2,572,280 (~2.6 M) |
| Compute | 6.1 GFLOPs per 640×640 image |
| Classes | 80 (COCO); we keep only `person` |
| Output grids | 80×80, 40×40, 20×20 → 8400 candidates |
| Post-processing | none (NMS-free) |

"n" = nano, the smallest of the n/s/m/l/x sizes: least accurate, but fastest — the
right trade-off for real-time counting on a laptop or edge device.

## 10. How it maps to our code

```python
model = YOLO("yolo26n.pt")                 # load pretrained COCO weights
results = model.track(frame, classes=[0], conf=0.10, imgsz=640, device=0, ...)
boxes = results[0].boxes                   # boxes.xyxy, boxes.conf, boxes.cls, boxes.id
```

`imgsz=640` → the frame is resized (keeping its shape, padding the rest) so the network
sees 640 px on its longest side; boxes are mapped back to the original frame size.
